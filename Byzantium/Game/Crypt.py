from __future__ import annotations
import hashlib
import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import Dream
import Sanctum
import Field
import os
import traceback
NAME_MAX = 8
MODE_SIEGE = 'Siege'
MODE_CAMPAIGN = 'campaign'
HEADER_SOULS = 'SOULS'
HEADER_GLYPH = 'GLYPH'
KIND_PURGE = 'PURGE'
KIND_WHISPER = 'WHISPER'
KIND_RALLY = 'RALLY'
KIND_WRATH = 'WRATH'
KIND_DEFECT = 'DEFECT'
KIND_DREAM = 'DREAM'
VEIL_PURGE = 'PURGE'
VEIL_SOULS = 'SOULS'
VEIL_GLYPH = 'GLYPH'

@dataclass(frozen=True)
class Self:
    soul: str = ''
    key: str = ''

    def box(self) -> Dict[str, str]:
        return {'soul': self.soul, 'key': self.key}

@dataclass(frozen=True)
class Soul:
    soul: str = ''
    key: str = ''

    def box(self) -> Dict[str, str]:
        return {'soul': self.soul, 'key': self.key}

@dataclass(frozen=True)
class Baton:
    self: Self = field(default_factory=Self)
    souls: Tuple[Soul, ...] = ()
    genesis: int = 1

    def box(self) -> Dict[str, Any]:
        return {'self': self.self.box(), 'souls': [soul.box() for soul in self.souls], 'genesis': int(self.genesis)}

@dataclass
class Veil:
    purgewindow: Optional[Tuple[int, int]] = None
    soulswindow: Optional[Tuple[int, int]] = None
    glyphwindow: Optional[Tuple[int, int]] = None
    dedupe: List[str] = field(default_factory=list)
    dedupesize: int = 2

    def accepts(self, raw: bytes, lane: str) -> bool:
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            return False
        size = len(raw)
        if size < 8 or size > 65535:
            return False
        if lane == VEIL_PURGE:
            window = self.purgewindow
        elif lane == VEIL_SOULS:
            window = self.soulswindow
        elif lane == VEIL_GLYPH:
            window = self.glyphwindow
        else:
            return False
        if window is None:
            return True
        lo, hi = window
        return int(lo) <= size <= int(hi)

    def seen(self, digest: str) -> bool:
        if self.dedupesize <= 0:
            return False
        return digest in self.dedupe

    def remember(self, digest: str) -> None:
        if self.dedupesize <= 0:
            return
        self.dedupe = [x for x in self.dedupe if x != digest]
        self.dedupe.append(digest)
        self.dedupe = self.dedupe[-max(1, int(self.dedupesize)):]

class Crypt:

    def __init__(self, state: Any=None, sanctum: Any=None, port: int=9000, dream: Any=None):
        self.sanctum = sanctum
        self.dream = self.wakedream(dream)
        self.mode = self.coercemode(state)
        self.skeleton = self.coerceskeleton(state)
        self.genesisnumber = self.coercegenesis(state)
        self.gate = self.coercegate(state, port)
        self.self = self.coerceself(state)
        self.souls = self.coercesouls(getattr(state, 'souls', []) if not isinstance(state, dict) else state.get('souls', []))
        self.veil = Veil()
        self.genesisdone = False
        self.complete: Tuple[Soul, ...] = ()
        self.state = Baton(self=self.self, souls=tuple(), genesis=int(self.genesisnumber))
        self.glyph = None
        self.bindhost = ''
        self.bindport: Optional[int] = None
        self.sock = self.bindtransport()
        self.sock.setblocking(False)
        self.live = False
        self.thread: Optional[threading.Thread] = None
        self.listensleep = 0.02
        self.souls = self.normalizeSouls(self.souls)
        self.state = self.buildstate()
        self.start()
        self.emitSouls()
        self.Genesis()

    def start(self):
        if self.live:
            return
        self.live = True
        self.thread = threading.Thread(target=self.listen, name='CryptListen', daemon=True)
        self.thread.start()

    def stop(self):
        self.live = False

    def listen(self):
        while self.live:
            try:
                self.poll()
            except Exception:
                pass
            time.sleep(self.listensleep)

    def wake(self):
        self.tick()
        return self.state

    def awake(self):
        return self.wake()

    def tick(self):
        self.poll()
        return self.state

    def close(self):
        self.live = False
        try:
            self.sock.close()
        except Exception:
            pass

    def bindtransport(self) -> socket.socket:
        if self.mode == MODE_SIEGE:
            return self.bindsiege()
        if self.mode == MODE_CAMPAIGN:
            return self.bindcampaign()
        raise ValueError(f'unsupported mode: {self.mode!r}')

    def bindsiege(self) -> socket.socket:
        last_error: Optional[Exception] = None
        seats = self.seatports()
        for port in seats:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(('127.0.0.1', port))
                self.bindhost = '127.0.0.1'
                self.bindport = port
                return sock
            except OSError as exc:
                last_error = exc
                try:
                    sock.close()
                except Exception:
                    pass
            finally:
                pass
        raise RuntimeError(f'No clean siege seat available in reserved range {seats}. Start the next node on a free reserved port or widen the Genesis range.') from last_error

    def bindcampaign(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('', self.gate))
        except OSError:
            sock.bind(('0.0.0.0', self.gate))
        self.bindhost = '0.0.0.0'
        self.bindport = self.gate
        return sock

    def seatports(self) -> List[int]:
        count = max(1, int(self.genesisnumber or 1))
        return [self.gate + i for i in range(count)]

    def siegepeers(self) -> List[Tuple[str, int]]:
        peers = []
        for port in self.seatports():
            if self.bindport is not None and int(port) == int(self.bindport):
                continue
            peers.append(('127.0.0.1', port))
        return peers

    def campaignpeers(self) -> List[Tuple[str, int]]:
        return [(self.broadcasttarget(), self.gate)]

    def peers(self) -> List[Tuple[str, int]]:
        if self.mode == MODE_SIEGE:
            return self.siegepeers()
        if self.mode == MODE_CAMPAIGN:
            return self.campaignpeers()
        return []

    def broadcasttarget(self) -> str:
        ip = self.localip()
        parts = ip.split('.')
        if len(parts) == 4:
            parts[-1] = '255'
            return '.'.join(parts)
        return '255.255.255.255'

    def localip(self) -> str:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(('8.8.8.8', 80))
            ip = probe.getsockname()[0]
            probe.close()
            return ip
        except Exception:
            return '127.0.0.1'

    def poll(self):
        while True:
            try:
                raw, addr = self.sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                self.receive(raw, addr)
            except Exception as exc:
                continue

    def receive(self, raw: bytes, addr: Tuple[str, int]):
        if not self.veil.accepts(raw, VEIL_GLYPH):
            return
        self.Cryptkeeper(raw, addr)

    def Cryptkeeper(self, raw: bytes, addr: Tuple[str, int]):
        packet = self.decrypt(raw)
        header = self.headerof(packet)
        if header == HEADER_SOULS:
            if not self.veil.accepts(raw, VEIL_SOULS):
                return
        elif header == HEADER_GLYPH:
            kind = self.kindof(packet)
            if kind == KIND_PURGE:
                if not self.veil.accepts(raw, VEIL_PURGE):
                    return
            elif not self.veil.accepts(raw, VEIL_GLYPH):
                return
        else:
            return
        digest = self.packethash(packet)
        if self.veil.seen(digest):
            return
        self.veil.remember(digest)
        if header == HEADER_SOULS:
            self.SoulSqueeze(packet, addr)
            return
        self.routeGlyph(packet, addr)

    def SoulSqueeze(self, packet: Dict[str, Any], addr: Tuple[str, int]):
        if self.genesisdone:
            self.emitCompleteSouls()
            return
        incoming = packet.get('souls', [])
        before = self.rosterhash(self.souls)
        merged = self.normalizeSouls(list(self.souls) + list(incoming or []))
        after = self.rosterhash(merged)
        if after != before:
            self.souls = merged
            self.state = self.buildstate()
            self.emitSouls()
        self.Genesis()

    def normalizeSouls(self, values: Iterable[Any]) -> List[Soul]:
        cards: Dict[str, Soul] = {}
        for value in list(values or []):
            soul = self.coercesoul(value)
            if not self.acceptsoulcard(soul):
                continue
            cards[soul.key] = Soul(soul=soul.soul, key=soul.key)
        if self.acceptself(self.self):
            cards[self.self.key] = Soul(soul=self.self.soul, key=self.self.key)
        return sorted(cards.values(), key=lambda soul: soul.key)

    def Genesis(self, state: Any=None):
        if state is not None:
            self.self = self.coerceself(state)
            self.souls = self.normalizeSouls(list(self.souls) + list(self.coercesouls(getattr(state, 'souls', []) if not isinstance(state, dict) else state.get('souls', []))))
            self.genesisnumber = self.coercegenesis(state)
            self.state = self.buildstate()
        if self.genesisdone:
            return self.state
        if len(self.souls) < max(1, int(self.genesisnumber or 1)):
            return self.state
        self.genesisdone = True
        self.complete = tuple(self.normalizeSouls(self.souls))
        self.state = self.buildstate()
        self.emitCompleteSouls()
        sanctum = self.wakesanctum()
        if hasattr(sanctum, 'genesis'):
            return sanctum.genesis(self.state)
        if hasattr(sanctum, 'Genesis'):
            return sanctum.Genesis(self.state)
        return self.state

    def unboxglyph(self, payload: Dict[str, Any]) -> Any:
        kind = str(payload.get('kind', '') or '').strip().upper()
        if kind == KIND_DREAM:
            selfraw = payload.get('self', ['', '']) or ['', '']
            if isinstance(selfraw, dict):
                selfpair = (str(selfraw.get('soul', '') or ''), str(selfraw.get('key', selfraw.get('pubkey', '')) or '').strip())
            else:
                selfpair = (str(selfraw[0] if len(selfraw) > 0 else '' or ''), str(selfraw[1] if len(selfraw) > 1 else '' or '').strip())
            cells = []
            for item in tuple(payload.get('cells', ()) or ()):
                purge = dict(item.get('purge', {}) or {})
                lock = dict(item.get('lock', {}) or {})
                cells.append(Field.Cell(soul=str(item.get('soul', '') or ''), key=str(item.get('key', '') or '').strip(), salt=int(item.get('salt', 0) or 0), purge=Field.Purge(chainbit=int(purge.get('chainbit', 0) or 0), lockbit=int(purge.get('lockbit', 0) or 0)), lock=Field.Lock(parent=str(lock.get('parent', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX), child=str(lock.get('child', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX)), sign=str(item.get('sign', Field.NULL_SIGN_HEX) or Field.NULL_SIGN_HEX)))
            return Field.State(cells=tuple(cells), self=selfpair, monument=tuple(payload.get('monument', ()) or ()))
        if 'saltbody' not in payload or 'lockbody' not in payload or 'textbody' not in payload:
            return payload
        saltbody = tuple((Field.Salt(key=str(item.get('key', '') or ''), salt=int(item.get('salt', 0) or 0)) for item in tuple(payload.get('saltbody', ()) or ())))
        lockraw = dict(payload.get('lockbody', {}) or {})
        lockbody = Field.Lock(parent=str(lockraw.get('parent', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX), child=str(lockraw.get('child', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX))
        textraw = dict(payload.get('textbody', {}) or {})
        textbody = Field.Text(text=str(textraw.get('text', '') or ''))
        return Field.SaltGlyph(key=str(payload.get('key', '') or ''), saltbody=saltbody, lockbody=lockbody, textbody=textbody, salthash=str(payload.get('salthash', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX), lockhash=str(payload.get('lockhash', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX), texthash=str(payload.get('texthash', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX), sign=str(payload.get('sign', Field.NULL_SIGN_HEX) or Field.NULL_SIGN_HEX), locksign=str(payload.get('locksign', Field.NULL_SIGN_HEX) or Field.NULL_SIGN_HEX))

    def routeGlyph(self, packet: Dict[str, Any], addr: Tuple[str, int]):
        if not self.genesisdone:
            return
        payload = dict(packet)
        payload.pop('header', None)
        payload = self.unboxglyph(payload)
        self.glyph = payload
        lane = getattr(getattr(self.dream, 'box', None), 'crypt', None)
        if lane is not None and hasattr(lane, 'glyph'):
            lane.glyph = payload
            if hasattr(self.dream, 'wake'):
                self.dream.wake()
            return
        if hasattr(self.dream, 'box'):
            self.dream.box.crypt = payload
            if hasattr(self.dream, 'wake'):
                self.dream.wake()
        else:
            pass

    def emitSouls(self):
        payload = {'header': HEADER_SOULS, 'souls': [soul.box() for soul in self.normalizeSouls(self.souls)]}
        self.emit(payload)

    def emitCompleteSouls(self):
        payload = {'header': HEADER_SOULS, 'souls': [soul.box() for soul in self.complete or tuple(self.normalizeSouls(self.souls))]}
        self.emit(payload)

    def emitGlyph(self, glyph: Dict[str, Any]):
        payload = {'header': HEADER_GLYPH}
        payload.update(dict(glyph or {}))
        self.emit(payload)

    def emit(self, packet: Dict[str, Any]):
        raw = self.encrypt(packet)
        peers = self.peers()
        for host, port in peers:
            try:
                self.sock.sendto(raw, (host, port))
            except Exception as exc:
                pass

    def buildstate(self) -> Baton:
        souls = tuple(self.complete) if self.genesisdone else tuple(self.normalizeSouls(self.souls))
        return Baton(self=Self(soul=self.self.soul, key=self.self.key), souls=souls, genesis=int(self.genesisnumber))

    def wakesanctum(self):
        if self.sanctum is None:
            self.sanctum = Sanctum.Sanctum()
            return self.sanctum
        if isinstance(self.sanctum, type):
            self.sanctum = self.sanctum()
            return self.sanctum
        return self.sanctum

    def wakedream(self, dream: Any):
        if dream is None:
            return getattr(Dream, 'dream', None) or Dream.Dream()
        if isinstance(dream, type):
            return dream()
        return dream

    def headerof(self, packet: Dict[str, Any]) -> str:
        return str(packet.get('header', '') or '').strip().upper()

    def kindof(self, packet: Dict[str, Any]) -> str:
        return str(packet.get('kind', '') or '').strip().upper()

    def packethash(self, packet: Dict[str, Any]) -> str:
        body = json.dumps(packet, sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(body.encode('utf-8')).hexdigest()

    def rosterhash(self, souls: Iterable[Any]) -> str:
        body = json.dumps([soul.box() for soul in self.normalizeSouls(souls)], sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(body.encode('utf-8')).hexdigest()

    def encrypt(self, packet: Dict[str, Any]) -> bytes:
        body = json.dumps(packet, sort_keys=True, separators=(',', ':'), default=str)
        data = body.encode('utf-8')
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        return bytes((byte ^ mask[i % len(mask)] for i, byte in enumerate(data)))

    def decrypt(self, raw: bytes) -> Dict[str, Any]:
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        data = bytes((byte ^ mask[i % len(mask)] for i, byte in enumerate(raw)))
        return json.loads(data.decode('utf-8'))

    def coercemode(self, value: Any) -> str:
        if isinstance(value, dict):
            raw = value.get('mode', MODE_SIEGE)
        else:
            raw = getattr(value, 'mode', MODE_SIEGE)
        text = str(raw or MODE_SIEGE).strip().lower()
        return text if text in (MODE_SIEGE, MODE_CAMPAIGN) else MODE_SIEGE

    def coerceskeleton(self, value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get('skeleton', '') or '')
        return str(getattr(value, 'skeleton', '') or '')

    def coercegenesis(self, value: Any) -> int:
        if isinstance(value, dict):
            raw = value.get('genesis', 1)
        else:
            raw = getattr(value, 'genesis', 1)
        try:
            return max(1, int(raw or 1))
        except Exception:
            return 1

    def coercegate(self, value: Any, fallback: int) -> int:
        if isinstance(value, dict):
            raw = value.get('gate', value.get('port', fallback))
        else:
            raw = getattr(value, 'gate', getattr(value, 'port', fallback))
        try:
            return int(raw or fallback)
        except Exception:
            return int(fallback)

    def coerceself(self, value: Any) -> Self:
        if isinstance(value, dict):
            if 'self' in value:
                return self.coerceself(value.get('self'))
            return Self(soul=self.cleansoul(value.get('soul', '')), key=self.cleankey(value.get('key', value.get('pubkey', ''))))
        if isinstance(value, Self):
            return Self(soul=value.soul, key=value.key)
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return Self(soul=self.cleansoul(value[0]), key=self.cleankey(value[1]))
        if hasattr(value, 'self'):
            return self.coerceself(getattr(value, 'self'))
        return Self(soul=self.cleansoul(getattr(value, 'soul', '')), key=self.cleankey(getattr(value, 'key', getattr(value, 'pubkey', ''))))

    def coercesouls(self, values: Iterable[Any]) -> List[Soul]:
        out: List[Soul] = []
        for value in list(values or []):
            soul = self.coercesoul(value)
            if self.acceptsoulcard(soul):
                out.append(soul)
        else:
            pass
        return out

    def coercesoul(self, value: Any) -> Soul:
        if isinstance(value, Soul):
            return Soul(soul=value.soul, key=value.key)
        if isinstance(value, dict):
            return Soul(soul=self.cleansoul(value.get('soul', '')), key=self.cleankey(value.get('key', value.get('pubkey', ''))))
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            return Soul(soul=self.cleansoul(value[0]), key=self.cleankey(value[1]))
        return Soul(soul=self.cleansoul(getattr(value, 'soul', '')), key=self.cleankey(getattr(value, 'key', getattr(value, 'pubkey', ''))))

    def acceptsoulname(self, value: str) -> bool:
        return isinstance(value, str) and 0 < len(value) <= NAME_MAX

    def acceptkey(self, value: str) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            bytes.fromhex(value)
        except ValueError:
            return False
        return True

    def acceptself(self, value: Self) -> bool:
        return self.acceptsoulname(value.soul) and self.acceptkey(value.key)

    def acceptsoulcard(self, value: Soul) -> bool:
        return self.acceptsoulname(value.soul) and self.acceptkey(value.key)

    def cleansoul(self, value: Any) -> str:
        return str(value or '').strip()[:NAME_MAX]

    def cleankey(self, value: Any) -> str:
        return str(value or '').strip()
__all__ = ['Crypt', 'Self', 'Soul', 'Baton', 'Veil', 'MODE_SIEGE', 'MODE_CAMPAIGN', 'HEADER_SOULS', 'HEADER_GLYPH', 'KIND_PURGE', 'KIND_WHISPER', 'KIND_RALLY', 'KIND_WRATH', 'KIND_DEFECT', 'KIND_DREAM', 'VEIL_PURGE', 'VEIL_SOULS', 'VEIL_GLYPH']
