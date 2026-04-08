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

NAME_MAX = 8

MODE_SIEGE = 'Siege'
MODE_CAMPAIGN = 'Campaign'

HEADER_SOULS = 'SOULS'
HEADER_GLYPH = 'GLYPH'

KIND_PURGE = 'PURGE'
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
        return {
            'self': self.self.box(),
            'souls': [soul.box() for soul in self.souls],
            'genesis': int(self.genesis),
        }

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

        if isinstance(state, dict):
            mode = str(state.get('mode', MODE_SIEGE) or MODE_SIEGE).strip()
            skeleton = str(state.get('skeleton', '') or '')
            genesis = int(state.get('genesis', 1) or 1)
            gate = int(state.get('gate', state.get('port', port)) or port)
            nested = state.get('self', state)
            selfcard = Self(
                soul=self.mustname(nested.get('soul', '')),
                key=self.mustkey(nested.get('key', nested.get('pubkey', ''))),
            )
            rawsouls = state.get('souls', [])
        else:
            mode = str(getattr(state, 'mode', MODE_SIEGE) or MODE_SIEGE).strip()
            skeleton = str(getattr(state, 'skeleton', '') or '')
            genesis = int(getattr(state, 'genesis', 1) or 1)
            gate = int(getattr(state, 'gate', getattr(state, 'port', port)) or port)
            nested = getattr(state, 'self', state)
            selfcard = Self(
                soul=self.mustname(getattr(nested, 'soul', '')),
                key=self.mustkey(getattr(nested, 'key', getattr(nested, 'pubkey', ''))),
            )
            rawsouls = getattr(state, 'souls', [])

        self.mode = mode if mode in (MODE_SIEGE, MODE_CAMPAIGN) else MODE_SIEGE
        self.skeleton = skeleton
        self.genesisnumber = max(1, genesis)
        self.gate = gate
        self.self = selfcard
        self.souls = self.normalizeSouls(rawsouls)

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

        self.state = self.buildstate()
        self.start()
        self.emitSouls()
        self.Genesis(state)

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
            except Exception as exc:
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
        for port in self.seatports():
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
        raise RuntimeError(f'No clean siege seat available in reserved range {self.seatports()}.') from last_error

    def bindcampaign(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.bind(('', self.gate))
        except OSError:
            sock.bind(('0.0.0.0', self.gate))
        self.bindhost = '0.0.0.0'
        self.bindport = self.gate
        return sock

    def seatports(self) -> List[int]:
        return [self.gate + i for i in range(max(1, int(self.genesisnumber or 1)))]

    def siegepeers(self) -> List[Tuple[str, int]]:
        return [('127.0.0.1', port) for port in self.seatports() if int(port) != int(self.bindport or -1)]

    def campaignpeers(self) -> List[Tuple[str, int]]:
        return [(self.broadcasttarget(), self.gate)]

    def peers(self) -> List[Tuple[str, int]]:
        if self.mode == MODE_SIEGE:
            return self.siegepeers()
        if self.mode == MODE_CAMPAIGN:
            return self.campaignpeers()
        return []

    def broadcasttarget(self) -> str:
        parts = self.localip().split('.')
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
            except Exception as exc:
                break
            try:
                self.receive(raw, addr)
            except Exception as exc:
                continue

    def receive(self, raw: bytes, addr: Tuple[str, int]):
        accepted = self.veil.accepts(raw, VEIL_GLYPH)
        if not accepted:
            return
        self.Cryptkeeper(raw, addr)

    def Cryptkeeper(self, raw: bytes, addr: Tuple[str, int]):
        try:
            packet = self.decrypt(raw)
        except Exception as exc:
            raise
        header = self.headerof(packet)

        if header == HEADER_SOULS:
            accepted = self.veil.accepts(raw, VEIL_SOULS)
            if not accepted:
                return
        elif header == HEADER_GLYPH:
            kind = self.kindof(packet)
            if kind == KIND_PURGE:
                accepted = self.veil.accepts(raw, VEIL_PURGE)
                if not accepted:
                    return
            elif True:
                accepted = self.veil.accepts(raw, VEIL_GLYPH)
                if not accepted:
                    return
        else:
            return

        skipdedupe = 100 <= len(raw) <= 125
        if not skipdedupe:
            digest = self.packethash(packet)
            seen = self.veil.seen(digest)
            if seen:
                return
            self.veil.remember(digest)

        if header == HEADER_SOULS:
            self.SoulSqueeze(packet, addr)
            return

        self.routeGlyph(packet, addr)

    def SoulSqueeze(self, packet: Dict[str, Any], addr: Tuple[str, int]):
        incomingpacket = self.packetSouls(packet.get('souls', []))

        if self.genesisdone:
            mine = tuple(self.complete or tuple(self.normalizeSouls(self.souls)))
            incominghash = self.packetRosterhash(incomingpacket)
            myhash = self.packetRosterhash(mine)
            if incominghash != myhash:
                self.emitCompleteSouls()
            return

        before = self.rosterhash(self.souls)
        merged = self.normalizeSouls(list(self.souls) + list(incomingpacket or []))
        after = self.rosterhash(merged)
        if after != before:
            self.souls = merged
            self.state = self.buildstate()
            self.emitSouls()
        self.Genesis()

    def normalizeSouls(self, values: Iterable[Any]) -> List[Soul]:
        cards: Dict[str, Soul] = {}
        for value in list(values or []):
            soul = self.soulcard(value)
            if soul is None:
                continue
            cards[soul.key] = soul
        if self.acceptself(self.self):
            cards[self.self.key] = Soul(soul=self.self.soul, key=self.self.key)
        return sorted(cards.values(), key=lambda soul: soul.key)

    def soulcard(self, value: Any) -> Optional[Soul]:
        if isinstance(value, Soul):
            return value
        if isinstance(value, dict):
            try:
                return Soul(
                    soul=self.mustname(value.get('soul', '')),
                    key=self.mustkey(value.get('key', value.get('pubkey', ''))),
                )
            except Exception:
                return None
        try:
            return Soul(
                soul=self.mustname(getattr(value, 'soul', '')),
                key=self.mustkey(getattr(value, 'key', getattr(value, 'pubkey', ''))),
            )
        except Exception:
            return None

    def packetSouls(self, values: Iterable[Any]) -> List[Soul]:
        cards: Dict[str, Soul] = {}
        for value in list(values or []):
            soul = self.soulcard(value)
            if soul is None:
                continue
            cards[soul.key] = soul
        return sorted(cards.values(), key=lambda soul: soul.key)


    def Genesis(self, state: Any=None):
        if state is not None:
            if isinstance(state, dict):
                nested = state.get('self', state)
                self.self = Self(
                    soul=self.mustname(nested.get('soul', '')),
                    key=self.mustkey(nested.get('key', nested.get('pubkey', ''))),
                )
                self.souls = self.normalizeSouls(list(self.souls) + list(state.get('souls', [])))
                self.genesisnumber = max(1, int(state.get('genesis', 1) or 1))
            else:
                nested = getattr(state, 'self', state)
                self.self = Self(
                    soul=self.mustname(getattr(nested, 'soul', '')),
                    key=self.mustkey(getattr(nested, 'key', getattr(nested, 'pubkey', ''))),
                )
                self.souls = self.normalizeSouls(list(self.souls) + list(getattr(state, 'souls', [])))
                self.genesisnumber = max(1, int(getattr(state, 'genesis', 1) or 1))
            self.state = self.buildstate()

        if self.genesisdone:
            return self.state
        need = max(1, int(self.genesisnumber or 1))
        have = len(self.souls)
        if have < need:
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

        if kind == KIND_PURGE:
            return {'kind': KIND_PURGE, 'key': self.mustkey(payload.get('key', ''))}

        if kind == KIND_DREAM:
            selfraw = payload.get('self', ['', '']) or ['', '']
            if not isinstance(selfraw, list) or len(selfraw) != 2:
                raise TypeError('dream self must be two-item list')

            dreamsoul = str(selfraw[0] or '')
            dreamkey = str(selfraw[1] or '')
            if not (dreamsoul == '' and dreamkey == ''):
                raise ValueError('dream self must be blank pair')

            cells = []
            for item in tuple(payload.get('cells', ()) or ()):
                purge = dict(item.get('purge', {}) or {})
                lock = dict(item.get('lock', {}) or {})
                cells.append(Field.Cell(
                    soul=self.mustname(item.get('soul', '')),
                    key=self.mustkey(item.get('key', '')),
                    salt=int(item.get('salt', 0) or 0),
                    purge=Field.Purge(
                        chainbit=int(purge.get('chainbit', 0) or 0),
                        lockbit=int(purge.get('lockbit', 0) or 0),
                    ),
                    lock=Field.Lock(
                        parent=self.musthash(lock.get('parent', Field.ZERO_HASH_HEX)),
                        child=self.musthash(lock.get('child', Field.ZERO_HASH_HEX)),
                    ),
                    sign=self.mustsign(item.get('sign', Field.NULL_SIGN_HEX)),
                ))

            return Field.State(
                cells=tuple(cells),
                self=('', ''),
                monument=tuple(str(x or '') for x in (payload.get('monument', ()) or ())),
            )

        saltbody = []
        for item in tuple(payload.get('saltbody', ()) or ()):
            saltbody.append(Field.Salt(
                key=self.mustkey(item.get('key', '')),
                salt=int(item.get('salt', 0) or 0),
            ))

        lockraw = dict(payload.get('lockbody', {}) or {})
        textraw = dict(payload.get('textbody', {}) or {})

        return Field.SaltGlyph(
            key=self.mustkey(payload.get('key', '')),
            saltbody=tuple(saltbody),
            lockbody=Field.Lock(
                parent=self.musthash(lockraw.get('parent', Field.ZERO_HASH_HEX)),
                child=self.musthash(lockraw.get('child', Field.ZERO_HASH_HEX)),
            ),
            textbody=Field.Text(text=str(textraw.get('text', '') or '')),
            salthash=self.musthash(payload.get('salthash', Field.ZERO_HASH_HEX)),
            lockhash=self.musthash(payload.get('lockhash', Field.ZERO_HASH_HEX)),
            texthash=self.musthash(payload.get('texthash', Field.ZERO_HASH_HEX)),
            sign=self.mustsign(payload.get('sign', Field.NULL_SIGN_HEX)),
            locksign=self.mustsign(payload.get('locksign', Field.NULL_SIGN_HEX)),
        )

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

    def emitSouls(self):
        packet = {'header': HEADER_SOULS, 'souls': [soul.box() for soul in self.normalizeSouls(self.souls)]}
        self.emit(packet)

    def emitCompleteSouls(self):
        souls = self.complete or tuple(self.normalizeSouls(self.souls))
        packet = {'header': HEADER_SOULS, 'souls': [soul.box() for soul in souls]}
        self.emit(packet)

    def emitGlyph(self, glyph: Dict[str, Any]):
        payload = {'header': HEADER_GLYPH}
        payload.update(dict(glyph or {}))
        self.emit(payload)

    def emit(self, packet: Dict[str, Any]):
        raw = self.encrypt(packet)
        burst = 3
        peers = self.peers()
        for host, port in peers:
            for shot in range(burst):
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

    def packetRosterhash(self, souls: Iterable[Any]) -> str:
        body = json.dumps([soul.box() for soul in self.packetSouls(souls)], sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(body.encode('utf-8')).hexdigest()


    def encrypt(self, packet: Dict[str, Any]) -> bytes:
        body = json.dumps(packet, sort_keys=True, separators=(',', ':'), default=str)
        data = body.encode('utf-8')
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        raw = bytes((byte ^ mask[i % len(mask)] for i, byte in enumerate(data)))
        return raw

    def decrypt(self, raw: bytes) -> Dict[str, Any]:
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        data = bytes((byte ^ mask[i % len(mask)] for i, byte in enumerate(raw)))
        packet = json.loads(data.decode('utf-8'))
        if not isinstance(packet, dict):
            raise TypeError('packet must decode to dict')
        return packet

    def mustname(self, value: Any) -> str:
        text = str(value or '').strip()
        if not (0 < len(text) <= NAME_MAX):
            raise ValueError('invalid soul name')
        return text

    def mustkey(self, value: Any) -> str:
        text = str(value or '').strip()
        if len(text) != 64:
            raise ValueError('invalid key length')
        bytes.fromhex(text)
        return text

    def musthash(self, value: Any) -> str:
        return self.mustkey(value)

    def mustsign(self, value: Any) -> str:
        text = str(value or '').strip()
        if len(text) != 128:
            raise ValueError('invalid sign length')
        bytes.fromhex(text)
        return text

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

__all__ = [
    'Crypt', 'Self', 'Soul', 'Baton', 'Veil',
    'MODE_SIEGE', 'MODE_CAMPAIGN',
    'HEADER_SOULS', 'HEADER_GLYPH',
    'KIND_PURGE', 'KIND_DREAM',
    'VEIL_PURGE', 'VEIL_SOULS', 'VEIL_GLYPH',
]