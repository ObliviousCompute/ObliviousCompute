from __future__ import annotations

import hashlib
import json
import select
import socket
import threading
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import Dream
import Field
import Sanctum

NameMax = 8

ModeSiege = 'siege'
ModeCampaign = 'campaign'

HeaderSouls = 'souls'
HeaderGlyph = 'glyph'

KindPurge = 'purge'
KindDream = 'dream'
KindSalt = 'salt'

VeilPurge = 'purge'
VeilSouls = 'souls'
VeilGlyph = 'glyph'


@dataclass(frozen=True)
class Self:
    soul: str = ''
    key: str = ''

    def Box(self) -> dict[str, str]:
        return {'soul': self.soul, 'key': self.key}


@dataclass(frozen=True)
class Soul:
    soul: str = ''
    key: str = ''

    def Box(self) -> dict[str, str]:
        return {'soul': self.soul, 'key': self.key}


@dataclass(frozen=True)
class Baton:
    self: Self = field(default_factory=Self)
    souls: tuple[Soul, ...] = ()
    genesis: int = 1

    def Box(self) -> dict[str, Any]:
        return {
            'self': self.self.Box(),
            'souls': [soul.Box() for soul in self.souls],
            'genesis': int(self.genesis),
        }


@dataclass
class Veil:
    purgewindow: Optional[tuple[int, int]] = None
    soulswindow: Optional[tuple[int, int]] = None
    glyphwindow: Optional[tuple[int, int]] = None
    dedupe: list[str] = field(default_factory=list)
    dedupesize: int = 2

    def Accepts(self, raw: bytes, lane: str) -> bool:
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            return False
        size = len(raw)
        if size < 8 or size > 65535:
            return False
        if lane == VeilPurge:
            window = self.purgewindow
        elif lane == VeilSouls:
            window = self.soulswindow
        elif lane == VeilGlyph:
            window = self.glyphwindow
        else:
            return False
        if window is None:
            return True
        low, high = window
        return int(low) <= size <= int(high)

    def Seen(self, digest: str) -> bool:
        if self.dedupesize <= 0:
            return False
        return digest in self.dedupe

    def Remember(self, digest: str) -> None:
        if self.dedupesize <= 0:
            return
        self.dedupe = [item for item in self.dedupe if item != digest]
        self.dedupe.append(digest)
        self.dedupe = self.dedupe[-max(1, int(self.dedupesize)):]


class Crypt:

    def __init__(self, state: Any = None, sanctum: Any = None, port: int = 9000, dream: Any = None):
        self.sanctum = sanctum
        self.dream = self.WakeDream(dream)

        if isinstance(state, dict):
            mode = str(state.get('mode', ModeSiege) or ModeSiege).strip().lower()
            skeleton = str(state.get('skeleton', '') or '')
            genesis = int(state.get('genesis', 1) or 1)
            gate = int(state.get('gate', state.get('port', port)) or port)
            nested = state.get('self', state)
            selfcard = Self(
                soul=self.MustName(nested.get('soul', '')),
                key=self.MustKey(nested.get('key', nested.get('pubkey', ''))),
            )
            rawsouls = state.get('souls', [])
        else:
            mode = str(getattr(state, 'mode', ModeSiege) or ModeSiege).strip().lower()
            skeleton = str(getattr(state, 'skeleton', '') or '')
            genesis = int(getattr(state, 'genesis', 1) or 1)
            gate = int(getattr(state, 'gate', getattr(state, 'port', port)) or port)
            nested = getattr(state, 'self', state)
            selfcard = Self(
                soul=self.MustName(getattr(nested, 'soul', '')),
                key=self.MustKey(getattr(nested, 'key', getattr(nested, 'pubkey', ''))),
            )
            rawsouls = getattr(state, 'souls', [])

        self.mode = mode if mode in (ModeSiege, ModeCampaign) else ModeSiege
        self.skeleton = skeleton
        self.genesisnumber = max(1, genesis)
        self.gate = gate
        self.self = selfcard
        self.souls = self.SoulSet(rawsouls)

        self.veil = Veil()
        self.genesisdone = False
        self.complete: tuple[Soul, ...] = ()
        self.state = Baton(self=self.self, souls=tuple(), genesis=int(self.genesisnumber))
        self.glyph = None
        self.Grind = None
        self.Grindlock = threading.Lock()
        self.Zzz = False

        self.bindhost = ''
        self.bindport: Optional[int] = None
        self.sock = self.BindTransport()
        self.sock.setblocking(True)

        self.live = False
        self.thread: Optional[threading.Thread] = None

        self.state = self.BuildState()
        self.Start()
        self.EmitSouls()
        self.Genesis(state)

    def Start(self):
        if self.live:
            return
        self.live = True
        self.thread = threading.Thread(target=self.Listen, name='CryptListen', daemon=True)
        self.thread.start()

    def Sleep(self):
        self.live = False
        try:
            self.sock.close()
        except Exception:
            pass
        thread = self.thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=0.2)
        self.thread = None

    def Listen(self):
        while self.live:
            try:
                raw, addr = self.Summon()
            except OSError:
                break
            except Exception:
                continue
            try:
                self.Receive(raw, addr)
                self.GrindSocket()
                self.Wake()
            except Exception:
                continue

    def Wake(self):
        with self.Grindlock:
            if self.Zzz:
                return self.state
            if self.Grind is None:
                return self.state
            if bool(getattr(self.dream, 'Dreaming', False)):
                return self.state
            payload = self.Grind
            self.Grind = None
            self.Zzz = True
        self.glyph = payload
        try:
            if hasattr(self.dream, 'box'):
                self.dream.box.crypt = payload
                self.dream.Wake()
        except Exception:
            with self.Grindlock:
                if self.Grind is None:
                    self.Grind = payload
                self.Zzz = False
            raise
        with self.Grindlock:
            self.Zzz = False
            more = self.Grind is not None
        if more and not bool(getattr(self.dream, 'Dreaming', False)):
            return self.Wake()
        return self.state

    def Awake(self):
        return self.Wake()

    def Tick(self):
        return self.Wake()

    def Summon(self):
        return self.sock.recvfrom(65535)

    def GrindSocket(self):
        while self.live:
            try:
                ready, writeable, broken = select.select([self.sock], [], [], 0.0)
            except OSError:
                break
            except Exception:
                break
            if not ready:
                break
            try:
                raw, addr = self.Summon()
            except OSError:
                break
            except Exception:
                break
            try:
                self.Receive(raw, addr)
            except Exception:
                continue
        return self.state

    def BindTransport(self) -> socket.socket:
        if self.mode == ModeSiege:
            return self.BindSiege()
        if self.mode == ModeCampaign:
            return self.BindCampaign()
        raise ValueError(f'unsupported mode: {self.mode!r}')

    def BindSiege(self) -> socket.socket:
        lasterror: Optional[Exception] = None
        for port in self.SeatPorts():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.bind(('127.0.0.1', port))
                self.bindhost = '127.0.0.1'
                self.bindport = port
                return sock
            except OSError as exc:
                lasterror = exc
                try:
                    sock.close()
                except Exception:
                    pass
        raise RuntimeError(f'No clean siege seat available in reserved range {self.SeatPorts()}.') from lasterror

    def BindCampaign(self) -> socket.socket:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.bind(('', self.gate))
        except OSError:
            sock.bind(('0.0.0.0', self.gate))
        self.bindhost = '0.0.0.0'
        self.bindport = self.gate
        return sock

    def SeatPorts(self) -> list[int]:
        return [self.gate + index for index in range(max(1, int(self.genesisnumber or 1)))]

    def SiegePeers(self) -> list[tuple[str, int]]:
        return [('127.0.0.1', port) for port in self.SeatPorts() if int(port) != int(self.bindport or -1)]

    def CampaignPeers(self) -> list[tuple[str, int]]:
        return [(self.BroadcastTarget(), self.gate)]

    def Peers(self) -> list[tuple[str, int]]:
        if self.mode == ModeSiege:
            return self.SiegePeers()
        if self.mode == ModeCampaign:
            return self.CampaignPeers()
        return []

    def BroadcastTarget(self) -> str:
        parts = self.LocalIp().split('.')
        if len(parts) == 4:
            parts[-1] = '255'
            return '.'.join(parts)
        return '255.255.255.255'

    def LocalIp(self) -> str:
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            probe.connect(('8.8.8.8', 80))
            ip = probe.getsockname()[0]
            probe.close()
            return ip
        except Exception:
            return '127.0.0.1'

    def Poll(self):
        return self.GrindSocket()

    def Receive(self, raw: bytes, addr: tuple[str, int]):
        if self.CampaignSelf(addr):
            return
        if not self.veil.Accepts(raw, VeilGlyph):
            return
        self.Cryptkeeper(raw, addr)

    def CampaignSelf(self, addr: tuple[str, int]) -> bool:
        if self.mode != ModeCampaign:
            return False
        host, port = addr
        if int(port) != int(self.bindport or -1):
            return False
        local = self.LocalIp()
        return host in (local, '127.0.0.1', '0.0.0.0')

    def Cryptkeeper(self, raw: bytes, addr: tuple[str, int]):
        packet = self.Decrypt(raw)
        header = self.HeaderOf(packet)

        if header == HeaderSouls:
            if not self.veil.Accepts(raw, VeilSouls):
                return
        elif header == HeaderGlyph:
            kind = self.KindOf(packet)
            if kind == KindPurge:
                if not self.veil.Accepts(raw, VeilPurge):
                    return
            elif not self.veil.Accepts(raw, VeilGlyph):
                return
        else:
            return

        skipdedupe = 100 <= len(raw) <= 125
        if not skipdedupe:
            digest = self.PacketHash(packet)
            if self.veil.Seen(digest):
                return
            self.veil.Remember(digest)

        if header == HeaderSouls:
            self.SoulSqueeze(packet, addr)
            return

        self.RouteGlyph(packet, addr)

    def SoulFlare(self):
        self.EmitCompleteSouls()

    def SoulSqueeze(self, packet: dict[str, Any], addr: tuple[str, int]):
        incomingpacket = self.SoulPack(packet.get('souls', []))
        need = max(1, int(self.genesisnumber or 1))
        incomingkeys = {soul.key for soul in incomingpacket}

        if self.genesisdone:
            self.SoulFlare()
            return

        if len(incomingpacket) >= need and self.self.key not in incomingkeys:
            locked = tuple(incomingpacket[:need])
            self.souls = list(locked)
            self.complete = locked
            self.genesisdone = True
            self.state = self.BuildState()

            sanctum = self.WakeSanctum()
            sanctum.Genesis(self.state)
            return

        before = self.RosterHash(self.souls)
        merged = self.SoulSet(list(self.souls) + list(incomingpacket or []))
        after = self.RosterHash(merged)
        if after != before:
            self.souls = merged
            self.state = self.BuildState()
            self.EmitSouls()
        self.Genesis()

    def SoulSet(self, values: Iterable[Any]) -> list[Soul]:
        cards: dict[str, Soul] = {}
        for value in list(values or []):
            soul = self.SoulShape(value)
            if soul is None:
                continue
            cards[soul.key] = soul
        if self.AcceptSelf(self.self):
            cards[self.self.key] = Soul(soul=self.self.soul, key=self.self.key)
        return sorted(cards.values(), key=lambda soul: soul.key)

    def SoulShape(self, value: Any) -> Optional[Soul]:
        if isinstance(value, Soul):
            return value
        if isinstance(value, dict):
            try:
                return Soul(
                    soul=self.MustName(value.get('soul', '')),
                    key=self.MustKey(value.get('key', value.get('pubkey', ''))),
                )
            except Exception:
                return None
        try:
            return Soul(
                soul=self.MustName(getattr(value, 'soul', '')),
                key=self.MustKey(getattr(value, 'key', getattr(value, 'pubkey', ''))),
            )
        except Exception:
            return None

    def SoulPack(self, values: Iterable[Any]) -> list[Soul]:
        cards: dict[str, Soul] = {}
        for value in list(values or []):
            soul = self.SoulShape(value)
            if soul is None:
                continue
            cards[soul.key] = soul
        return sorted(cards.values(), key=lambda soul: soul.key)

    def Genesis(self, state: Any = None):
        if state is not None:
            if isinstance(state, dict):
                nested = state.get('self', state)
                self.self = Self(
                    soul=self.MustName(nested.get('soul', '')),
                    key=self.MustKey(nested.get('key', nested.get('pubkey', ''))),
                )
                self.souls = self.SoulSet(list(self.souls) + list(state.get('souls', [])))
                self.genesisnumber = max(1, int(state.get('genesis', 1) or 1))
            else:
                nested = getattr(state, 'self', state)
                self.self = Self(
                    soul=self.MustName(getattr(nested, 'soul', '')),
                    key=self.MustKey(getattr(nested, 'key', getattr(nested, 'pubkey', ''))),
                )
                self.souls = self.SoulSet(list(self.souls) + list(getattr(state, 'souls', [])))
                self.genesisnumber = max(1, int(getattr(state, 'genesis', 1) or 1))
            self.state = self.BuildState()

        if self.genesisdone:
            return self.state
        need = max(1, int(self.genesisnumber or 1))
        have = len(self.souls)
        if have < need:
            return self.state

        self.genesisdone = True
        self.complete = tuple(self.SoulSet(self.souls))
        self.state = self.BuildState()
        self.EmitCompleteSouls()

        sanctum = self.WakeSanctum()
        return sanctum.Genesis(self.state)

    def UnboxGlyph(self, payload: dict[str, Any]) -> Any:
        kind = str(payload.get('kind', '') or '').strip().lower()

        if kind == KindPurge:
            return {'kind': KindPurge, 'key': self.MustKey(payload.get('key', ''))}

        if kind == KindDream:
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
                cells.append(
                    Field.Cell(
                        soul=self.MustName(item.get('soul', '')),
                        key=self.MustKey(item.get('key', '')),
                        salt=int(item.get('salt', 0) or 0),
                        purge=Field.Purge(
                            chainbit=int(purge.get('chainbit', 0) or 0),
                            lockbit=int(purge.get('lockbit', 0) or 0),
                        ),
                        lock=Field.Lock(
                            parent=self.MustHash(lock.get('parent', Field.ZeroHashHex)),
                            child=self.MustHash(lock.get('child', Field.ZeroHashHex)),
                        ),
                        sign=self.MustSign(item.get('sign', Field.NullSignHex)),
                    )
                )

            return Field.State(
                cells=tuple(cells),
                self=('', ''),
                monument=tuple(str(item or '') for item in (payload.get('monument', ()) or ())),
                pristine=int(payload.get('pristine', 1) or 0),
            )

        saltbody = []
        for item in tuple(payload.get('saltbody', ()) or ()):
            saltbody.append(
                Field.Salt(
                    key=self.MustKey(item.get('key', '')),
                    salt=int(item.get('salt', 0) or 0),
                )
            )

        lockraw = dict(payload.get('lockbody', {}) or {})
        textraw = dict(payload.get('textbody', {}) or {})

        return Field.SaltGlyph(
            key=self.MustKey(payload.get('key', '')),
            saltbody=tuple(saltbody),
            lockbody=Field.Lock(
                parent=self.MustHash(lockraw.get('parent', Field.ZeroHashHex)),
                child=self.MustHash(lockraw.get('child', Field.ZeroHashHex)),
            ),
            textbody=Field.Text(text=str(textraw.get('text', '') or '')),
            salthash=self.MustHash(payload.get('salthash', Field.ZeroHashHex)),
            lockhash=self.MustHash(payload.get('lockhash', Field.ZeroHashHex)),
            texthash=self.MustHash(payload.get('texthash', Field.ZeroHashHex)),
            sign=self.MustSign(payload.get('sign', Field.NullSignHex)),
            locksign=self.MustSign(payload.get('locksign', Field.NullSignHex)),
        )

    def RouteGlyph(self, packet: dict[str, Any], addr: tuple[str, int]):
        if not self.genesisdone:
            return
        payload = dict(packet)
        payload.pop('header', None)
        payload = self.UnboxGlyph(payload)
        with self.Grindlock:
            self.Grind = payload
        self.glyph = payload

    def EmitSouls(self):
        packet = {'header': HeaderSouls, 'souls': [soul.Box() for soul in self.SoulSet(self.souls)]}
        self.Cast(packet)

    def EmitCompleteSouls(self):
        souls = self.complete or tuple(self.SoulSet(self.souls))
        packet = {'header': HeaderSouls, 'souls': [soul.Box() for soul in souls]}
        self.Cast(packet)

    def EmitGlyph(self, glyph: dict[str, Any]):
        payload = {'header': HeaderGlyph}
        payload.update(dict(glyph or {}))
        self.Emit(payload)

    def Cast(self, packet: dict[str, Any]):
        raw = self.Encrypt(packet)
        burst = 3
        peers = self.Peers()
        for host, port in peers:
            for shot in range(burst):
                try:
                    self.sock.sendto(raw, (host, port))
                except Exception:
                    pass

    def Emit(self, packet: dict[str, Any]):
        return self.Cast(packet)

    def BuildState(self) -> Baton:
        souls = tuple(self.complete) if self.genesisdone else tuple(self.SoulSet(self.souls))
        return Baton(self=Self(soul=self.self.soul, key=self.self.key), souls=souls, genesis=int(self.genesisnumber))

    def WakeSanctum(self):
        if self.sanctum is None:
            self.sanctum = Sanctum.Sanctum()
            return self.sanctum
        if isinstance(self.sanctum, type):
            self.sanctum = self.sanctum()
            return self.sanctum
        return self.sanctum

    def WakeDream(self, dream: Any):
        if dream is None:
            return getattr(Dream, 'dream', None) or Dream.Dream()
        if isinstance(dream, type):
            return dream()
        return dream

    def HeaderOf(self, packet: dict[str, Any]) -> str:
        return str(packet.get('header', '') or '').strip().lower()

    def KindOf(self, packet: dict[str, Any]) -> str:
        return str(packet.get('kind', '') or '').strip().lower()

    def PacketHash(self, packet: dict[str, Any]) -> str:
        body = json.dumps(packet, sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(body.encode('utf-8')).hexdigest()

    def RosterHash(self, souls: Iterable[Any]) -> str:
        body = json.dumps([soul.Box() for soul in self.SoulSet(souls)], sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(body.encode('utf-8')).hexdigest()

    def PacketRosterHash(self, souls: Iterable[Any]) -> str:
        body = json.dumps([soul.Box() for soul in self.SoulPack(souls)], sort_keys=True, separators=(',', ':'), default=str)
        return hashlib.sha256(body.encode('utf-8')).hexdigest()

    def Encrypt(self, packet: dict[str, Any]) -> bytes:
        body = json.dumps(packet, sort_keys=True, separators=(',', ':'), default=str)
        data = body.encode('utf-8')
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        return bytes((byte ^ mask[index % len(mask)] for index, byte in enumerate(data)))

    def Decrypt(self, raw: bytes) -> dict[str, Any]:
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        data = bytes((byte ^ mask[index % len(mask)] for index, byte in enumerate(raw)))
        packet = json.loads(data.decode('utf-8'))
        if not isinstance(packet, dict):
            raise TypeError('packet must decode to dict')
        return packet

    def MustName(self, value: Any) -> str:
        text = str(value or '').strip()
        if not (0 < len(text) <= NameMax):
            raise ValueError('invalid soul name')
        return text

    def MustKey(self, value: Any) -> str:
        text = str(value or '').strip()
        Field.VerifyKey(text)
        return text

    def MustHash(self, value: Any) -> str:
        text = str(value or '').strip()
        Field.VerifyHash(text, fieldname='hash')
        return text

    def MustSign(self, value: Any) -> str:
        text = str(value or '').strip()
        Field.VerifySignHex(text, fieldname='sign')
        return text

    def AcceptSoulName(self, value: str) -> bool:
        return isinstance(value, str) and 0 < len(value) <= NameMax

    def AcceptKey(self, value: str) -> bool:
        try:
            return bool(Field.VerifyKey(value))
        except Exception:
            return False

    def AcceptSelf(self, value: Self) -> bool:
        return self.AcceptSoulName(value.soul) and self.AcceptKey(value.key)

    def AcceptSoulCard(self, value: Soul) -> bool:
        return self.AcceptSoulName(value.soul) and self.AcceptKey(value.key)


crypt = None
