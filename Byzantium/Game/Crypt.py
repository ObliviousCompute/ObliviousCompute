from __future__ import annotations

import hashlib
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

KindDream = 0
KindNightmare = 1
KindPurge = 2
KindWhisper = 3
KindRally = 4
KindDefect = 5
KindWrath = 6
KindSouls = 7
KindSalt = (KindWhisper, KindRally, KindDefect, KindWrath)

HeaderMagic = b'BYZ!'
HeaderVersion = 1
HeaderSize = 64
ReceiptMagic = b'^_^'
ReceiptSize = 352
PayoutSlots = 23
PayoutSlotSize = 8
TextBrickSize = 96
SoulCardSize = 40
CellMetaSize = 64
CellSize = 768

PurgeSize = 64
SaltSize = 512
NightmareSize = 768
SoulsSize = 1024
DreamSize = 18496

GlyphSizes = {
    KindDream: DreamSize,
    KindNightmare: NightmareSize,
    KindPurge: PurgeSize,
    KindWhisper: SaltSize,
    KindRally: SaltSize,
    KindDefect: SaltSize,
    KindWrath: SaltSize,
    KindSouls: SoulsSize,
}
LegalSizes = frozenset(GlyphSizes.values())

Zero4 = b'\x00' * 4
Zero8 = b'\x00' * 8
Zero19 = b'\x00' * 19
Zero32 = b'\x00' * 32
Zero352 = b'\x00' * ReceiptSize


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
class SoulsGlyph:
    expected: int
    souls: tuple[Soul, ...] = ()


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
    dedupe: list[str] = field(default_factory=list)
    dedupesize: int = 8

    def Accepts(self, raw: bytes) -> bool:
        return isinstance(raw, (bytes, bytearray)) and len(raw) in LegalSizes

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
            genesis = self.MustGenesis(state.get('genesis', 1))
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
            genesis = self.MustGenesis(getattr(state, 'genesis', 1))
            gate = int(getattr(state, 'gate', getattr(state, 'port', port)) or port)
            nested = getattr(state, 'self', state)
            selfcard = Self(
                soul=self.MustName(getattr(nested, 'soul', '')),
                key=self.MustKey(getattr(nested, 'key', getattr(nested, 'pubkey', ''))),
            )
            rawsouls = getattr(state, 'souls', [])

        self.mode = mode if mode in (ModeSiege, ModeCampaign) else ModeSiege
        self.skeleton = skeleton
        self.genesisnumber = genesis
        self.gate = gate
        self.self = selfcard
        self.souls = self.SoulPack(rawsouls) or self.SoulPack(()) or []

        self.veil = Veil()
        self.genesisdone = False
        self.complete: tuple[Soul, ...] = ()
        self.state = Baton(self=self.self, souls=tuple(), genesis=int(self.genesisnumber))
        self.glyph = None
        self.Reap = []
        self.Reaping = False
        self.ReapLock = threading.Lock()

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

    # ---------- transport ----------

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
        with self.ReapLock:
            if not self.Reaping or not self.Reap:
                return self.state
            batch = list(self.Reap)
            self.Reap = []
            self.Reaping = False

        try:
            if hasattr(self.dream, 'box'):
                lane = getattr(self.dream.box, 'crypt', None)
                if lane is None:
                    self.dream.box.crypt = list(batch)
                elif isinstance(lane, list):
                    lane.extend(batch)
                else:
                    self.dream.box.crypt = [lane] + list(batch)
                self.dream.Wake()
        except Exception:
            with self.ReapLock:
                if not self.Reaping:
                    self.Reaping = True
                    self.Reap = []
                self.Reap = list(batch) + list(self.Reap)
            raise
        return self.state

    def Summon(self):
        return self.sock.recvfrom(65535)

    def GrindSocket(self):
        while self.live:
            try:
                ready, _writeable, _broken = select.select([self.sock], [], [], 0.0)
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
        return [self.gate + index for index in range(self.genesisnumber)]

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
        if not self.veil.Accepts(raw):
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
        plain = self.Decrypt(raw)
        header = self.ParseHeader(plain)
        kind = header['kind']
        glyph = self.Unpack(plain)

        # Purge requests are deliberately repeatable. Everything else is burst-deduped.
        if kind != KindPurge:
            digest = hashlib.sha256(plain).hexdigest()
            if self.veil.Seen(digest):
                return
            self.veil.Remember(digest)

        if isinstance(glyph, SoulsGlyph):
            self.SoulSqueeze(glyph, addr)
            return
        self.RouteGlyph(glyph, addr)

    # ---------- genesis / souls ----------

    def SoulFlare(self):
        self.EmitSouls()

    def SoulSqueeze(self, glyph: SoulsGlyph, addr: tuple[str, int]):
        if int(glyph.expected) != int(self.genesisnumber):
            return self.state
        if self.genesisdone:
            incomingsouls = self.SoulSnap(glyph.souls)
            if incomingsouls is None:
                return self.state
            completesouls = tuple(self.complete or ())
            self.SoulSwap(incomingsouls, completesouls)
            return self.state

        incomingsouls = self.SoulPack(glyph.souls)
        if incomingsouls is None:
            return self.state

        beforekeys = self.SoulKeys(self.souls)
        merged = self.SoulPack(list(self.souls) + list(incomingsouls))
        if merged is None:
            return self.state
        afterkeys = self.SoulKeys(merged)

        if afterkeys != beforekeys:
            self.souls = merged
            self.state = self.BuildState()
            self.SoulFlare()

        self.Genesis(merged)
        return self.state

    def SoulSwap(self, incomingsouls: Iterable[Any], completesouls: Iterable[Any]):
        incomingsouls = self.SoulSnap(incomingsouls)
        completesouls = self.SoulSnap(completesouls)
        if incomingsouls is None or completesouls is None:
            return self.state
        incomingkeys = self.SoulKeys(incomingsouls)
        completekeys = self.SoulKeys(completesouls)
        if not incomingkeys:
            return self.state
        if not incomingkeys.issubset(completekeys):
            return self.state
        if len(incomingkeys) >= len(completekeys):
            return self.state
        self.EmitCompleteSouls()
        return self.state

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

    def SoulPack(self, values: Iterable[Any]) -> Optional[list[Soul]]:
        cards: dict[str, Soul] = {}
        for value in list(values or []):
            soul = self.SoulShape(value)
            if soul is None:
                continue
            cards[soul.key] = soul
        if self.AcceptSelf(self.self) and self.self.key not in cards:
            cards[self.self.key] = Soul(soul=self.self.soul, key=self.self.key)
        souls = sorted(cards.values(), key=lambda soul: soul.key)
        if len(souls) > self.genesisnumber or len(souls) > Field.SeatCount:
            return None
        return souls

    def SoulSnap(self, values: Iterable[Any]) -> Optional[list[Soul]]:
        cards: dict[str, Soul] = {}
        for value in list(values or []):
            soul = self.SoulShape(value)
            if soul is None:
                continue
            cards[soul.key] = soul
        souls = sorted(cards.values(), key=lambda soul: soul.key)
        if len(souls) > self.genesisnumber or len(souls) > Field.SeatCount:
            return None
        return souls

    def SoulKeys(self, souls: Iterable[Any]) -> set[str]:
        keys: set[str] = set()
        for soul in list(souls or []):
            shaped = self.SoulShape(soul)
            if shaped is None:
                continue
            keys.add(shaped.key)
        return keys

    def Genesis(self, state: Any = None):
        if state is not None:
            if isinstance(state, dict):
                nested = state.get('self', state)
                self.self = Self(
                    soul=self.MustName(nested.get('soul', '')),
                    key=self.MustKey(nested.get('key', nested.get('pubkey', ''))),
                )
                self.genesisnumber = self.MustGenesis(state.get('genesis', 1))
                merged = self.SoulPack(list(self.souls) + list(state.get('souls', [])))
                if merged is not None:
                    self.souls = merged
            elif self.AcceptSoulIterable(state):
                merged = self.SoulPack(state)
                if merged is not None:
                    self.souls = merged
            else:
                nested = getattr(state, 'self', state)
                self.self = Self(
                    soul=self.MustName(getattr(nested, 'soul', '')),
                    key=self.MustKey(getattr(nested, 'key', getattr(nested, 'pubkey', ''))),
                )
                self.genesisnumber = self.MustGenesis(getattr(state, 'genesis', 1))
                merged = self.SoulPack(list(self.souls) + list(getattr(state, 'souls', [])))
                if merged is not None:
                    self.souls = merged
            self.state = self.BuildState()

        if self.genesisdone:
            return self.state
        need = self.genesisnumber
        have = len(self.souls)
        if have < need:
            return self.state

        self.genesisdone = True
        self.complete = tuple(self.souls)
        self.state = self.BuildState()
        self.EmitCompleteSouls()
        sanctum = self.WakeSanctum()
        return sanctum.Genesis(self.state)

    def AcceptSoulIterable(self, value: Any) -> bool:
        if not isinstance(value, (list, tuple)):
            return False
        return all(self.SoulShape(item) is not None for item in value)

    # ---------- fixed geometry ----------

    def U32(self, value: int, fieldname: str = 'value') -> bytes:
        number = int(value)
        if number < 0 or number > 0xFFFFFFFF:
            raise ValueError(f'{fieldname} must fit unsigned 32 bits')
        return number.to_bytes(4, 'big')

    def ReadU32(self, data: bytes) -> int:
        if len(data) != 4:
            raise ValueError('u32 requires exactly four bytes')
        return int.from_bytes(data, 'big')

    def BuildHeader(
        self,
        kind: int,
        *,
        flags: int = 0,
        expected: int = 0,
        populated: int = 0,
        key: bytes = Zero32,
    ) -> bytes:
        if kind not in GlyphSizes:
            raise ValueError('unknown glyph kind')
        if not 0 <= int(flags) <= 0xFF:
            raise ValueError('header flags must fit one byte')
        if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
            raise ValueError('header key must be exactly 32 bytes')
        out = bytearray(HeaderSize)
        out[0:4] = HeaderMagic
        out[4] = HeaderVersion
        out[5] = int(kind)
        out[6] = int(flags)
        out[7] = 0
        out[8:12] = self.U32(GlyphSizes[kind], 'glyph size')
        out[12:16] = self.U32(expected, 'expected souls')
        out[16:20] = self.U32(populated, 'populated souls')
        out[20:24] = Zero4
        out[24:56] = bytes(key)
        out[56:64] = Zero8
        return bytes(out)

    def ParseHeader(self, data: bytes) -> dict[str, Any]:
        if not isinstance(data, (bytes, bytearray)) or len(data) < HeaderSize:
            raise ValueError('glyph shorter than fixed header')
        header = bytes(data[:HeaderSize])
        if header[0:4] != HeaderMagic:
            raise ValueError('glyph header magic mismatch')
        if header[4] != HeaderVersion:
            raise ValueError('unsupported glyph header version')
        kind = int(header[5])
        if kind not in GlyphSizes:
            raise ValueError('unknown glyph header kind')
        if header[7] != 0 or header[20:24] != Zero4 or header[56:64] != Zero8:
            raise ValueError('glyph header reserved bytes must be zero')
        size = self.ReadU32(header[8:12])
        if size != GlyphSizes[kind] or len(data) != size:
            raise ValueError('glyph length does not match canonical kind geometry')
        flags = int(header[6])
        expected = self.ReadU32(header[12:16])
        populated = self.ReadU32(header[16:20])
        key = header[24:56]

        if kind == KindDream:
            if flags not in (0, 1) or expected != 0 or populated != 0 or key != Zero32:
                raise ValueError('dream header metadata is non-canonical')
        elif kind == KindSouls:
            if flags != 0 or not (1 <= expected <= Field.SeatCount) or populated > expected or key != Zero32:
                raise ValueError('souls header metadata is non-canonical')
        elif kind == KindPurge:
            if flags != 0 or expected != 0 or populated != 0 or key == Zero32:
                raise ValueError('purge header metadata is non-canonical')
        else:
            if flags != 0 or expected != 0 or populated != 0 or key != Zero32:
                raise ValueError('glyph header metadata is non-canonical')
        return {
            'kind': kind,
            'flags': flags,
            'size': size,
            'expected': expected,
            'populated': populated,
            'key': key,
        }

    def EncodeName(self, value: Any) -> bytes:
        text = str(value or '')
        if not text.strip() or len(text) > NameMax or '\x00' in text:
            raise ValueError('invalid soul name')
        raw = text.encode('utf-8')
        if len(raw) > NameMax:
            raise ValueError('soul name must fit fixed 8-byte UTF-8 field')
        return raw + (b'\x00' * (NameMax - len(raw)))

    def DecodeName(self, data: bytes) -> str:
        if len(data) != NameMax:
            raise ValueError('soul name field must be exactly eight bytes')
        cut = data.find(b'\x00')
        if cut < 0:
            raw = data
        else:
            raw = data[:cut]
            if any(data[cut:]):
                raise ValueError('soul name padding must be zero')
        try:
            text = raw.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('soul name is not valid UTF-8') from exc
        if not text.strip() or len(text) > NameMax or len(raw) > NameMax:
            raise ValueError('invalid decoded soul name')
        return text

    def PadLock(self, lock: Field.Lock) -> bytes:
        Field.VerifyLockShape(lock, allowempty=True)
        if lock.kind != Field.KindEmpty and lock.tag == Field.ZeroTagHex:
            raise ValueError('zero tag is reserved for payout padding')
        if lock.tag:
            tag = bytes.fromhex(self.MustTag(lock.tag))
        else:
            tag = Zero4

        out = bytearray(ReceiptSize)
        out[0:3] = ReceiptMagic
        out[3] = int(lock.kind)
        out[4:8] = tag
        out[8:40] = bytes.fromhex(self.MustHash(lock.parent))
        out[40:72] = bytes.fromhex(self.MustHash(lock.child))
        out[72:104] = bytes.fromhex(self.MustHash(lock.texthash))
        out[104:168] = bytes.fromhex(self.MustSign(lock.sign))

        expected = Field.KindSpendCounts[int(lock.kind)]
        if len(lock.payout) != expected:
            raise ValueError('receipt payout count does not match kind')
        cursor = 168
        for index in range(PayoutSlots):
            if index < expected:
                leg = lock.payout[index]
                if leg.tag == Field.ZeroTagHex:
                    raise ValueError('occupied payout slot may not use zero tag')
                out[cursor:cursor + 4] = bytes.fromhex(self.MustTag(leg.tag))
                out[cursor + 4:cursor + 8] = self.U32(leg.salt, 'payout salt')
            else:
                out[cursor:cursor + 8] = b'\x00' * 8
            cursor += PayoutSlotSize
        if cursor != ReceiptSize:
            raise AssertionError('receipt geometry drift')
        return bytes(out)

    def Unlock(self, data: bytes) -> Field.Lock:
        if not isinstance(data, (bytes, bytearray)) or len(data) != ReceiptSize:
            raise ValueError('PadLock block must be exactly 352 bytes')
        raw = bytes(data)
        if raw[0:3] != ReceiptMagic:
            raise ValueError('PadLock receipt magic mismatch')
        kind = int(raw[3])
        Field.VerifyKind(kind)
        tagraw = raw[4:8]
        tag = '' if tagraw == Zero4 and kind == Field.KindEmpty else tagraw.hex()
        if tag:
            self.MustTag(tag)
            if tag == Field.ZeroTagHex:
                raise ValueError('receipt tag may not use zero padding tag')

        expected = Field.KindSpendCounts[kind]
        payout = []
        cursor = 168
        for index in range(PayoutSlots):
            slot = raw[cursor:cursor + PayoutSlotSize]
            if index < expected:
                tagbytes = slot[:4]
                if tagbytes == Zero4:
                    raise ValueError('occupied payout slot requires nonzero tag')
                payout.append(Field.Payout(tag=tagbytes.hex(), salt=self.ReadU32(slot[4:8])))
            elif slot != b'\x00' * PayoutSlotSize:
                raise ValueError('unused payout slot must be all zero')
            cursor += PayoutSlotSize

        lock = Field.Lock(
            kind=kind,
            tag=tag,
            parent=raw[8:40].hex(),
            child=raw[40:72].hex(),
            payout=tuple(payout),
            texthash=raw[72:104].hex(),
            sign=raw[104:168].hex(),
        )
        Field.VerifyLockShape(lock, allowempty=True)
        return lock

    def PackText(self, textbody: Field.Text) -> bytes:
        if not isinstance(textbody, Field.Text):
            raise TypeError('PackText expects Field.Text')
        raw = textbody.text.encode('utf-8')
        if len(raw) > TextBrickSize - 1:
            raise ValueError('Salt text exceeds 95-byte fixed text payload')
        return bytes([len(raw)]) + raw + (b'\x00' * (TextBrickSize - 1 - len(raw)))

    def UnpackText(self, data: bytes) -> Field.Text:
        if len(data) != TextBrickSize:
            raise ValueError('text brick must be exactly 96 bytes')
        size = int(data[0])
        if size > TextBrickSize - 1:
            raise ValueError('text brick length byte out of range')
        body = data[1:1 + size]
        if any(data[1 + size:]):
            raise ValueError('text brick padding must be zero')
        try:
            text = body.decode('utf-8')
        except UnicodeDecodeError as exc:
            raise ValueError('text brick is not valid UTF-8') from exc
        return Field.Text(text=text)

    def PackCell(self, cell: Field.Cell) -> bytes:
        Field.VerifyCell(cell)
        meta = bytearray(CellMetaSize)
        meta[0:8] = self.EncodeName(cell.soul)
        meta[8:40] = bytes.fromhex(self.MustKey(cell.key))
        meta[40:44] = self.U32(cell.salt, 'cell salt')
        purge = (int(cell.purge.chainbit) & 1) | ((int(cell.purge.lockbit) & 1) << 1)
        meta[44] = purge
        meta[45:64] = Zero19
        low = Zero352 if cell.lowlock is None else self.PadLock(cell.lowlock)
        out = bytes(meta) + self.PadLock(cell.lock) + low
        if len(out) != CellSize:
            raise AssertionError('cell geometry drift')
        return out

    def UnpackCell(self, data: bytes) -> Field.Cell:
        if len(data) != CellSize:
            raise ValueError('cell stone must be exactly 768 bytes')
        meta = data[:CellMetaSize]
        if any(meta[45:64]):
            raise ValueError('cell metadata reserved bytes must be zero')
        purgebyte = int(meta[44])
        if purgebyte & ~0b11:
            raise ValueError('cell purge byte has non-canonical bits')
        key = meta[8:40].hex()
        self.MustKey(key)
        lock = self.Unlock(data[64:416])
        lowraw = data[416:768]
        lowlock = None if lowraw == Zero352 else self.Unlock(lowraw)
        cell = Field.Cell(
            soul=self.DecodeName(meta[0:8]),
            key=key,
            salt=self.ReadU32(meta[40:44]),
            purge=Field.PurgeLocks(chainbit=purgebyte & 1, lockbit=(purgebyte >> 1) & 1),
            lock=lock,
            lowlock=lowlock,
        )
        Field.VerifyCell(cell)
        return cell

    def PackPurge(self, glyph: Any) -> bytes:
        key = self.PurgeKey(glyph)
        keyraw = bytes.fromhex(self.MustKey(key))
        return self.BuildHeader(KindPurge, key=keyraw)

    def UnpackPurge(self, data: bytes) -> dict[str, Any]:
        header = self.ParseHeader(data)
        if header['kind'] != KindPurge:
            raise ValueError('not a PurgeGlyph')
        key = header['key'].hex()
        self.MustKey(key)
        return {'kind': Dream.PurgeGlyph, 'key': key}

    def PackSalt(self, glyph: Field.SaltGlyph) -> bytes:
        Field.SaltGlyphShape(glyph)
        Field.VerifyTextBody(glyph.textbody, glyph.lockbody.texthash)
        Field.VerifyCanonicalText(glyph.textbody, glyph.lockbody.kind)
        wirekind = int(glyph.lockbody.kind) + 2
        if wirekind not in KindSalt:
            raise ValueError('receipt kind is not a SaltGlyph kind')
        out = self.BuildHeader(wirekind) + self.PadLock(glyph.lockbody) + self.PackText(glyph.textbody)
        if len(out) != SaltSize:
            raise AssertionError('SaltGlyph geometry drift')
        return out

    def UnpackSalt(self, data: bytes) -> Field.SaltGlyph:
        header = self.ParseHeader(data)
        kind = header['kind']
        if kind not in KindSalt:
            raise ValueError('not a SaltGlyph')
        lock = self.Unlock(data[64:416])
        if int(lock.kind) + 2 != kind:
            raise ValueError('wire Salt kind does not match PadLock receipt kind')
        textbody = self.UnpackText(data[416:512])
        Field.VerifyTextBody(textbody, lock.texthash)
        Field.VerifyCanonicalText(textbody, lock.kind)
        glyph = Field.SaltGlyph(lockbody=lock, textbody=textbody)
        Field.SaltGlyphShape(glyph)
        return glyph

    def PackNightmare(self, glyph: Field.NightmareGlyph) -> bytes:
        Field.VerifyNightmare(glyph)
        out = self.BuildHeader(KindNightmare) + self.PadLock(glyph.lowlock) + self.PadLock(glyph.lock)
        if len(out) != NightmareSize:
            raise AssertionError('NightmareGlyph geometry drift')
        return out

    def UnpackNightmare(self, data: bytes) -> Field.NightmareGlyph:
        header = self.ParseHeader(data)
        if header['kind'] != KindNightmare:
            raise ValueError('not a NightmareGlyph')
        glyph = Field.NightmareGlyph(
            lowlock=self.Unlock(data[64:416]),
            lock=self.Unlock(data[416:768]),
        )
        Field.VerifyNightmare(glyph)
        return glyph

    def PackDream(self, state: Field.State) -> bytes:
        Field.VerifyDream(state)
        scrubbed = Field.Scrub(state)
        header = self.BuildHeader(KindDream, flags=int(scrubbed.pristine))
        body = b''.join(self.PackCell(cell) for cell in scrubbed.cells)
        out = header + body
        if len(out) != DreamSize:
            raise AssertionError('DreamGlyph geometry drift')
        return out

    def UnpackDream(self, data: bytes) -> Field.State:
        header = self.ParseHeader(data)
        if header['kind'] != KindDream:
            raise ValueError('not a DreamGlyph')
        cells = []
        cursor = HeaderSize
        for _index in range(Field.SeatCount):
            cells.append(self.UnpackCell(data[cursor:cursor + CellSize]))
            cursor += CellSize
        if cursor != DreamSize:
            raise AssertionError('DreamGlyph geometry drift')
        state = Field.State(cells=tuple(cells), self=Field.Clean.self(), pristine=int(header['flags']))
        Field.VerifyDream(state)
        return state

    def PackSoulCard(self, soul: Soul) -> bytes:
        card = self.SoulShape(soul)
        if card is None:
            raise ValueError('invalid soul card')
        return self.EncodeName(card.soul) + bytes.fromhex(self.MustKey(card.key))

    def UnpackSoulCard(self, data: bytes) -> Soul:
        if len(data) != SoulCardSize:
            raise ValueError('soul card must be exactly 40 bytes')
        return Soul(soul=self.DecodeName(data[:8]), key=self.MustKey(data[8:40].hex()))

    def PackSouls(self, souls: Iterable[Any], *, expected: Optional[int] = None) -> bytes:
        expectedcount = self.genesisnumber if expected is None else self.MustGenesis(expected)
        cards = self.SoulSnap(souls)
        if cards is None:
            raise ValueError('invalid souls roster')
        cards = sorted(cards, key=lambda soul: soul.key)
        if len(cards) > expectedcount:
            raise ValueError('souls roster exceeds expected count')
        header = self.BuildHeader(KindSouls, expected=expectedcount, populated=len(cards))
        body = bytearray(Field.SeatCount * SoulCardSize)
        cursor = 0
        for card in cards:
            body[cursor:cursor + SoulCardSize] = self.PackSoulCard(card)
            cursor += SoulCardSize
        out = header + bytes(body)
        if len(out) != SoulsSize:
            raise AssertionError('SoulsGlyph geometry drift')
        return out

    def UnpackSouls(self, data: bytes) -> SoulsGlyph:
        header = self.ParseHeader(data)
        if header['kind'] != KindSouls:
            raise ValueError('not a SoulsGlyph')
        cards = []
        cursor = HeaderSize
        for index in range(Field.SeatCount):
            slot = data[cursor:cursor + SoulCardSize]
            if index < header['populated']:
                if slot == b'\x00' * SoulCardSize:
                    raise ValueError('populated soul card may not be zero')
                cards.append(self.UnpackSoulCard(slot))
            elif slot != b'\x00' * SoulCardSize:
                raise ValueError('unused soul card must be all zero')
            cursor += SoulCardSize
        if tuple(card.key for card in cards) != tuple(sorted(card.key for card in cards)):
            raise ValueError('SoulsGlyph cards must be sorted by key')
        if len({card.key for card in cards}) != len(cards):
            raise ValueError('SoulsGlyph contains duplicate keys')
        return SoulsGlyph(expected=int(header['expected']), souls=tuple(cards))

    def Pack(self, glyph: Any) -> bytes:
        if isinstance(glyph, SoulsGlyph):
            return self.PackSouls(glyph.souls, expected=glyph.expected)
        if isinstance(glyph, Field.State):
            return self.PackDream(glyph)
        if isinstance(glyph, Field.NightmareGlyph):
            return self.PackNightmare(glyph)
        if isinstance(glyph, Field.SaltGlyph):
            return self.PackSalt(glyph)
        if isinstance(glyph, (dict, str)):
            kind = self.LogicalKind(glyph)
            if kind == KindPurge:
                return self.PackPurge(glyph)
        raise TypeError(f'unsupported logical glyph: {type(glyph).__name__}')

    def Unpack(self, data: bytes) -> Any:
        header = self.ParseHeader(data)
        kind = header['kind']
        if kind == KindSouls:
            return self.UnpackSouls(data)
        if kind == KindDream:
            return self.UnpackDream(data)
        if kind == KindNightmare:
            return self.UnpackNightmare(data)
        if kind == KindPurge:
            return self.UnpackPurge(data)
        if kind in KindSalt:
            return self.UnpackSalt(data)
        raise ValueError('unknown canonical glyph kind')

    # ---------- routing / emission ----------

    def RouteGlyph(self, glyph: Any, addr: tuple[str, int]):
        if not self.genesisdone:
            return
        with self.ReapLock:
            if not self.Reaping:
                self.Reaping = True
                self.Reap = []
            self.Reap.append(glyph)
        self.glyph = glyph

    def EmitSouls(self):
        self.Emit(self.PackSouls(self.souls, expected=self.genesisnumber))

    def EmitCompleteSouls(self):
        souls = self.complete or tuple(self.souls)
        self.Emit(self.PackSouls(souls, expected=self.genesisnumber))

    def EmitGlyph(self, glyph: Any):
        self.Emit(self.Pack(glyph))

    def Emit(self, packet: Any):
        plain = bytes(packet) if isinstance(packet, (bytes, bytearray)) else self.Pack(packet)
        if len(plain) not in LegalSizes:
            raise ValueError('attempted to emit non-canonical glyph size')
        raw = self.Encrypt(plain)
        peers = self.Peers()
        for host, port in peers:
            for _shot in range(3):
                try:
                    self.sock.sendto(raw, (host, port))
                except Exception:
                    pass

    def Cast(self, packet: Any):
        return self.Emit(packet)

    # ---------- helpers ----------

    def BuildState(self) -> Baton:
        souls = tuple(self.complete) if self.genesisdone else tuple(self.souls)
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

    def PurgeKey(self, glyph: Any) -> str:
        if isinstance(glyph, str):
            return self.MustKey(glyph)
        if isinstance(glyph, dict):
            return self.MustKey(glyph.get('key', ''))
        return self.MustKey(getattr(glyph, 'key', ''))

    def LogicalKind(self, glyph: Any) -> int:
        if isinstance(glyph, Field.State):
            return KindDream
        if isinstance(glyph, Field.NightmareGlyph):
            return KindNightmare
        if isinstance(glyph, Field.SaltGlyph):
            return int(glyph.lockbody.kind) + 2
        if isinstance(glyph, SoulsGlyph):
            return KindSouls
        if isinstance(glyph, str):
            return KindPurge
        if isinstance(glyph, dict):
            raw = glyph.get('kind', '')
            if isinstance(raw, int) and raw in GlyphSizes:
                return int(raw)
            text = str(raw or '').strip().lower()
            names = {
                Dream.DreamGlyph: KindDream,
                Dream.NightmareGlyph: KindNightmare,
                Dream.PurgeGlyph: KindPurge,
                'whisper': KindWhisper,
                'rally': KindRally,
                'defect': KindDefect,
                'wrath': KindWrath,
                'souls': KindSouls,
            }
            if text in names:
                return names[text]
        raise ValueError('unknown logical glyph kind')

    def PacketHash(self, packet: Any) -> str:
        body = bytes(packet) if isinstance(packet, (bytes, bytearray)) else self.Pack(packet)
        return hashlib.sha256(body).hexdigest()

    def Encrypt(self, data: bytes) -> bytes:
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError('Encrypt expects canonical glyph bytes')
        body = bytes(data)
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        return bytes(byte ^ mask[index % len(mask)] for index, byte in enumerate(body))

    def Decrypt(self, raw: bytes) -> bytes:
        if not isinstance(raw, (bytes, bytearray)) or len(raw) not in LegalSizes:
            raise ValueError('encrypted glyph has non-canonical size')
        mask = hashlib.sha256(self.skeleton.encode('utf-8')).digest()
        return bytes(byte ^ mask[index % len(mask)] for index, byte in enumerate(bytes(raw)))

    def MustGenesis(self, value: Any) -> int:
        number = int(value or 1)
        if number < 1 or number > Field.SeatCount:
            raise ValueError(f'genesis must be 1..{Field.SeatCount} for fixed SoulsGlyph geometry')
        return number

    def MustName(self, value: Any) -> str:
        text = str(value or '').strip()
        if not (0 < len(text) <= NameMax):
            raise ValueError('invalid soul name')
        if len(text.encode('utf-8')) > NameMax:
            raise ValueError('soul name must fit fixed 8-byte UTF-8 field')
        return text

    def MustKey(self, value: Any) -> str:
        text = str(value or '').strip().lower()
        Field.VerifyKey(text)
        # A participant with a zero four-byte prefix is indistinguishable from payout padding.
        Field.PlayerTag(text)
        return text

    def MustTag(self, value: Any) -> str:
        text = str(value or '').strip().lower()
        Field.VerifyTag(text)
        if text == Field.ZeroTagHex:
            raise ValueError('zero tag is reserved for fixed-slot padding')
        return text

    def MustHash(self, value: Any) -> str:
        text = str(value or '').strip().lower()
        Field.VerifyHash(text, fieldname='hash')
        return text

    def MustSign(self, value: Any) -> str:
        text = str(value or '').strip().lower()
        Field.VerifySignHex(text, fieldname='sign')
        return text

    def AcceptSoulName(self, value: str) -> bool:
        try:
            return self.MustName(value) == str(value or '').strip()
        except Exception:
            return False

    def AcceptKey(self, value: str) -> bool:
        try:
            self.MustKey(value)
            return True
        except Exception:
            return False

    def AcceptSelf(self, value: Self) -> bool:
        return self.AcceptSoulName(value.soul) and self.AcceptKey(value.key)

    def AcceptSoulCard(self, value: Soul) -> bool:
        return self.AcceptSoulName(value.soul) and self.AcceptKey(value.key)


crypt = None
