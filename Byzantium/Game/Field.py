from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from typing import Iterable, Optional, Tuple
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

HashBytes = 32
HashHexLen = HashBytes * 2
KeyBytes = 32
KeyHexLen = KeyBytes * 2
TagHexLen = 8
ZeroTagHex = '0' * TagHexLen
SignBytes = 64
SignHexLen = SignBytes * 2
ZeroHashHex = '00' * HashBytes
NullSignHex = '00' * SignBytes
FileCount = 4
SeatsPerFile = 6
SeatCount = FileCount * SeatsPerFile
MillionInvariant = 1000000
TextHumanMax = 60
TextTagWidth = 8
TextWireMaxBytes = 95
TextMaxLen = TextHumanMax + TextTagWidth
KindEmpty = 0
KindWhisper = 1
KindRally = 2
KindDefect = 3
KindWrath = 4
KindNames = {
    KindEmpty: 'empty',
    KindWhisper: 'whisper',
    KindRally: 'rally',
    KindDefect: 'defect',
    KindWrath: 'wrath',
}
KindValues = {name: value for value, name in KindNames.items()}
KindSpendCounts = {
    KindEmpty: 0,
    KindWhisper: 1,
    KindRally: 5,
    KindDefect: 6,
    KindWrath: 23,
}
TextTags = {
    KindWhisper: '|whisper',
    KindRally: '|rally!!',
    KindDefect: '|defect!',
    KindWrath: '|wrath!!',
}
TagKinds = {tag: kind for kind, tag in TextTags.items()}

@dataclass(frozen=True)
class PurgeLocks:
    chainbit: int = 1
    lockbit: int = 1

    def __post_init__(self) -> None:
        if self.chainbit not in (0, 1):
            raise ValueError('chainbit must be 0 or 1')
        if self.lockbit not in (0, 1):
            raise ValueError('lockbit must be 0 or 1')

@dataclass(frozen=True)
class Payout:
    tag: str
    salt: int

    def __post_init__(self) -> None:
        VerifyTag(self.tag)
        if self.tag == ZeroTagHex:
            raise ValueError('payout tag must not be zero padding tag')
        VerifyNonNegative(self.salt, fieldname='payout.salt')

@dataclass(frozen=True)
class Lock:
    kind: int = KindEmpty
    tag: str = ''
    parent: str = ZeroHashHex
    child: str = ZeroHashHex
    payout: Tuple[Payout, ...] = ()
    texthash: str = ZeroHashHex
    sign: str = NullSignHex

    def __post_init__(self) -> None:
        VerifyKind(self.kind)
        if self.kind == KindEmpty and not self.tag:
            pass
        else:
            VerifyTag(self.tag)
        VerifyHash(self.parent, fieldname='lock.parent')
        VerifyHash(self.child, fieldname='lock.child')
        if not isinstance(self.payout, tuple):
            raise TypeError('lock.payout must be tuple')
        for leg in self.payout:
            if not isinstance(leg, Payout):
                raise TypeError('lock.payout must contain Payout objects')
        VerifyHash(self.texthash, fieldname='lock.texthash')
        VerifySignHex(self.sign, fieldname='lock.sign')

    @property
    def Empty(self) -> bool:
        return (
            self.kind == KindEmpty
            and self.parent == ZeroHashHex
            and self.child == ZeroHashHex
            and len(self.payout) == 0
            and self.texthash == ZeroHashHex
            and self.sign == NullSignHex
        )

class Clean:

    @staticmethod
    def purge_locks() -> PurgeLocks:
        return PurgeLocks(chainbit=0, lockbit=0)

    @staticmethod
    def lock(tag: str = '') -> Lock:
        return Lock(kind=KindEmpty, tag=str(tag or ''), parent=ZeroHashHex, child=ZeroHashHex,
                    payout=tuple(), texthash=ZeroHashHex, sign=NullSignHex)

    @staticmethod
    def self() -> Tuple[str, str]:
        return ('', '')

    @staticmethod
    def pristine() -> int:
        return 1

def GenesisLock(key: str) -> Lock:
    VerifyKey(key)
    return Clean.lock(PlayerTag(key))

@dataclass(frozen=True)
class Cell:
    soul: str
    key: str
    salt: int
    purge: PurgeLocks = field(default_factory=Clean.purge_locks)
    lock: Optional[Lock] = None
    lowlock: Optional[Lock] = None

    def __post_init__(self) -> None:
        if not isinstance(self.soul, str):
            raise TypeError('soul must be str')
        VerifyKey(self.key)
        VerifyNonNegative(self.salt, fieldname='cell.salt')
        if not isinstance(self.purge, PurgeLocks):
            raise TypeError('cell.purge must be PurgeLocks')
        if self.lock is None:
            object.__setattr__(self, 'lock', GenesisLock(self.key))
        if not isinstance(self.lock, Lock):
            raise TypeError('cell.lock must be Lock')
        if self.lowlock is not None and not isinstance(self.lowlock, Lock):
            raise TypeError('cell.lowlock must be Lock or None')

@dataclass(frozen=True)
class State:
    cells: Tuple[Cell, ...]
    self: Tuple[str, str] = field(default_factory=Clean.self)
    pristine: int = field(default_factory=Clean.pristine)

    def __post_init__(self) -> None:
        if len(self.cells) != SeatCount:
            raise ValueError(f'state must contain exactly {SeatCount} seats')
        for cell in self.cells:
            VerifyCell(cell)
        VerifySelf(self.self)
        VerifyBit(self.pristine, fieldname='state.pristine')

    @property
    def SaltTotal(self) -> int:
        return sum((int(cell.salt) for cell in self.cells))

@dataclass(frozen=True)
class Text:
    text: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError('text.text must be str')
        if len(self.text) > TextMaxLen:
            raise ValueError(f'text.text must be at most {TextMaxLen} chars')

@dataclass(frozen=True)
class SaltGlyph:
    lockbody: Lock
    textbody: Text

    def __post_init__(self) -> None:
        if not isinstance(self.lockbody, Lock):
            raise TypeError('saltglyph.lockbody must be Lock')
        if not isinstance(self.textbody, Text):
            raise TypeError('saltglyph.textbody must be Text')

@dataclass(frozen=True)
class NightmareGlyph:
    lowlock: Lock
    lock: Lock

    def __post_init__(self) -> None:
        VerifyNightmare(self)

@dataclass(frozen=True)
class Chain:
    linked: bool
    relation: str = 'reject'
    open: bool = False
    reason: str = ''

def KindValue(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError('kind must not be bool')
    if isinstance(value, int):
        VerifyKind(value)
        return int(value)
    text = str(value or '').strip().lower()
    if text.isdigit():
        number = int(text)
        VerifyKind(number)
        return number
    if text not in KindValues:
        raise ValueError(f'unknown receipt kind: {value!r}')
    return KindValues[text]

def KindName(value: object) -> str:
    return KindNames[KindValue(value)]

def TextTag(value: object) -> str:
    kind = KindValue(value)
    if kind not in TextTags:
        raise ValueError('empty receipt kind has no Salt text tag')
    tag = TextTags[kind]
    if len(tag) != TextTagWidth:
        raise ValueError('Salt text tag geometry corrupted')
    return tag

def SplitText(value: object) -> tuple[str, str]:
    raw = str(value or '').replace('\r', ' ').replace('\n', ' ')
    if len(raw) < TextTagWidth:
        return ('', raw)
    tag = raw[-TextTagWidth:]
    kind = TagKinds.get(tag)
    if kind is None:
        return ('', raw)
    return (KindName(kind), raw[:-TextTagWidth])

def CanonicalText(value: object, kind: object) -> 'Text':
    body = str(value or '').replace('\r', ' ').replace('\n', ' ')[:TextHumanMax]
    tag = TextTag(kind)
    while body and len((body + tag).encode('utf-8')) > TextWireMaxBytes:
        body = body[:-1]
    packet = body + tag
    if len(packet.encode('utf-8')) > TextWireMaxBytes:
        raise ValueError('Salt text packet exceeds fixed wire text brick')
    return Text(text=packet)

def VerifyCanonicalText(textbody: 'Text', kind: object) -> bool:
    if not isinstance(textbody, Text):
        raise TypeError('expected Text')
    action, body = SplitText(textbody.text)
    expected = KindName(kind)
    if action != expected:
        raise ValueError(f'Salt text tag must be canonical {TextTag(kind)!r}')
    if len(body) > TextHumanMax:
        raise ValueError(f'Salt human text must be at most {TextHumanMax} chars')
    if len(textbody.text.encode('utf-8')) > TextWireMaxBytes:
        raise ValueError('Salt text packet exceeds fixed wire text brick')
    return True

def VerifyKind(value: int) -> bool:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError('kind must be int')
    if value not in KindNames:
        raise ValueError('kind must be 0..4')
    return True

def PlayerTag(key: str) -> str:
    VerifyKey(key)
    tag = key[:TagHexLen].lower()
    if tag == ZeroTagHex:
        raise ValueError('player key derives reserved zero padding tag')
    return tag

def VerifyTag(value: str) -> bool:
    if not isinstance(value, str):
        raise TypeError('tag must be a hex string')
    if len(value) != TagHexLen:
        raise ValueError(f'tag must be exactly {TagHexLen} hex chars')
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('tag must be valid hex') from exc
    return True

def VerifyState(state: State, *, expectedkeys: Optional[Iterable[str]]=None, expectedsalt: int=MillionInvariant) -> State:
    if not isinstance(state, State):
        raise TypeError('expected State')
    if len(state.cells) != SeatCount:
        raise ValueError(f'state must contain exactly {SeatCount} seats')
    VerifySelf(state.self)
    VerifyBit(state.pristine, fieldname='state.pristine')
    keys = []
    tags = []
    for cell in state.cells:
        VerifyCell(cell)
        keys.append(cell.key)
        tags.append(PlayerTag(cell.key))
    VerifyNonNegative(expectedsalt, fieldname='expectedsalt')
    if state.SaltTotal != int(expectedsalt):
        raise ValueError(f'million invariant violated: total={state.SaltTotal} expected={expectedsalt}')
    if len(set(keys)) != SeatCount:
        raise ValueError('key invariant violated: duplicate keys in state')
    if len(set(tags)) != SeatCount:
        raise ValueError('player tag invariant violated: duplicate 8-character tags in state')
    if expectedkeys is not None:
        known = tuple(expectedkeys)
        if len(known) != SeatCount:
            raise ValueError(f'expectedkeys must contain exactly {SeatCount} keys')
        for key in known:
            VerifyKey(key)
        if set(keys) != set(known):
            raise ValueError('key invariant violated: unknown or missing keys')
    return state

def VerifyCell(cell: Cell) -> Cell:
    if not isinstance(cell, Cell):
        raise TypeError('expected Cell')
    if not isinstance(cell.soul, str):
        raise TypeError('cell.soul must be str')
    if not isinstance(cell.purge, PurgeLocks):
        raise TypeError('cell.purge must be PurgeLocks')
    if not isinstance(cell.lock, Lock):
        raise TypeError('cell.lock must be Lock')
    if cell.lowlock is not None and not isinstance(cell.lowlock, Lock):
        raise TypeError('cell.lowlock must be Lock or None')
    VerifyKey(cell.key)
    VerifyNonNegative(cell.salt, fieldname='cell.salt')
    expectedtag = PlayerTag(cell.key)
    if cell.lock.tag != expectedtag:
        raise ValueError('cell.lock tag does not match cell key')
    VerifyLockShape(cell.lock, allowempty=True)
    if cell.lowlock is not None:
        if cell.lowlock.tag != expectedtag:
            raise ValueError('cell.lowlock tag does not match cell key')
        VerifyLockShape(cell.lowlock, allowempty=False)
        if cell.lock.kind == KindEmpty:
            raise ValueError('cell.lowlock requires a non-empty cell.lock')
        if cell.lowlock.parent != cell.lock.parent:
            raise ValueError('cell LockSet must share one parent')
        if cell.lowlock.child >= cell.lock.child:
            raise ValueError('cell.lowlock must be the lower child')
    return cell

def VerifySelf(value: Tuple[str, str]) -> Tuple[str, str]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError('state.self must be a (soul, key) tuple')
    soul, key = value
    if not isinstance(soul, str):
        raise TypeError('state.self soul must be str')
    if key:
        VerifyKey(key)
    return value

def VerifyHash(value: str, *, fieldname: str='hash') -> bool:
    if not isinstance(value, str):
        raise TypeError(f'{fieldname} must be a hex string')
    if len(value) != HashHexLen:
        raise ValueError(f'{fieldname} must be exactly {HashBytes} bytes / {HashHexLen} hex chars')
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f'{fieldname} must be valid hex') from exc
    return True

def VerifyKey(value: str) -> bool:
    if not isinstance(value, str):
        raise TypeError('key must be a hex string')
    if len(value) != KeyHexLen:
        raise ValueError(f'key must be exactly {KeyBytes} bytes / {KeyHexLen} hex chars')
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('key must be valid hex') from exc
    return True

def VerifySignHex(value: str, *, fieldname: str='sign') -> bool:
    if not isinstance(value, str):
        raise TypeError(f'{fieldname} must be a hex string')
    if value == NullSignHex:
        return True
    if len(value) != SignHexLen:
        raise ValueError(f'{fieldname} must be exactly {SignBytes} bytes / {SignHexLen} hex chars')
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f'{fieldname} must be valid hex') from exc
    return True

def VerifyNonNegative(value: int, *, fieldname: str='value') -> bool:
    if not isinstance(value, int):
        raise TypeError(f'{fieldname} must be an integer')
    if value < 0:
        raise ValueError(f'{fieldname} must be non-negative')
    return True

def VerifyBit(value: int, *, fieldname: str='bit') -> bool:
    if not isinstance(value, int):
        raise TypeError(f'{fieldname} must be an integer')
    if value not in (0, 1):
        raise ValueError(f'{fieldname} must be 0 or 1')
    return True

def VerifySign(key: str, digest: str, sign: str, *, allownull: bool=False) -> bool:
    VerifyKey(key)
    VerifyHash(digest, fieldname='digest')
    VerifySignHex(sign, fieldname='sign')
    if sign == NullSignHex:
        return bool(allownull)
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(key))
        pubkey.verify(bytes.fromhex(sign), bytes.fromhex(digest))
        return True
    except (InvalidSignature, ValueError):
        return False

def TextBody(textbody: Text) -> bytes:
    if not isinstance(textbody, Text):
        raise TypeError('expected Text')
    return str(textbody.text).encode('utf-8')

def TextHash(textbody: Text) -> str:
    return hashlib.sha256(TextBody(textbody)).hexdigest()

def ChildBody(kind: int, tag: str, parent: str, payout: Tuple[Payout, ...], texthash: str) -> bytes:
    VerifyKind(kind)
    VerifyTag(tag)
    VerifyHash(parent, fieldname='parent')
    VerifyHash(texthash, fieldname='texthash')
    parts = [str(int(kind)), tag, parent]
    for leg in payout:
        if not isinstance(leg, Payout):
            raise TypeError('payout must contain Payout objects')
        parts.extend([leg.tag, str(int(leg.salt))])
    parts.append(texthash)
    return '|'.join(parts).encode('utf-8')

def ChildHex(kind: int, tag: str, parent: str, payout: Tuple[Payout, ...], texthash: str) -> str:
    return hashlib.sha256(ChildBody(kind, tag, parent, payout, texthash)).hexdigest()

def DerivedChild(lock: Lock) -> str:
    if not isinstance(lock, Lock):
        raise TypeError('expected Lock')
    if lock.kind == KindEmpty:
        return ZeroHashHex
    return ChildHex(lock.kind, lock.tag, lock.parent, lock.payout, lock.texthash)

def LockHash(lock: Lock) -> str:
    if not isinstance(lock, Lock):
        raise TypeError('expected Lock')
    VerifyHash(lock.parent, fieldname='lock.parent')
    VerifyHash(lock.child, fieldname='lock.child')
    body = bytes.fromhex(lock.parent) + bytes.fromhex(lock.child)
    return hashlib.sha256(body).hexdigest()

def ReceiptBody(lock: Lock, *, includesign: bool=True) -> bytes:
    VerifyLockShape(lock, allowempty=True)
    parts = [
        str(int(lock.kind)), lock.tag, lock.parent, lock.child,
    ]
    for leg in lock.payout:
        parts.extend([leg.tag, str(int(leg.salt))])
    parts.append(lock.texthash)
    if includesign:
        parts.append(lock.sign)
    return '|'.join(parts).encode('utf-8')

def ReceiptHash(lock: Lock) -> str:
    return hashlib.sha256(ReceiptBody(lock, includesign=True)).hexdigest()

def VerifyLockShape(lock: Lock, *, allowempty: bool=False) -> Lock:
    if not isinstance(lock, Lock):
        raise TypeError('expected Lock')
    VerifyKind(lock.kind)
    if lock.kind == KindEmpty:
        if not allowempty:
            raise ValueError('empty lock not allowed here')
        if lock.payout:
            raise ValueError('empty lock must have no payout')
        if lock.parent != ZeroHashHex or lock.child != ZeroHashHex:
            raise ValueError('empty lock must have zero parent/child')
        if lock.texthash != ZeroHashHex or lock.sign != NullSignHex:
            raise ValueError('empty lock must have zero texthash/sign')
        if lock.tag:
            VerifyTag(lock.tag)
        return lock
    VerifyTag(lock.tag)
    if lock.tag == ZeroTagHex:
        raise ValueError('receipt tag must not be zero padding tag')
    expected = KindSpendCounts[lock.kind]
    if len(lock.payout) != expected:
        raise ValueError(f'{KindName(lock.kind)} requires exactly {expected} payout legs')
    for leg in lock.payout:
        if not isinstance(leg, Payout):
            raise TypeError('lock.payout must contain Payout objects')
        VerifyTag(leg.tag)
        VerifyNonNegative(leg.salt, fieldname='payout.salt')
    VerifyHash(lock.parent, fieldname='lock.parent')
    VerifyHash(lock.child, fieldname='lock.child')
    VerifyHash(lock.texthash, fieldname='lock.texthash')
    VerifySignHex(lock.sign, fieldname='lock.sign')
    return lock

def VerifyLock(key: str, lock: Lock, sign: Optional[str]=None, *, allownull: bool=False) -> bool:
    VerifyKey(key)
    VerifyLockShape(lock, allowempty=True)
    if lock.tag and lock.tag != PlayerTag(key):
        raise ValueError('lock tag does not match signer key')
    actualsign = lock.sign if sign is None else str(sign or '')
    VerifySignHex(actualsign, fieldname='sign')
    if lock.kind == KindEmpty:
        return bool(allownull or actualsign == NullSignHex)
    derived = DerivedChild(lock)
    if derived != lock.child:
        raise ValueError('lock child does not reconstruct from receipt core')
    digest = LockHash(lock)
    if actualsign == NullSignHex:
        return bool(allownull)
    if not VerifySign(key, digest, actualsign, allownull=False):
        raise ValueError('lock sign failed verification')
    return True

def VerifyTextBody(textbody: Text, texthashvalue: str) -> bool:
    if not isinstance(textbody, Text):
        raise TypeError('expected Text')
    VerifyHash(texthashvalue, fieldname='texthash')
    if TextHash(textbody) != texthashvalue:
        raise ValueError('texthash mismatch')
    return True

def SaltGlyphShape(glyph: SaltGlyph) -> SaltGlyph:
    if not isinstance(glyph, SaltGlyph):
        raise TypeError('expected SaltGlyph')
    VerifyLockShape(glyph.lockbody, allowempty=False)
    if not isinstance(glyph.textbody, Text):
        raise TypeError('saltglyph.textbody must be Text')
    return glyph

def VerifySalt(glyph: SaltGlyph, state: Optional[State]=None, *, key: str='') -> SaltGlyph:
    SaltGlyphShape(glyph)
    VerifyTextBody(glyph.textbody, glyph.lockbody.texthash)
    VerifyCanonicalText(glyph.textbody, glyph.lockbody.kind)
    signerkey = str(key or '').strip()
    if not signerkey and state is not None:
        cell = FindCell(state, glyph.lockbody.tag)
        if cell is None:
            raise ValueError('glyph signer tag not found in state')
        signerkey = cell.key
    if not signerkey:
        raise ValueError('full signer key required to verify SaltGlyph')
    VerifyLock(signerkey, glyph.lockbody)
    return glyph

def VerifyNightmare(glyph: NightmareGlyph, state: Optional[State]=None, *, key: str='') -> NightmareGlyph:
    if not isinstance(glyph, NightmareGlyph):
        raise TypeError('expected NightmareGlyph')
    low = glyph.lowlock
    high = glyph.lock
    VerifyLockShape(low, allowempty=False)
    VerifyLockShape(high, allowempty=False)
    if low.tag != high.tag:
        raise ValueError('nightmare receipts must share one signer')
    if low.parent != high.parent:
        raise ValueError('nightmare receipts must share one parent')
    if low.child >= high.child:
        raise ValueError('nightmare receipts must be ordered low to high')
    signerkey = str(key or '').strip()
    if not signerkey and state is not None:
        cell = FindCell(state, low.tag)
        if cell is None:
            raise ValueError('nightmare signer tag not found in state')
        signerkey = cell.key
    if signerkey:
        VerifyLock(signerkey, low)
        VerifyLock(signerkey, high)
    return glyph

def VerifyDream(state: State, *, expectedkeys: Optional[Iterable[str]]=None) -> State:
    verified = VerifyState(Scrub(state), expectedkeys=expectedkeys)
    for cell in verified.cells:
        if cell.lock.kind != KindEmpty:
            VerifyLock(cell.key, cell.lock)
        if cell.lowlock is not None:
            VerifyLock(cell.key, cell.lowlock)
    return verified

def LockTotal(lock: Lock) -> int:
    VerifyLockShape(lock, allowempty=True)
    return sum((int(leg.salt) for leg in lock.payout))

def SeatRange(filenumber: int) -> range:
    VerifyNonNegative(filenumber, fieldname='filenumber')
    if filenumber < 1 or filenumber > FileCount:
        raise ValueError(f'filenumber must be 1..{FileCount}')
    firstseat = (filenumber - 1) * SeatsPerFile + 1
    lastseat = firstseat + SeatsPerFile - 1
    return range(firstseat, lastseat + 1)

def FileCells(state: State, filenumber: int) -> Tuple[Cell, ...]:
    VerifyState(state)
    seats = SeatRange(filenumber)
    return tuple((state.cells[seat - 1] for seat in seats))

def SortFile(filecellsvalue: Iterable[Cell]) -> Tuple[Cell, ...]:
    ordered = tuple(sorted(tuple(filecellsvalue), key=lambda cell: (-int(cell.salt), str(cell.key))))
    if len(ordered) != SeatsPerFile:
        raise ValueError(f'file must contain exactly {SeatsPerFile} seats')
    return ordered

def LockSet(cell: Cell) -> Tuple[Lock, ...]:
    VerifyCell(cell)
    if cell.lowlock is not None:
        return (cell.lowlock, cell.lock)
    if cell.lock.kind != KindEmpty:
        return (cell.lock,)
    return tuple()

def Burned(cell: Cell) -> bool:
    VerifyCell(cell)
    return bool(int(cell.salt) == 0 and cell.lowlock is not None)

def CanonicalLocks(*locks: Lock) -> Tuple[Lock, ...]:
    unique: dict[str, Lock] = {}
    tag = ''
    parent = ''
    for lock in locks:
        VerifyLockShape(lock, allowempty=False)
        if not tag:
            tag = lock.tag
            parent = lock.parent
        if lock.tag != tag or lock.parent != parent:
            raise ValueError('canonical locks must share signer and parent')
        unique[lock.child] = lock
    return tuple(sorted(unique.values(), key=lambda item: item.child)[:2])

def ContinuationChild(cell: Cell) -> str:
    VerifyCell(cell)
    return cell.lowlock.child if cell.lowlock is not None else cell.lock.child

def CellIndex(state: State, identity: str) -> Optional[int]:
    target = FindCell(state, identity)
    if target is None:
        return None
    for index, cell in enumerate(state.cells):
        if cell.key == target.key:
            return index
    return None

def DefectParts(lock: Lock) -> Tuple[Tuple[Payout, ...], Optional[Payout]]:
    legs = tuple(lock.payout)
    if lock.kind != KindDefect or len(legs) != KindSpendCounts[KindDefect]:
        return (legs, None)
    return (legs[:-1], legs[-1])

def DefectViable(state: State, lock: Lock) -> bool:
    VerifyState(state)
    VerifyLockShape(lock, allowempty=False)
    spend, victim = DefectParts(lock)
    if victim is None:
        return True
    signerq = CellIndex(state, lock.tag)
    victimq = CellIndex(state, victim.tag)
    if signerq is None or victimq is None or signerq == victimq or (signerq // SeatsPerFile) == (victimq // SeatsPerFile):
        return False
    signer = state.cells[signerq]
    target = state.cells[victimq]
    if int(target.salt) >= int(signer.salt):
        return False
    floor = 10000 if signerq % SeatsPerFile == 0 else 1000
    if sum(int(leg.salt) for leg in spend) != floor:
        return False
    filetags = {PlayerTag(cell.key) for index, cell in enumerate(state.cells) if index != signerq and (index // SeatsPerFile) == (signerq // SeatsPerFile)}
    if {leg.tag for leg in spend} != filetags or int(victim.salt) != 0:
        return False
    for leg in spend:
        recipient = FindCell(state, leg.tag)
        if recipient is None or (Burned(recipient) and int(leg.salt) > 0):
            return False
    return True

def Swap(state: State, first: str, second: str) -> State:
    a = CellIndex(state, first)
    b = CellIndex(state, second)
    if a is None or b is None or a == b:
        raise ValueError('swap requires two known actors')
    cells = list(state.cells)
    cells[a], cells[b] = cells[b], cells[a]
    return State(cells=tuple(cells), self=state.self, pristine=state.pristine)

def Transfer(state: State, sourceid: str, targetid: str, amount: int) -> State:
    source = FindCell(state, sourceid)
    target = FindCell(state, targetid)
    amount = int(amount)
    if source is None or target is None or source.key == target.key or amount < 0 or source.salt < amount:
        raise ValueError('invalid transfer')
    replacements = {
        source.key: replace(source, salt=source.salt - amount),
        target.key: replace(target, salt=target.salt + amount),
    }
    return State(cells=tuple(replacements.get(cell.key, cell) for cell in state.cells), self=state.self, pristine=state.pristine)

def ApplyEffect(state: State, lock: Lock, *, verifydefect: bool=True) -> Tuple[State, Tuple[Chain, ...]]:
    VerifyState(state)
    signer = FindCell(state, lock.tag)
    if signer is None:
        raise ValueError('receipt signer tag not found in state')
    debit = LockTotal(lock)
    if debit > signer.salt:
        raise ValueError('receipt debit exceeds signer salt')
    if verifydefect and lock.kind == KindDefect and not DefectViable(state, lock):
        raise ValueError('defect geometry is not viable')
    replacements: dict[str, Cell] = {signer.key: replace(signer, salt=signer.salt - debit)}
    credits: dict[str, int] = {}
    for leg in lock.payout:
        target = FindCell(state, leg.tag)
        if target is None:
            raise ValueError('payout tag not found in state')
        if int(leg.salt) > 0 and Burned(target):
            raise ValueError('positive payout to burned actor')
        credits[target.key] = credits.get(target.key, 0) + int(leg.salt)
    chains = [Chain(linked=True, relation='Link', reason='debit')]
    for key, amount in credits.items():
        target = FindCell(state, key)
        base = replacements.get(key, target)
        replacements[key] = replace(base, salt=base.salt + int(amount))
        chains.append(Chain(linked=True, relation='Link', reason='credit'))
    candidate = State(cells=tuple(replacements.get(cell.key, cell) for cell in state.cells), self=state.self, pristine=state.pristine)
    if lock.kind == KindDefect:
        _spend, victim = DefectParts(lock)
        if victim is None:
            raise ValueError('defect requires swap victim')
        candidate = Swap(candidate, lock.tag, victim.tag)
    VerifyState(candidate, expectedkeys=FindKeys(state))
    return (candidate, tuple(chains))

def Trace(state: State, tag: str, needed: int, unwound: set[str], trail: set[str]) -> Optional[State]:
    cell = FindCell(state, tag)
    if cell is None:
        return None
    if cell.salt >= int(needed):
        return state
    if cell.key in trail:
        return None
    nexttrail = set(trail)
    nexttrail.add(cell.key)
    candidate = state
    for lock in sorted(LockSet(cell), key=lambda item: item.child):
        receiptid = ReceiptHash(lock)
        if receiptid in unwound:
            continue
        before = candidate
        beforeunwound = set(unwound)
        repaired = Revoke(candidate, lock, unwound=unwound, trail=nexttrail)
        if repaired is None:
            candidate = before
            unwound.clear()
            unwound.update(beforeunwound)
            continue
        candidate = repaired
        refreshed = FindCell(candidate, tag)
        if refreshed is not None and refreshed.salt >= int(needed):
            return candidate
    return None

def Revoke(state: State, lock: Lock, *, unwound: Optional[set[str]]=None, trail: Optional[set[str]]=None) -> Optional[State]:
    VerifyState(state)
    VerifyLockShape(lock, allowempty=False)
    unwound = set() if unwound is None else unwound
    trail = set() if trail is None else trail
    receiptid = ReceiptHash(lock)
    if receiptid in unwound:
        return state
    candidate = state
    if lock.kind == KindDefect:
        _spend, victim = DefectParts(lock)
        if victim is None:
            return None
        try:
            candidate = Swap(candidate, lock.tag, victim.tag)
        except Exception:
            return None
    credits: dict[str, int] = {}
    for leg in lock.payout:
        if int(leg.salt) <= 0:
            continue
        credits[leg.tag] = credits.get(leg.tag, 0) + int(leg.salt)
    for tag, amount in credits.items():
        repaired = Trace(candidate, tag, amount, unwound, trail)
        if repaired is None:
            return None
        candidate = repaired
        target = FindCell(candidate, tag)
        source = FindCell(candidate, lock.tag)
        if target is None or source is None or target.salt < amount:
            return None
        try:
            candidate = Transfer(candidate, tag, lock.tag, amount)
        except Exception:
            return None
    unwound.add(receiptid)
    VerifyState(candidate, expectedkeys=FindKeys(state))
    return candidate

def BurnShares(pair: Tuple[Lock, Lock], estate: int) -> dict[str, int]:
    claims = []
    for lock in pair:
        for index, leg in enumerate(lock.payout):
            if int(leg.salt) > 0:
                claims.append((lock.child, leg.tag, int(leg.salt), index))
    total = sum(item[2] for item in claims)
    if estate <= 0 or total <= 0:
        return {}
    rows = []
    paid = 0
    for child, tag, amount, index in claims:
        numerator = int(estate) * amount
        share, remainder = divmod(numerator, total)
        rows.append([child, tag, share, remainder, index])
        paid += share
    leftover = int(estate) - paid
    rows.sort(key=lambda row: (-row[3], row[0], row[1], row[4]))
    for index in range(leftover):
        rows[index][2] += 1
    payouts: dict[str, int] = {}
    for _child, tag, share, _remainder, _index in rows:
        payouts[tag] = payouts.get(tag, 0) + int(share)
    return payouts

def Burn(state: State, pair: Tuple[Lock, Lock]) -> State:
    VerifyState(state)
    low, high = CanonicalLocks(*pair)
    signer = FindCell(state, low.tag)
    if signer is None:
        raise ValueError('burn signer not found')
    estate = int(signer.salt)
    if LockTotal(low) + LockTotal(high) <= estate:
        raise ValueError('burn requires insolvency')
    payouts = BurnShares((low, high), estate)
    replacements: dict[str, Cell] = {signer.key: replace(signer, salt=0, lowlock=low, lock=high)}
    for tag, amount in payouts.items():
        target = FindCell(state, tag)
        if target is None:
            raise ValueError('burn payout tag not found')
        base = replacements.get(target.key, target)
        replacements[target.key] = replace(base, salt=base.salt + int(amount))
    candidate = State(cells=tuple(replacements.get(cell.key, cell) for cell in state.cells), self=state.self, pristine=state.pristine)
    VerifyState(candidate, expectedkeys=FindKeys(state))
    return candidate

def SetLockSet(state: State, tag: str, locks: Tuple[Lock, ...]) -> State:
    signer = FindCell(state, tag)
    if signer is None:
        raise ValueError('lockset signer not found')
    if len(locks) == 1:
        replacement = replace(signer, lock=locks[0], lowlock=None)
    elif len(locks) == 2:
        low, high = CanonicalLocks(*locks)
        replacement = replace(signer, lowlock=low, lock=high)
    else:
        raise ValueError('lockset must contain one or two receipts')
    return ReplaceCell(state, replacement)

def Adopt(state: State, evidence: Lock | Iterable[Lock]) -> Tuple[State, Tuple[Chain, ...]]:
    VerifyState(state)
    incoming = (evidence,) if isinstance(evidence, Lock) else tuple(evidence)
    if not incoming:
        raise ValueError('Adopt requires receipt evidence')
    first = incoming[0]
    signer = FindCell(state, first.tag)
    if signer is None:
        raise ValueError('receipt signer tag not found in state')
    for lock in incoming:
        VerifyLock(signer.key, lock)
        if lock.tag != first.tag:
            raise ValueError('Adopt evidence must share one signer')
    current = LockSet(signer)
    currentchildren = {lock.child for lock in current}
    if len(incoming) == 1 and first.child in currentchildren:
        return (state, (Chain(linked=True, relation='Link', reason='idempotent'),))
    if Burned(signer):
        return (state, (Chain(linked=False, relation='reject', open=True, reason='burned actor'),))

    if len(incoming) == 1 and first.parent == ContinuationChild(signer):
        if LockTotal(first) > signer.salt:
            return (state, (Chain(linked=False, relation='reject', open=True, reason='insolvent continuation'),))
        candidate, chains = ApplyEffect(state, first)
        candidate = SetLockSet(candidate, first.tag, (first,))
        return (candidate, chains)

    parents = {lock.parent for lock in incoming}
    if len(parents) != 1:
        return (state, (Chain(linked=False, relation='reject', open=True, reason='nightmare parent mismatch'),))
    parent = next(iter(parents))
    relevant = tuple(lock for lock in current if lock.parent == parent)
    if not relevant and signer.lock.kind != KindEmpty:
        return (state, (Chain(linked=False, relation='reject', open=True, reason='stale fork'),))
    candidates = CanonicalLocks(*(relevant + incoming))
    if len(candidates) != 2:
        return (state, (Chain(linked=False, relation='reject', open=True, reason='no competing pair'),))
    if len(current) == 2 and tuple(current) == candidates:
        return (state, (Chain(linked=True, relation='Link', reason='idempotent nightmare'),))
    if signer.lowlock is not None and signer.salt == 0:
        return (state, (Chain(linked=False, relation='reject', open=True, reason='zero LockSet re-settlement is undefined'),))

    candidate = state
    unwound: set[str] = set()
    for old in relevant:
        repaired = Revoke(candidate, old, unwound=unwound, trail={signer.key})
        if repaired is None:
            return (state, (Chain(linked=False, relation='reject', open=True, reason='trace failed'),))
        candidate = repaired

    source = FindCell(candidate, first.tag)
    if source is None:
        return (state, (Chain(linked=False, relation='reject', open=True, reason='signer vanished'),))
    obligation = sum(LockTotal(lock) for lock in candidates)
    if obligation > source.salt:
        candidate = Burn(candidate, (candidates[0], candidates[1]))
        return (candidate, (Chain(linked=True, relation='Link', reason='burn'),))
    if any(lock.kind == KindDefect and not DefectViable(candidate, lock) for lock in candidates):
        return (state, (Chain(linked=False, relation='reject', open=True, reason='defect geometry is not viable'),))

    chains: Tuple[Chain, ...] = tuple()
    for lock in reversed(candidates):
        candidate, lockchains = ApplyEffect(candidate, lock, verifydefect=False)
        chains = lockchains
    candidate = SetLockSet(candidate, first.tag, candidates)
    return (candidate, chains or (Chain(linked=True, relation='Link', reason='adopt'),))

def Purge(state: State) -> State:
    VerifyState(state)
    return State(
        cells=tuple((replace(cell, purge=Clean.purge_locks()) for cell in state.cells)),
        self=state.self,
        pristine=state.pristine,
    )

def Scrub(state: State) -> State:
    purged = Purge(state)
    return State(
        cells=purged.cells,
        self=Clean.self(),
        pristine=purged.pristine,
    )

def Stasis(state: State) -> State:
    VerifyState(state)
    ordered = []
    for filenumber in range(1, FileCount + 1):
        ordered.extend(SortFile(FileCells(state, filenumber)))
    return State(cells=tuple(ordered), self=state.self, pristine=state.pristine)

def FindKeys(state: State) -> Tuple[str, ...]:
    VerifyState(state)
    return tuple((cell.key for cell in state.cells))

def FindCell(state: State, identity: str) -> Optional[Cell]:
    VerifyState(state)
    token = str(identity or '').strip().lower()
    if len(token) == KeyHexLen:
        VerifyKey(token)
        for cell in state.cells:
            if cell.key.lower() == token:
                return cell
        return None
    VerifyTag(token)
    for cell in state.cells:
        if PlayerTag(cell.key) == token:
            return cell
    return None

def ReplaceCell(state: State, replacement: Cell) -> State:
    VerifyState(state)
    VerifyCell(replacement)
    cells = []
    found = False
    for cell in state.cells:
        if cell.key == replacement.key:
            cells.append(replacement)
            found = True
        else:
            cells.append(cell)
    if not found:
        raise ValueError('replacement key not found in state')
    return State(cells=tuple(cells), self=state.self, pristine=state.pristine)
