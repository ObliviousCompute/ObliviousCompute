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
SignBytes = 64
SignHexLen = SignBytes * 2
ZeroHashHex = '00' * HashBytes
NullSignHex = '00' * SignBytes
FileCount = 4
SeatsPerFile = 6
SeatCount = FileCount * SeatsPerFile
MillionInvariant = 1000000
SaltGlyphSpendCounts = {1, 5, 6, 23}
TextMaxLen = 68

@dataclass(frozen=True)
class Purge:
    chainbit: int = 1
    lockbit: int = 1

    def __post_init__(self) -> None:
        if self.chainbit not in (0, 1):
            raise ValueError('chainbit must be 0 or 1')
        if self.lockbit not in (0, 1):
            raise ValueError('lockbit must be 0 or 1')

@dataclass(frozen=True)
class Lock:
    parent: str = ZeroHashHex
    child: str = ZeroHashHex

    def __post_init__(self) -> None:
        VerifyHash(self.parent, fieldname='lock.parent')
        VerifyHash(self.child, fieldname='lock.child')

class Clean:

    @staticmethod
    def purge() -> Purge:
        return Purge(chainbit=0, lockbit=0)

    @staticmethod
    def lock() -> Lock:
        return Lock(parent=ZeroHashHex, child=ZeroHashHex)

    @staticmethod
    def sign() -> str:
        return NullSignHex

    @staticmethod
    def self() -> Tuple[str, str]:
        return ('', '')

    @staticmethod
    def monument() -> Tuple[str, ...]:
        return tuple()

@dataclass(frozen=True)
class Cell:
    soul: str
    key: str
    salt: int
    purge: Purge = field(default_factory=Clean.purge)
    lock: Lock = field(default_factory=Clean.lock)
    sign: str = field(default_factory=Clean.sign)

    def __post_init__(self) -> None:
        if not isinstance(self.soul, str):
            raise TypeError('soul must be str')
        VerifyKey(self.key)
        VerifyNonNegative(self.salt, fieldname='cell.salt')
        VerifySignHex(self.sign, fieldname='cell.sign')
        if not isinstance(self.purge, Purge):
            raise TypeError('cell.purge must be Purge')
        if not isinstance(self.lock, Lock):
            raise TypeError('cell.lock must be Lock')

@dataclass(frozen=True)
class State:
    cells: Tuple[Cell, ...]
    self: Tuple[str, str] = field(default_factory=Clean.self)
    monument: Tuple[str, ...] = field(default_factory=Clean.monument)

    def __post_init__(self) -> None:
        if len(self.cells) != SeatCount:
            raise ValueError(f'state must contain exactly {SeatCount} seats')
        for cell in self.cells:
            VerifyCell(cell)
        VerifySelf(self.self)
        VerifyMonument(self.monument)

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
class Salt:
    key: str
    salt: int

    def __post_init__(self) -> None:
        VerifyKey(self.key)
        VerifyNonNegative(self.salt, fieldname='salt.salt')

@dataclass(frozen=True)
class SaltGlyph:
    key: str
    saltbody: Tuple[Salt, ...]
    lockbody: Lock
    textbody: Text
    salthash: str
    lockhash: str
    texthash: str
    sign: str = NullSignHex
    locksign: str = NullSignHex

    def __post_init__(self) -> None:
        VerifyKey(self.key)
        if not isinstance(self.saltbody, tuple):
            raise TypeError('saltglyph.saltbody must be tuple')
        for salt in self.saltbody:
            if not isinstance(salt, Salt):
                raise TypeError('saltglyph.saltbody must contain Salt objects')
        if not isinstance(self.lockbody, Lock):
            raise TypeError('saltglyph.lockbody must be Lock')
        if not isinstance(self.textbody, Text):
            raise TypeError('saltglyph.textbody must be Text')
        VerifyHash(self.salthash, fieldname='saltglyph.salthash')
        VerifyHash(self.lockhash, fieldname='saltglyph.lockhash')
        VerifyHash(self.texthash, fieldname='saltglyph.texthash')
        VerifySignHex(self.sign, fieldname='saltglyph.sign')
        VerifySignHex(self.locksign, fieldname='saltglyph.locksign')

@dataclass(frozen=True)
class Chain:
    linked: bool
    relation: str = 'reject'
    equivocate: bool = False
    winner: Optional[Cell] = None
    loser: Optional[Cell] = None
    open: bool = False
    reason: str = ''

def VerifyState(state: State, *, expectedkeys: Optional[Iterable[str]]=None, expectedsalt: int=MillionInvariant) -> State:
    if not isinstance(state, State):
        raise TypeError('expected State')
    if len(state.cells) != SeatCount:
        raise ValueError(f'state must contain exactly {SeatCount} seats')
    VerifySelf(state.self)
    VerifyMonument(state.monument)
    keys = []
    for cell in state.cells:
        VerifyCell(cell)
        keys.append(cell.key)
    VerifyNonNegative(expectedsalt, fieldname='expectedsalt')
    if state.SaltTotal != int(expectedsalt):
        raise ValueError(f'million invariant violated: total={state.SaltTotal} expected={expectedsalt}')
    if len(set(keys)) != SeatCount:
        raise ValueError('key invariant violated: duplicate keys in state')
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
    if not isinstance(cell.purge, Purge):
        raise TypeError('cell.purge must be Purge')
    if not isinstance(cell.lock, Lock):
        raise TypeError('cell.lock must be Lock')
    VerifyKey(cell.key)
    VerifyNonNegative(cell.salt, fieldname='cell.salt')
    VerifySignHex(cell.sign, fieldname='cell.sign')
    VerifyHash(cell.lock.parent, fieldname='cell.lock.parent')
    VerifyHash(cell.lock.child, fieldname='cell.lock.child')
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

def VerifyMonument(value: Tuple[str, ...]) -> Tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError('state.monument must be a tuple')
    for item in value:
        if not isinstance(item, str):
            raise TypeError('state.monument entries must be str')
    return value

def VerifySalt(glyph: SaltGlyph) -> SaltGlyph:
    if not isinstance(glyph, SaltGlyph):
        raise TypeError('expected SaltGlyph')
    SaltGlyphShape(glyph)
    VerifySaltBody(glyph.key, glyph.saltbody, glyph.salthash)
    VerifyTextBody(glyph.textbody, glyph.texthash)
    VerifyLockBody(glyph.key, glyph.lockbody, glyph.lockhash, glyph.locksign)
    VerifyReceipt(glyph.key, glyph.salthash, glyph.lockhash, glyph.texthash, glyph.locksign, glyph.sign)
    return glyph

def VerifyDream(state: State, *, expectedkeys: Optional[Iterable[str]]=None) -> State:
    return VerifyState(Scrub(state), expectedkeys=expectedkeys)

def VerifyLock(key: str, lock: Lock, sign: str, *, allownull: bool=False) -> bool:
    VerifyKey(key)
    if not isinstance(lock, Lock):
        raise TypeError('expected Lock')
    VerifySignHex(sign, fieldname='sign')
    digest = LockHash(lock)
    if sign == NullSignHex:
        return bool(allownull)
    if not VerifySign(key, digest, sign, allownull=False):
        raise ValueError('lock sign failed verification')
    return True

def VerifyReceipt(key: str, salthash: str, lockhash: str, texthash: str, locksign: str, sign: str) -> bool:
    VerifyKey(key)
    VerifyHash(salthash, fieldname='salthash')
    VerifyHash(lockhash, fieldname='lockhash')
    VerifyHash(texthash, fieldname='texthash')
    VerifySignHex(locksign, fieldname='locksign')
    VerifySignHex(sign, fieldname='sign')
    digest = ReceiptHash(key, salthash, lockhash, texthash, locksign)
    if not VerifySign(key, digest, sign, allownull=False):
        raise ValueError('receipt sign failed verification')
    return True

def VerifyLockBody(key: str, lockbody: Lock, lockhashvalue: str, locksign: str) -> bool:
    VerifyKey(key)
    if not isinstance(lockbody, Lock):
        raise TypeError('expected Lock')
    VerifyHash(lockhashvalue, fieldname='lockhash')
    if LockHash(lockbody) != lockhashvalue:
        raise ValueError('lockhash mismatch')
    if not VerifySign(key, lockhashvalue, locksign, allownull=False):
        raise ValueError('locksign failed verification')
    return True

def VerifyTextBody(textbody: Text, texthashvalue: str) -> bool:
    if not isinstance(textbody, Text):
        raise TypeError('expected Text')
    VerifyHash(texthashvalue, fieldname='texthash')
    if TextHash(textbody) != texthashvalue:
        raise ValueError('texthash mismatch')
    return True

def VerifySaltBody(key: str, saltbody: Tuple[Salt, ...], salthashvalue: str) -> bool:
    VerifyKey(key)
    if not isinstance(saltbody, tuple):
        raise TypeError('saltbody must be tuple')
    VerifyHash(salthashvalue, fieldname='salthash')
    if SaltHash(key, saltbody) != salthashvalue:
        raise ValueError('salthash mismatch')
    return True

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

def LockBody(lockbody: Lock) -> bytes:
    if not isinstance(lockbody, Lock):
        raise TypeError('expected Lock')
    VerifyHash(lockbody.parent, fieldname='lockbody.parent')
    VerifyHash(lockbody.child, fieldname='lockbody.child')
    return bytes.fromhex(lockbody.parent) + bytes.fromhex(lockbody.child)

def LockHash(lockbody: Lock) -> str:
    return hashlib.sha256(LockBody(lockbody)).hexdigest()

def TextBody(textbody: Text) -> bytes:
    if not isinstance(textbody, Text):
        raise TypeError('expected Text')
    return str(textbody.text).encode('utf-8')

def TextHash(textbody: Text) -> str:
    return hashlib.sha256(TextBody(textbody)).hexdigest()

def SaltBody(key: str, saltbody: Tuple[Salt, ...]) -> bytes:
    VerifyKey(key)
    if not isinstance(saltbody, tuple):
        raise TypeError('saltbody must be tuple')
    parts = [key]
    for salt in saltbody:
        if not isinstance(salt, Salt):
            raise TypeError('saltbody must contain Salt objects')
        parts.extend([salt.key, str(salt.salt)])
    return '|'.join(parts).encode('utf-8')

def SaltHash(key: str, saltbody: Tuple[Salt, ...]) -> str:
    return hashlib.sha256(SaltBody(key, saltbody)).hexdigest()

def SaltGlyphShape(glyph: SaltGlyph) -> SaltGlyph:
    if not isinstance(glyph, SaltGlyph):
        raise TypeError('expected SaltGlyph')
    VerifyKey(glyph.key)
    if len(glyph.saltbody) not in SaltGlyphSpendCounts:
        raise ValueError('saltglyph saltbody count must be 1, 5, 6, or 23')
    for salt in glyph.saltbody:
        if not isinstance(salt, Salt):
            raise TypeError('saltglyph.saltbody must contain Salt objects')
        VerifyKey(salt.key)
        VerifyNonNegative(salt.salt, fieldname='salt.salt')
    if not isinstance(glyph.lockbody, Lock):
        raise TypeError('saltglyph.lockbody must be Lock')
    if not isinstance(glyph.textbody, Text):
        raise TypeError('saltglyph.textbody must be Text')
    VerifyHash(glyph.salthash, fieldname='saltglyph.salthash')
    VerifyHash(glyph.lockhash, fieldname='saltglyph.lockhash')
    VerifyHash(glyph.texthash, fieldname='saltglyph.texthash')
    VerifySignHex(glyph.locksign, fieldname='saltglyph.locksign')
    VerifySignHex(glyph.sign, fieldname='saltglyph.sign')
    return glyph

def ReceiptBody(key: str, salthash: str, lockhash: str, texthash: str, locksign: str) -> bytes:
    VerifyKey(key)
    VerifyHash(salthash, fieldname='salthash')
    VerifyHash(lockhash, fieldname='lockhash')
    VerifyHash(texthash, fieldname='texthash')
    VerifySignHex(locksign, fieldname='locksign')
    return '|'.join([key, salthash, lockhash, texthash, locksign]).encode('utf-8')

def ReceiptHash(key: str, salthash: str, lockhash: str, texthash: str, locksign: str) -> str:
    return hashlib.sha256(ReceiptBody(key, salthash, lockhash, texthash, locksign)).hexdigest()

def SaltGlyphBody(glyph: SaltGlyph) -> bytes:
    SaltGlyphShape(glyph)
    return ReceiptBody(glyph.key, glyph.salthash, glyph.lockhash, glyph.texthash, glyph.locksign)

def SaltGlyphHash(glyph: SaltGlyph) -> str:
    return hashlib.sha256(SaltGlyphBody(glyph)).hexdigest()

def CellBody(cell: Cell, *, includepurge: bool=False) -> bytes:
    VerifyCell(cell)
    parts = [cell.soul, cell.key, str(cell.salt), cell.lock.parent, cell.lock.child, cell.sign]
    if includepurge:
        parts.extend([str(cell.purge.chainbit), str(cell.purge.lockbit)])
    return '|'.join(parts).encode('utf-8')

def CellHash(cell: Cell, *, includepurge: bool=False) -> str:
    return hashlib.sha256(CellBody(cell, includepurge=includepurge)).hexdigest()

def ReceiptTotal(glyph: SaltGlyph) -> int:
    SaltGlyphShape(glyph)
    return sum((int(salt.salt) for salt in glyph.saltbody))

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

def FileTotals(state: State) -> Tuple[int, int, int, int]:
    frozen = Stasis(state)
    return tuple((sum((int(cell.salt) for cell in FileCells(frozen, filenumber))) for filenumber in range(1, FileCount + 1)))

def Link(current: Cell, candidate: Cell) -> Chain:
    VerifyCell(current)
    VerifyCell(candidate)
    samekey = current.key == candidate.key
    sameparent = current.lock.parent == candidate.lock.parent
    samechild = current.lock.child == candidate.lock.child
    if samekey and sameparent and samechild and (current.soul == candidate.soul) and (current.salt == candidate.salt) and (current.sign == candidate.sign):
        return Chain(linked=True, relation='Link', open=False)
    # ================= LINCHPIN ================= #
    if candidate.lock.parent == current.lock.child:
    # ============================================ #
        if candidate.key != current.key:
            return Chain(linked=False, relation='reject', open=True, reason='key changed')
        if candidate.salt < current.salt and candidate.sign == NullSignHex:
            return Chain(linked=False, relation='reject', open=True, reason='debit without sign')
        return Chain(linked=True, relation='Link', open=False)
    # ================= LINCHPIN ================= #
    if samekey and sameparent and (not samechild):
    # ============================================ #
        winner, loser = Equivocation(current, candidate)
        return Chain(linked=True, relation='Link', equivocate=True, winner=winner, loser=loser, open=False, reason='Equivocation')
    return Chain(linked=False, relation='reject', open=True, reason='no Link')

def Equivocation(a: Cell, b: Cell) -> Tuple[Cell, Cell]:
    VerifyCell(a)
    VerifyCell(b)
    if a.key != b.key:
        raise ValueError('Equivocation requires same key')
    if a.lock.parent != b.lock.parent:
        raise ValueError('Equivocation requires same parent')
    if a.lock.child == b.lock.child:
        raise ValueError('Equivocation requires competing children')
    ah = LockHash(a.lock)
    bh = LockHash(b.lock)
    return (a, b) if ah <= bh else (b, a)

def MutateReceipt(state: State, glyph: SaltGlyph) -> Tuple[State, Tuple[Chain, ...]]:
    VerifyState(state)
    VerifySalt(glyph)
    signer = FindCell(state, glyph.key)
    if signer is None:
        raise ValueError('glyph signer key not found in state')
    debit = ReceiptTotal(glyph)
    signercandidate = replace(signer, salt=signer.salt - debit, lock=glyph.lockbody, sign=glyph.locksign)
    signerchain = Link(signer, signercandidate)
    if not signerchain.linked:
        return (ReplaceCell(state, OpenCell(signer)), (signerchain,))
    credits: dict[str, int] = {}
    for spend in glyph.saltbody:
        credits[spend.key] = credits.get(spend.key, 0) + int(spend.salt)
    replacements: dict[str, Cell] = {signer.key: signerchain.winner or signercandidate}
    chains = [signerchain]
    for key, amount in credits.items():
        target = FindCell(state, key)
        if target is None:
            raise ValueError('spend key not found in state')
        base = replacements.get(key, target)
        targetcandidate = replace(base, salt=base.salt + int(amount))
        replacements[key] = targetcandidate
        chains.append(Chain(linked=True, relation='Link', open=False, reason='credit'))
    nextcells = tuple((replacements.get(cell.key, cell) for cell in state.cells))
    nextstate = State(cells=nextcells, self=state.self, monument=state.monument)
    VerifyState(nextstate, expectedkeys=FindKeys(state))
    return (nextstate, tuple(chains))

def Assimilate(local: State, incoming: State) -> Tuple[State, Tuple[Chain, ...]]:
    VerifyState(local)
    keys = FindKeys(local)
    VerifyDream(incoming, expectedkeys=keys)
    scrubbed = Scrub(incoming)
    localmap = {cell.key: cell for cell in local.cells}
    cells = []
    chains = []
    for candidate in scrubbed.cells:
        current = localmap.get(candidate.key)
        if current is None:
            raise ValueError('incoming state key missing from local state')
        if current.purge.lockbit == 0:
            cells.append(CloseCell(candidate))
            chains.append(Chain(linked=True, relation='Link', open=False, reason='open'))
            continue
        outcome = Link(current, candidate)
        chains.append(outcome)
        if outcome.linked:
            chosen = outcome.winner or candidate
            cells.append(CloseCell(chosen))
        else:
            cells.append(OpenCell(current))
    nextstate = State(cells=tuple(cells), self=local.self, monument=incoming.monument)
    VerifyState(nextstate, expectedkeys=keys)
    return (nextstate, tuple(chains))

def MutateState(local: State, incoming: State) -> Tuple[State, Tuple[Chain, ...]]:
    return Assimilate(local, incoming)

def MutatePurge(cell: Cell, *, chainbit: Optional[int]=None, lockbit: Optional[int]=None) -> Cell:
    VerifyCell(cell)
    cb = cell.purge.chainbit if chainbit is None else int(chainbit)
    lb = cell.purge.lockbit if lockbit is None else int(lockbit)
    return replace(cell, purge=Purge(chainbit=cb, lockbit=lb))

def OpenCell(cell: Cell) -> Cell:
    return MutatePurge(cell, chainbit=0)

def CloseCell(cell: Cell) -> Cell:
    return MutatePurge(cell, chainbit=1)

def Scrub(state: State) -> State:
    VerifyState(state)
    return State(cells=tuple((replace(cell, purge=Clean.purge()) for cell in state.cells)), self=Clean.self(), monument=tuple(state.monument))

def ScrubPurge(state: State) -> State:
    return Scrub(state)

def Stasis(state: State) -> State:
    VerifyState(state)
    ordered = []
    for filenumber in range(1, FileCount + 1):
        ordered.extend(SortFile(FileCells(state, filenumber)))
    return State(cells=tuple(ordered), self=state.self, monument=state.monument)

def FindKeys(state: State) -> Tuple[str, ...]:
    VerifyState(state)
    return tuple((cell.key for cell in state.cells))

def FindCell(state: State, key: str) -> Optional[Cell]:
    VerifyState(state)
    VerifyKey(key)
    for cell in state.cells:
        if cell.key == key:
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
    return State(cells=tuple(cells), self=state.self, monument=state.monument)
