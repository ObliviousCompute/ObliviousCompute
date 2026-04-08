from __future__ import annotations
from dataclasses import dataclass, field, replace
import hashlib
from typing import Iterable, Optional, Tuple
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
HASH_BYTES = 32
HASH_HEX_LEN = HASH_BYTES * 2
KEY_BYTES = 32
KEY_HEX_LEN = KEY_BYTES * 2
SIGN_BYTES = 64
SIGN_HEX_LEN = SIGN_BYTES * 2
ZERO_HASH_HEX = '00' * HASH_BYTES
NULL_SIGN_HEX = '00' * SIGN_BYTES
FILE_COUNT = 4
SEATS_PER_FILE = 6
SEAT_COUNT = FILE_COUNT * SEATS_PER_FILE
MILLION_INVARIANT = 1000000
SALT_GLYPH_SPEND_COUNTS = {1, 5, 6, 23}
TEXT_MAX_LEN = 68

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
    parent: str = ZERO_HASH_HEX
    child: str = ZERO_HASH_HEX

    def __post_init__(self) -> None:
        VerifyHash(self.parent, field_name='lock.parent')
        VerifyHash(self.child, field_name='lock.child')

class Clean:

    @staticmethod
    def purge() -> Purge:
        return Purge(chainbit=0, lockbit=0)

    @staticmethod
    def lock() -> Lock:
        return Lock(parent=ZERO_HASH_HEX, child=ZERO_HASH_HEX)

    @staticmethod
    def sign() -> str:
        return NULL_SIGN_HEX

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
        VerifyNonNegative(self.salt, field_name='cell.salt')
        VerifySignHex(self.sign, field_name='cell.sign')
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
        if len(self.cells) != SEAT_COUNT:
            raise ValueError(f'state must contain exactly {SEAT_COUNT} seats')
        for cell in self.cells:
            VerifyCell(cell)
        VerifySelf(self.self)
        VerifyMonument(self.monument)

    @property
    def saltTotal(self) -> int:
        return sum((int(cell.salt) for cell in self.cells))

@dataclass(frozen=True)
class Text:
    text: str = ''

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError('text.text must be str')
        if len(self.text) > TEXT_MAX_LEN:
            raise ValueError(f'text.text must be at most {TEXT_MAX_LEN} chars')

@dataclass(frozen=True)
class Salt:
    key: str
    salt: int

    def __post_init__(self) -> None:
        VerifyKey(self.key)
        VerifyNonNegative(self.salt, field_name='salt.salt')

@dataclass(frozen=True)
class SaltGlyph:
    key: str
    saltbody: Tuple[Salt, ...]
    lockbody: Lock
    textbody: Text
    salthash: str
    lockhash: str
    texthash: str
    sign: str = NULL_SIGN_HEX
    locksign: str = NULL_SIGN_HEX

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
        VerifyHash(self.salthash, field_name='saltglyph.salthash')
        VerifyHash(self.lockhash, field_name='saltglyph.lockhash')
        VerifyHash(self.texthash, field_name='saltglyph.texthash')
        VerifySignHex(self.sign, field_name='saltglyph.sign')
        VerifySignHex(self.locksign, field_name='saltglyph.locksign')

@dataclass(frozen=True)
class Chain:
    linked: bool
    relation: str = 'reject'
    equivocate: bool = False
    winner: Optional[Cell] = None
    loser: Optional[Cell] = None
    open: bool = False
    reason: str = ''

def VerifyState(state: State, *, expectedKeys: Optional[Iterable[str]]=None, expectedSalt: int=MILLION_INVARIANT) -> State:
    if not isinstance(state, State):
        raise TypeError('expected State')
    if len(state.cells) != SEAT_COUNT:
        raise ValueError(f'state must contain exactly {SEAT_COUNT} seats')
    VerifySelf(state.self)
    VerifyMonument(state.monument)
    keys = []
    for cell in state.cells:
        VerifyCell(cell)
        keys.append(cell.key)
    VerifyNonNegative(expectedSalt, field_name='expectedSalt')
    if state.saltTotal != int(expectedSalt):
        raise ValueError(f'million invariant violated: total={state.saltTotal} expected={expectedSalt}')
    if len(set(keys)) != SEAT_COUNT:
        raise ValueError('key invariant violated: duplicate keys in state')
    if expectedKeys is not None:
        known = tuple(expectedKeys)
        if len(known) != SEAT_COUNT:
            raise ValueError(f'expectedKeys must contain exactly {SEAT_COUNT} keys')
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
    VerifyNonNegative(cell.salt, field_name='cell.salt')
    VerifySignHex(cell.sign, field_name='cell.sign')
    VerifyHash(cell.lock.parent, field_name='cell.lock.parent')
    VerifyHash(cell.lock.child, field_name='cell.lock.child')
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

def VerifyDream(state: State, *, expectedKeys: Optional[Iterable[str]]=None) -> State:
    return VerifyState(Scrub(state), expectedKeys=expectedKeys)

def VerifyLock(key: str, lock: Lock, sign: str, *, allow_null: bool=False) -> bool:
    VerifyKey(key)
    if not isinstance(lock, Lock):
        raise TypeError('expected Lock')
    VerifySignHex(sign, field_name='sign')
    digest = LockHash(lock)
    if sign == NULL_SIGN_HEX:
        return bool(allow_null)
    if not VerifySign(key, digest, sign, allow_null=False):
        raise ValueError('lock sign failed verification')
    return True

def VerifyReceipt(key: str, salthash: str, lockhash: str, texthash: str, locksign: str, sign: str) -> bool:
    VerifyKey(key)
    VerifyHash(salthash, field_name='salthash')
    VerifyHash(lockhash, field_name='lockhash')
    VerifyHash(texthash, field_name='texthash')
    VerifySignHex(locksign, field_name='locksign')
    VerifySignHex(sign, field_name='sign')
    digest = ReceiptHash(key, salthash, lockhash, texthash, locksign)
    if not VerifySign(key, digest, sign, allow_null=False):
        raise ValueError('receipt sign failed verification')
    return True

def VerifyLockBody(key: str, lockbody: Lock, lockhashvalue: str, locksign: str) -> bool:
    VerifyKey(key)
    if not isinstance(lockbody, Lock):
        raise TypeError('expected Lock')
    VerifyHash(lockhashvalue, field_name='lockhash')
    if LockHash(lockbody) != lockhashvalue:
        raise ValueError('lockhash mismatch')
    if not VerifySign(key, lockhashvalue, locksign, allow_null=False):
        raise ValueError('locksign failed verification')
    return True

def VerifyTextBody(textbody: Text, texthashvalue: str) -> bool:
    if not isinstance(textbody, Text):
        raise TypeError('expected Text')
    VerifyHash(texthashvalue, field_name='texthash')
    if TextHash(textbody) != texthashvalue:
        raise ValueError('texthash mismatch')
    return True

def VerifySaltBody(key: str, saltbody: Tuple[Salt, ...], salthashvalue: str) -> bool:
    VerifyKey(key)
    if not isinstance(saltbody, tuple):
        raise TypeError('saltbody must be tuple')
    VerifyHash(salthashvalue, field_name='salthash')
    if SaltHash(key, saltbody) != salthashvalue:
        raise ValueError('salthash mismatch')
    return True

def VerifyHash(value: str, *, field_name: str='hash') -> bool:
    if not isinstance(value, str):
        raise TypeError(f'{field_name} must be a hex string')
    if len(value) != HASH_HEX_LEN:
        raise ValueError(f'{field_name} must be exactly {HASH_BYTES} bytes / {HASH_HEX_LEN} hex chars')
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f'{field_name} must be valid hex') from exc
    return True

def VerifyKey(value: str) -> bool:
    if not isinstance(value, str):
        raise TypeError('key must be a hex string')
    if len(value) != KEY_HEX_LEN:
        raise ValueError(f'key must be exactly {KEY_BYTES} bytes / {KEY_HEX_LEN} hex chars')
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError('key must be valid hex') from exc
    return True

def VerifySignHex(value: str, *, field_name: str='sign') -> bool:
    if not isinstance(value, str):
        raise TypeError(f'{field_name} must be a hex string')
    if value == NULL_SIGN_HEX:
        return True
    if len(value) != SIGN_HEX_LEN:
        raise ValueError(f'{field_name} must be exactly {SIGN_BYTES} bytes / {SIGN_HEX_LEN} hex chars')
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f'{field_name} must be valid hex') from exc
    return True

def VerifyNonNegative(value: int, *, field_name: str='value') -> bool:
    if not isinstance(value, int):
        raise TypeError(f'{field_name} must be an integer')
    if value < 0:
        raise ValueError(f'{field_name} must be non-negative')
    return True

def VerifySign(key: str, digest: str, sign: str, *, allow_null: bool=False) -> bool:
    VerifyKey(key)
    VerifyHash(digest, field_name='digest')
    VerifySignHex(sign, field_name='sign')
    if sign == NULL_SIGN_HEX:
        return bool(allow_null)
    try:
        pubkey = Ed25519PublicKey.from_public_bytes(bytes.fromhex(key))
        pubkey.verify(bytes.fromhex(sign), bytes.fromhex(digest))
        return True
    except (InvalidSignature, ValueError):
        return False

def LockBody(lockbody: Lock) -> bytes:
    if not isinstance(lockbody, Lock):
        raise TypeError('expected Lock')
    VerifyHash(lockbody.parent, field_name='lockbody.parent')
    VerifyHash(lockbody.child, field_name='lockbody.child')
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
    if len(glyph.saltbody) not in SALT_GLYPH_SPEND_COUNTS:
        raise ValueError('saltglyph saltbody count must be 1, 5, 6, or 23')
    for salt in glyph.saltbody:
        if not isinstance(salt, Salt):
            raise TypeError('saltglyph.saltbody must contain Salt objects')
        VerifyKey(salt.key)
        VerifyNonNegative(salt.salt, field_name='salt.salt')
    if not isinstance(glyph.lockbody, Lock):
        raise TypeError('saltglyph.lockbody must be Lock')
    if not isinstance(glyph.textbody, Text):
        raise TypeError('saltglyph.textbody must be Text')
    VerifyHash(glyph.salthash, field_name='saltglyph.salthash')
    VerifyHash(glyph.lockhash, field_name='saltglyph.lockhash')
    VerifyHash(glyph.texthash, field_name='saltglyph.texthash')
    VerifySignHex(glyph.locksign, field_name='saltglyph.locksign')
    VerifySignHex(glyph.sign, field_name='saltglyph.sign')
    return glyph

def ReceiptBody(key: str, salthash: str, lockhash: str, texthash: str, locksign: str) -> bytes:
    VerifyKey(key)
    VerifyHash(salthash, field_name='salthash')
    VerifyHash(lockhash, field_name='lockhash')
    VerifyHash(texthash, field_name='texthash')
    VerifySignHex(locksign, field_name='locksign')
    return '|'.join([key, salthash, lockhash, texthash, locksign]).encode('utf-8')

def ReceiptHash(key: str, salthash: str, lockhash: str, texthash: str, locksign: str) -> str:
    return hashlib.sha256(ReceiptBody(key, salthash, lockhash, texthash, locksign)).hexdigest()

def SaltGlyphBody(glyph: SaltGlyph) -> bytes:
    SaltGlyphShape(glyph)
    return ReceiptBody(glyph.key, glyph.salthash, glyph.lockhash, glyph.texthash, glyph.locksign)

def SaltGlyphHash(glyph: SaltGlyph) -> str:
    return hashlib.sha256(SaltGlyphBody(glyph)).hexdigest()

def CellBody(cell: Cell, *, includePurge: bool=False) -> bytes:
    VerifyCell(cell)
    parts = [cell.soul, cell.key, str(cell.salt), cell.lock.parent, cell.lock.child, cell.sign]
    if includePurge:
        parts.extend([str(cell.purge.chainbit), str(cell.purge.lockbit)])
    return '|'.join(parts).encode('utf-8')

def CellHash(cell: Cell, *, includePurge: bool=False) -> str:
    return hashlib.sha256(CellBody(cell, includePurge=includePurge)).hexdigest()

def ReceiptTotal(glyph: SaltGlyph) -> int:
    SaltGlyphShape(glyph)
    return sum((int(salt.salt) for salt in glyph.saltbody))

def SeatRange(fileNumber: int) -> range:
    VerifyNonNegative(fileNumber, field_name='fileNumber')
    if fileNumber < 1 or fileNumber > FILE_COUNT:
        raise ValueError(f'fileNumber must be 1..{FILE_COUNT}')
    firstSeat = (fileNumber - 1) * SEATS_PER_FILE + 1
    lastSeat = firstSeat + SEATS_PER_FILE - 1
    return range(firstSeat, lastSeat + 1)

def FileCells(state: State, fileNumber: int) -> Tuple[Cell, ...]:
    VerifyState(state)
    seats = SeatRange(fileNumber)
    return tuple((state.cells[seat - 1] for seat in seats))

def SortFile(fileCellsValue: Iterable[Cell]) -> Tuple[Cell, ...]:
    ordered = tuple(sorted(tuple(fileCellsValue), key=lambda cell: (-int(cell.salt), str(cell.key))))
    if len(ordered) != SEATS_PER_FILE:
        raise ValueError(f'file must contain exactly {SEATS_PER_FILE} seats')
    return ordered

def FileTotals(state: State) -> Tuple[int, int, int, int]:
    frozen = Stasis(state)
    return tuple((sum((int(cell.salt) for cell in FileCells(frozen, fileNumber))) for fileNumber in range(1, FILE_COUNT + 1)))

def Link(current: Cell, candidate: Cell) -> Chain:
    VerifyCell(current)
    VerifyCell(candidate)
    sameKey = current.key == candidate.key
    sameParent = current.lock.parent == candidate.lock.parent
    sameChild = current.lock.child == candidate.lock.child
    if sameKey and sameParent and sameChild and (current.soul == candidate.soul) and (current.salt == candidate.salt) and (current.sign == candidate.sign):
        return Chain(linked=True, relation='Link', open=False)
    # ================= LINCHPIN ================= #
    if candidate.lock.parent == current.lock.child:
    # ============================================ #
        if candidate.key != current.key:
            return Chain(linked=False, relation='reject', open=True, reason='key changed')
        if candidate.salt < current.salt and candidate.sign == NULL_SIGN_HEX:
            return Chain(linked=False, relation='reject', open=True, reason='debit without sign')
        return Chain(linked=True, relation='Link', open=False)
    # ================= LINCHPIN ================= #
    if sameKey and sameParent and (not sameChild):
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
    signerCandidate = replace(signer, salt=signer.salt - debit, lock=glyph.lockbody, sign=glyph.locksign)
    signerChain = Link(signer, signerCandidate)
    if not signerChain.linked:
        return (ReplaceCell(state, OpenCell(signer)), (signerChain,))
    credits: dict[str, int] = {}
    for spend in glyph.saltbody:
        credits[spend.key] = credits.get(spend.key, 0) + int(spend.salt)
    replacements: dict[str, Cell] = {signer.key: signerChain.winner or signerCandidate}
    chains = [signerChain]
    for key, amount in credits.items():
        target = FindCell(state, key)
        if target is None:
            raise ValueError('spend key not found in state')
        base = replacements.get(key, target)
        targetCandidate = replace(base, salt=base.salt + int(amount))
        replacements[key] = targetCandidate
        chains.append(Chain(linked=True, relation='Link', open=False, reason='credit'))
    nextCells = tuple((replacements.get(cell.key, cell) for cell in state.cells))
    nextState = State(cells=nextCells, self=state.self, monument=state.monument)
    VerifyState(nextState, expectedKeys=FindKeys(state))
    return (nextState, tuple(chains))

def Assimilate(local: State, incoming: State) -> Tuple[State, Tuple[Chain, ...]]:
    VerifyState(local)
    keys = FindKeys(local)
    VerifyDream(incoming, expectedKeys=keys)
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
    nextState = State(cells=tuple(cells), self=local.self, monument=incoming.monument)
    VerifyState(nextState, expectedKeys=keys)
    return (nextState, tuple(chains))

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
    for fileNumber in range(1, FILE_COUNT + 1):
        ordered.extend(SortFile(FileCells(state, fileNumber)))
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
