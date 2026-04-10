from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Any, Optional, Sequence, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

import Dream
import Field

TransferPair = Tuple[str, int]

Purge = 'purge'
Whisper = 'whisper'
Rally = 'rally'
Wrath = 'wrath'
Defect = 'defect'
Salt = 'salt'

MainGate = '9000'
DefaultMode = ''
DefaultSkeleton = 'skeleton'
DefaultSecret = 'password'
DefaultSoul = 'SATOSHI'
DefaultGenesis = 1
TextMaxLen = 68

ExpectedSpendCounts = {
    Whisper: 1,
    Rally: 5,
    Wrath: 23,
    Defect: 6,
}


@dataclass(frozen=True)
class State:
    mode: str
    gate: str
    skeleton: str
    soul: str
    secret: str = ''
    key: str = ''
    genesis: int = 1


@dataclass(frozen=True)
class Glyph:
    kind: str
    key: str = ''
    pairs: Tuple[TransferPair, ...] = ()
    text: str = ''
    lock: Optional[Field.Lock] = None

    def __post_init__(self) -> None:
        kind = str(self.kind or '').strip().lower()
        key = str(self.key or '').strip()
        pairs = tuple((str(k or '').strip(), int(v or 0)) for k, v in tuple(self.pairs or ()))
        text = str(self.text or '')[:TextMaxLen]
        lock = self.lock
        if lock is not None and not isinstance(lock, Field.Lock):
            raise TypeError('glyph.lock must be Field.Lock or None')
        object.__setattr__(self, 'kind', kind)
        object.__setattr__(self, 'key', key)
        object.__setattr__(self, 'pairs', pairs)
        object.__setattr__(self, 'text', text)
        object.__setattr__(self, 'lock', lock)

    @property
    def TotalAmount(self) -> int:
        return sum(int(amount) for _, amount in self.pairs)

    @property
    def SpendCount(self) -> int:
        return len(self.pairs)

    @classmethod
    def Value(cls, value: Any) -> 'Glyph':
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            if 'kind' not in value:
                raise KeyError('glyph dict must contain kind')
            return cls(
                kind=str(value['kind'] or ''),
                key=str(value.get('key', '') or '').strip(),
                pairs=tuple(value.get('pairs', ()) or ()),
                text=str(value.get('text', '') or ''),
                lock=LockParse(value.get('lock', None)),
            )
        kind = getattr(value, 'kind')
        return cls(
            kind=str(kind or ''),
            key=str(getattr(value, 'key', '') or '').strip(),
            pairs=tuple(getattr(value, 'pairs', ()) or ()),
            text=str(getattr(value, 'text', '') or ''),
            lock=LockParse(getattr(value, 'lock', None)),
        )


def Sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def Sha256Hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def SafeInt(value: Any, fallback: int = 1) -> int:
    try:
        text = str(value or '').strip()
        return int(text) if text else int(fallback)
    except Exception:
        return int(fallback)


def StateParse(
    value: Any = None,
    *,
    mode: str = '',
    gate: str = '',
    skeleton: str = '',
    soul: str = '',
    secret: str = '',
    key: str = '',
    genesis: Any = None,
) -> State:
    if value is None:
        return State(
            mode=str(mode or DefaultMode),
            gate=str(gate or MainGate),
            skeleton=str(skeleton or DefaultSkeleton),
            soul=str(soul or DefaultSoul),
            secret=str(secret or DefaultSecret),
            key=str(key or ''),
            genesis=max(1, SafeInt(genesis or DefaultGenesis, DefaultGenesis)),
        )
    if isinstance(value, State):
        return value
    if isinstance(value, dict):
        return State(
            mode=str(value['mode'] if 'mode' in value else mode or DefaultMode),
            gate=str(value['gate'] if 'gate' in value else gate or MainGate),
            skeleton=str(value['skeleton'] if 'skeleton' in value else skeleton or DefaultSkeleton),
            soul=str(value['soul'] if 'soul' in value else soul or DefaultSoul),
            secret=str(value['secret'] if 'secret' in value else secret or DefaultSecret),
            key=str(value.get('key', key or '') or ''),
            genesis=max(1, SafeInt(value['genesis'] if 'genesis' in value else genesis or DefaultGenesis, DefaultGenesis)),
        )
    return State(
        mode=str(getattr(value, 'mode')),
        gate=str(getattr(value, 'gate')),
        skeleton=str(getattr(value, 'skeleton')),
        soul=str(getattr(value, 'soul')),
        secret=str(getattr(value, 'secret', secret or DefaultSecret) or DefaultSecret),
        key=str(getattr(value, 'key', key or '') or ''),
        genesis=max(1, SafeInt(getattr(value, 'genesis', genesis or DefaultGenesis), DefaultGenesis)),
    )


def LockParse(value: Any) -> Optional[Field.Lock]:
    if value is None:
        return None
    if isinstance(value, Field.Lock):
        return value
    if isinstance(value, dict):
        if 'parent' not in value or 'child' not in value:
            raise KeyError('lock dict must contain parent and child')
        return Field.Lock(
            parent=str(value['parent'] or Field.ZeroHashHex),
            child=str(value['child'] or Field.ZeroHashHex),
        )
    parent = getattr(value, 'parent')
    child = getattr(value, 'child')
    return Field.Lock(
        parent=str(parent or Field.ZeroHashHex),
        child=str(child or Field.ZeroHashHex),
    )


def StateKey(soul: str, secret: str) -> Ed25519PrivateKey:
    seed = Sha256(f'Byzantium::State::V3::{soul}::{secret}'.encode('utf-8'))
    return Ed25519PrivateKey.from_private_bytes(seed)


class Vault:
    def __init__(
        self,
        dream: object | None = None,
        *,
        state: Any = None,
        mode: str = '',
        gate: str = '',
        skeleton: str = '',
        soul: str = '',
        secret: str = '',
        genesis: Any = None,
        citadel: object | None = None,
    ) -> None:
        self.secret = ''
        self.privatekey: Optional[Ed25519PrivateKey] = None
        self.publickey = None
        self.publickeyhex = ''
        self.state = StateParse(
            state,
            mode=mode,
            gate=gate,
            skeleton=skeleton,
            soul=soul,
            secret=secret,
            genesis=genesis,
        )
        self.mode = self.state.mode
        self.gate = self.state.gate
        self.skeleton = self.state.skeleton
        self.soul = self.state.soul
        self.key = self.state.key
        self.genesis = self.state.genesis
        self.dream = self.WakeDream(dream, citadel=citadel)
        Dream.dream = self.dream
        self.Genesis(self.state)

    def Wake(self):
        child = self.dream
        if child and hasattr(child, 'Wake'):
            return child.Wake()
        return child

    def WakeDream(self, dream: object | None, *, citadel: object | None = None) -> object:
        if dream is None:
            return Dream.Dream(citadel=citadel)
        if isinstance(dream, type):
            try:
                return dream(citadel=citadel)
            except TypeError:
                return dream()
        return dream

    def SyncState(self, state: State) -> State:
        self.state = state
        self.mode = state.mode
        self.gate = state.gate
        self.skeleton = state.skeleton
        self.soul = state.soul
        self.key = state.key
        self.genesis = state.genesis
        return self.state

    def Genesis(self, state: Any = None) -> Any:
        source = state if state is not None else self.state
        rawstate = StateParse(source)
        self.secret = str(rawstate.secret or DefaultSecret)
        self.privatekey = StateKey(rawstate.soul, self.secret)
        self.publickey = self.privatekey.public_key()
        self.publickeyhex = self.publickey.public_bytes(
            encoding=Encoding.Raw,
            format=PublicFormat.Raw,
        ).hex()
        lawfulstate = replace(rawstate, secret='', key=self.publickeyhex)
        self.SyncState(lawfulstate)
        if hasattr(self.dream, 'Genesis'):
            return self.dream.Genesis(self.state)
        self.SendState()
        self.Wake()
        return self.state

    def SendState(self) -> Any:
        if not hasattr(self.dream, 'box'):
            raise AttributeError('Dream must expose box')
        self.dream.box.vault = self.state
        return self.dream.box.vault

    def Emit(self, payload: Any) -> Any:
        if not hasattr(self.dream, 'box'):
            raise AttributeError('Dream must expose box')
        self.dream.box.vault = payload
        return self.dream.box.vault

    def SignDigestHex(self, digesthex: str) -> str:
        Field.VerifyHash(digesthex, fieldname='digesthex')
        if self.privatekey is None:
            raise RuntimeError('private key not initialized')
        return self.privatekey.sign(bytes.fromhex(digesthex)).hex()

    def NormalizePairs(self, pairs: Sequence[TransferPair]) -> list[TransferPair]:
        out: list[TransferPair] = []
        for recipient, amount in list(pairs or []):
            recipientkey = str(recipient or '').strip()
            amountint = int(amount)
            Field.VerifyKey(recipientkey)
            Field.VerifyNonNegative(amountint, fieldname='amount')
            out.append((recipientkey, amountint))
        return out

    def AssertGlyphShape(self, glyph: Glyph, pairs: Sequence[TransferPair]) -> None:
        spendcount = len(tuple(pairs))
        if glyph.kind in ExpectedSpendCounts:
            expected = ExpectedSpendCounts[glyph.kind]
            if spendcount != expected:
                raise ValueError(f'{glyph.kind} requires exactly {expected} spend legs')
            if sum(int(amount) for _, amount in pairs) <= 0:
                raise ValueError('totalamount must be positive')

    def Text(self, glyph: Glyph) -> Field.Text:
        kind = str(glyph.kind or '').strip().lower()
        rawtext = str(glyph.text or '')
        if not kind:
            return Field.Text(text=rawtext[:TextMaxLen])
        suffix = f'|{kind}'
        if rawtext.lower().endswith(suffix):
            return Field.Text(text=rawtext[:TextMaxLen])
        return Field.Text(text=f'{rawtext}{suffix}'[:TextMaxLen])

    def HashText(self, textbody: Field.Text) -> str:
        return Field.TextHash(textbody)

    def Salt(self, pairs: Sequence[TransferPair]) -> Tuple[Field.Salt, ...]:
        return tuple(Field.Salt(key=recipientkey, salt=int(amount)) for recipientkey, amount in pairs)

    def HashSalt(self, saltbody: Tuple[Field.Salt, ...]) -> str:
        return Field.SaltHash(self.publickeyhex, saltbody)

    def BlankLock(self, lock: Field.Lock) -> bool:
        return (
            str(lock.parent or Field.ZeroHashHex) == Field.ZeroHashHex
            and str(lock.child or Field.ZeroHashHex) == Field.ZeroHashHex
        )

    def BaseLock(self, glyph: Glyph) -> Field.Lock:
        currentlock = glyph.lock
        if currentlock is None or self.BlankLock(currentlock):
            return Field.Lock(parent=Field.ZeroHashHex, child=Field.ZeroHashHex)
        return currentlock

    def ChildLockHex(self, kind: str, parent: str, texthash: str, pairs: Sequence[TransferPair]) -> str:
        parts = [str(kind or '').lower(), str(parent or Field.ZeroHashHex), str(texthash or Field.ZeroHashHex)]
        for recipientkey, amount in pairs:
            parts.extend([str(recipientkey), str(int(amount))])
        return Sha256Hex('|'.join(parts).encode('utf-8'))

    def HashLock(self, lockbody: Field.Lock) -> str:
        return Field.LockHash(lockbody)

    def NextLock(self, glyph: Glyph, texthash: str, pairs: Sequence[TransferPair]) -> Field.Lock:
        currentlock = self.BaseLock(glyph)
        parent = str(currentlock.child or Field.ZeroHashHex)
        child = self.ChildLockHex(glyph.kind, parent, texthash, pairs)
        return Field.Lock(parent=parent, child=child)

    def Lock(self, glyph: Glyph, texthash: str, pairs: Sequence[TransferPair]) -> Field.Lock:
        return self.NextLock(glyph, texthash, pairs)

    def SaltGlyph(self, glyph: Glyph) -> Field.SaltGlyph:
        pairs = self.NormalizePairs(glyph.pairs)
        self.AssertGlyphShape(glyph, pairs)
        textbody = self.Text(glyph)
        texthash = self.HashText(textbody)
        saltbody = self.Salt(pairs)
        salthash = self.HashSalt(saltbody)
        lockbody = self.Lock(glyph, texthash, pairs)
        lockhash = self.HashLock(lockbody)
        locksign = self.SignDigestHex(lockhash)
        proto = Field.SaltGlyph(
            key=self.publickeyhex,
            saltbody=saltbody,
            lockbody=lockbody,
            textbody=textbody,
            salthash=salthash,
            lockhash=lockhash,
            texthash=texthash,
            sign=Field.NullSignHex,
            locksign=locksign,
        )
        sign = self.SignDigestHex(Field.SaltGlyphHash(proto))
        body = replace(proto, sign=sign)
        Field.VerifySalt(body)
        return body

    def Purge(self, glyph: Glyph) -> dict[str, Any]:
        targetkey = str(glyph.key or '').strip()
        Field.VerifyKey(targetkey)
        return {'kind': Purge, 'key': targetkey}

    def Refine(self, glyph: Any) -> Any:
        baton = Glyph.Value(glyph)
        if baton.kind == Purge:
            return self.Purge(baton)
        if baton.kind in (Whisper, Rally, Wrath, Defect):
            return self.SaltGlyph(baton)
        raise ValueError(f'unsupported glyph kind: {baton.kind!r}')

    def Intent(self, value: Any) -> Any:
        return self.Glyph(value)

    def Submit(self, value: Any) -> Any:
        return self.Glyph(value)

    def Transact(self, value: Any) -> Any:
        return self.Glyph(value)

    def Glyph(self, glyph: Any) -> Any:
        payload = self.Refine(glyph)
        self.Emit(payload)
        self.Wake()
        return payload
