from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
import Field
import Dream
TransferPair = Tuple[str, int]
PURGE = 'PURGE'
WHISPER = 'WHISPER'
RALLY = 'RALLY'
WRATH = 'WRATH'
DEFECT = 'DEFECT'
SALT = 'SALT'
maingate = '9000'
defaultmode = ''
defaultskeleton = 'skeleton'
defaultsecret = 'password'
defaultsoul = 'SATOSHI'
defaultgenesis = 1
textmaxlen = 68
expectedspendcounts = {WHISPER: 1, RALLY: 5, WRATH: 23, DEFECT: 6}

@dataclass
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
    lock: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'kind', str(self.kind or '').upper())
        object.__setattr__(self, 'key', str(self.key or '').strip())
        object.__setattr__(self, 'pairs', tuple(((str(k or '').strip(), int(v or 0)) for k, v in tuple(self.pairs or ()))))
        object.__setattr__(self, 'text', str(self.text or '')[:textmaxlen])
        object.__setattr__(self, 'lock', self.lock)

    @property
    def totalamount(self) -> int:
        return sum((int(amount) for _, amount in self.pairs))

    @property
    def spendcount(self) -> int:
        return len(self.pairs)

    @classmethod
    def fromany(cls, value: Any) -> 'Glyph':
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls(kind=str(value.get('kind', '') or ''), key=str(value.get('key', '') or '').strip(), pairs=tuple(value.get('pairs', ()) or ()), text=str(value.get('text', '') or ''), lock=value.get('lock', None))
        return cls(kind=str(getattr(value, 'kind', '') or getattr(getattr(value, 'action', ''), 'value', getattr(value, 'action', '')) or ''), key=str(getattr(value, 'key', '') or '').strip(), pairs=tuple(getattr(value, 'pairs', ()) or ()), text=str(getattr(value, 'text', '') or ''), lock=getattr(value, 'lock', None))

def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()

def sha256hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def safeint(value: Any, fallback: int=1) -> int:
    try:
        text = str(value or '').strip()
        return int(text) if text else int(fallback)
    except Exception:
        return int(fallback)

def coercestate(state: Any=None, *, mode: str='', gate: str='', skeleton: str='', soul: str='', secret: str='', key: str='', genesis: Any=None) -> State:
    if state is not None:
        if isinstance(state, State):
            return state
        if isinstance(state, dict):
            return State(mode=str(state.get('mode', mode or defaultmode) or defaultmode), gate=str(state.get('gate', gate or maingate) or maingate), skeleton=str(state.get('skeleton', skeleton or defaultskeleton) or defaultskeleton), soul=str(state.get('soul', soul or defaultsoul) or defaultsoul), secret=str(state.get('secret', secret or defaultsecret) or defaultsecret), key=str(state.get('key', key or '') or ''), genesis=max(1, safeint(state.get('genesis', genesis or defaultgenesis), defaultgenesis)))
        return State(mode=str(getattr(state, 'mode', mode or defaultmode) or defaultmode), gate=str(getattr(state, 'gate', gate or maingate) or maingate), skeleton=str(getattr(state, 'skeleton', skeleton or defaultskeleton) or defaultskeleton), soul=str(getattr(state, 'soul', soul or defaultsoul) or defaultsoul), secret=str(getattr(state, 'secret', secret or defaultsecret) or defaultsecret), key=str(getattr(state, 'key', key or '') or ''), genesis=max(1, safeint(getattr(state, 'genesis', genesis or defaultgenesis), defaultgenesis)))
    return State(mode=str(mode or defaultmode), gate=str(gate or maingate), skeleton=str(skeleton or defaultskeleton), soul=str(soul or defaultsoul), secret=str(secret or defaultsecret), key=str(key or ''), genesis=max(1, safeint(genesis or defaultgenesis, defaultgenesis)))

def derivesigningkeyfromstate(soul: str, secret: str) -> Ed25519PrivateKey:
    seed = sha256(f'Byzantium::State::V3::{soul}::{secret}'.encode('utf-8'))
    return Ed25519PrivateKey.from_private_bytes(seed)

class Vault:

    def __init__(self, dream: object | None=None, *, state: Any=None, mode: str='', gate: str='', skeleton: str='', soul: str='', secret: str='', genesis: Any=None, citadel: object | None=None) -> None:
        self.secret = ''
        self.privatekey = None
        self.publickey = None
        self.publickeyhex = ''
        self.state = coercestate(state, mode=getattr(state, 'mode', mode or ''), gate=gate, skeleton=skeleton, soul=soul, secret=secret, genesis=genesis)
        self.mode = self.state.mode
        self.gate = self.state.gate
        self.skeleton = self.state.skeleton
        self.soul = self.state.soul
        self.key = self.state.key
        self.genesis = self.state.genesis
        self.dream = self.wakeDream(dream, citadel=citadel)
        Dream.dream = self.dream
        self.Genesis(self.state)

    def wake(self):
        child = self.dream
        if child and hasattr(child, 'wake'):
            return child.wake()
        return child

    def wakeDream(self, dream: object | None, *, citadel: object | None=None) -> object:
        if dream is None:
            out = Dream.Dream(citadel=citadel)
            return out
        if isinstance(dream, type):
            try:
                out = dream(citadel=citadel)
                return out
            except TypeError:
                out = dream()
                return out
        return dream

    def syncstate(self, state: State) -> State:
        self.state = state
        self.mode = self.state.mode
        self.gate = self.state.gate
        self.skeleton = self.state.skeleton
        self.soul = self.state.soul
        self.key = self.state.key
        self.genesis = self.state.genesis
        return self.state

    def Genesis(self, state: Any=None) -> Any:
        source = state if state is not None else self.state
        rawstate = coercestate(source, mode=getattr(source, 'mode', getattr(self, 'mode', '') or ''), gate=getattr(source, 'gate', getattr(self, 'gate', '') or ''), skeleton=getattr(source, 'skeleton', getattr(self, 'skeleton', '') or ''), soul=getattr(source, 'soul', getattr(self, 'soul', '') or ''), secret=getattr(source, 'secret', getattr(self, 'secret', '') or ''), key=getattr(source, 'key', getattr(self, 'key', '') or ''), genesis=getattr(source, 'genesis', getattr(self, 'genesis', defaultgenesis)))
        self.secret = str(rawstate.secret or defaultsecret)
        self.privatekey = derivesigningkeyfromstate(rawstate.soul, self.secret)
        self.publickey = self.privatekey.public_key()
        self.publickeyhex = self.publickey.public_bytes(encoding=Encoding.Raw, format=PublicFormat.Raw).hex()
        lawfulstate = replace(rawstate, secret='', key=self.publickeyhex)
        self.syncstate(lawfulstate)
        if hasattr(self.dream, 'Genesis'):
            return self.dream.Genesis(self.state)
        self.sendstate()
        self.wake()
        return self.state

    def sendstate(self) -> Any:
        if hasattr(self.dream, 'box'):
            self.dream.box.vault = self.state
            return self.dream.box.vault
        raise AttributeError('Dream must expose box')

    def emit(self, payload: Any) -> Any:
        if hasattr(self.dream, 'box'):
            self.dream.box.vault = payload
            try:
                if hasattr(payload, 'key'):
                    pass
            except Exception:
                pass
            return self.dream.box.vault
        raise AttributeError('Dream must expose box')

    def signdigesthex(self, digesthex: str) -> str:
        Field.VerifyHash(digesthex, field_name='digesthex')
        return self.privatekey.sign(bytes.fromhex(digesthex)).hex()

    def normalizepairs(self, pairs: Sequence[TransferPair]) -> List[TransferPair]:
        out: List[TransferPair] = []
        for recipient, amount in list(pairs or []):
            recipientkey = str(recipient or '').strip()
            amountint = int(amount)
            Field.VerifyKey(recipientkey)
            Field.VerifyNonNegative(amountint, field_name='amount')
            out.append((recipientkey, amountint))
        return out

    def assertglyphshape(self, glyph: Glyph, pairs: Sequence[TransferPair]) -> None:
        spendcount = len(tuple(pairs))
        if glyph.kind in expectedspendcounts:
            expected = expectedspendcounts[glyph.kind]
            if spendcount != expected:
                raise ValueError(f'{glyph.kind} requires exactly {expected} spend legs')
        else:
            pass
        totalamount = sum((int(amount) for _, amount in pairs))
        if glyph.kind in expectedspendcounts and totalamount <= 0:
            raise ValueError('totalamount must be positive')

    def textbody(self, rawtext: str) -> Field.Text:
        out = Field.Text(text=str(rawtext or '')[:textmaxlen])
        return out

    def taggedtext(self, glyph: Glyph) -> str:
        kind = str(glyph.kind or '').strip().lower()
        rawtext = str(glyph.text or '')
        if not kind:
            return rawtext[:textmaxlen]
        suffix = f'|{kind}'
        if rawtext.lower().endswith(suffix):
            return rawtext[:textmaxlen]
        return f'{rawtext}{suffix}'[:textmaxlen]

    def texthash(self, textbody: Field.Text) -> str:
        out = Field.TextHash(textbody)
        return out

    def saltbody(self, pairs: Sequence[TransferPair]) -> Tuple[Field.Salt, ...]:
        out = tuple((Field.Salt(key=recipientkey, salt=int(amount)) for recipientkey, amount in pairs))
        return out

    def salthash(self, saltbody: Tuple[Field.Salt, ...]) -> str:
        out = Field.SaltHash(self.publickeyhex, saltbody)
        return out

    def coerce_lock(self, value: Any) -> Field.Lock:
        if isinstance(value, Field.Lock):
            return value
        if isinstance(value, dict):
            out = Field.Lock(parent=str(value.get('parent', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX), child=str(value.get('child', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX))
            return out
        out = Field.Lock(parent=str(getattr(value, 'parent', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX), child=str(getattr(value, 'child', Field.ZERO_HASH_HEX) or Field.ZERO_HASH_HEX))
        return out

    def blanklock(self, lock: Field.Lock) -> bool:
        out = str(lock.parent or Field.ZERO_HASH_HEX) == Field.ZERO_HASH_HEX and str(lock.child or Field.ZERO_HASH_HEX) == Field.ZERO_HASH_HEX
        return out

    def baselock(self, glyph: Glyph) -> Field.Lock:
        currentlock = self.coerce_lock(glyph.lock)
        if self.blanklock(currentlock):
            out = Field.Lock(parent=Field.ZERO_HASH_HEX, child=Field.ZERO_HASH_HEX)
            return out
        return currentlock

    def childlockhex(self, kind: str, parent: str, texthash: str, pairs: Sequence[TransferPair]) -> str:
        parts = [str(kind or '').upper(), str(parent or Field.ZERO_HASH_HEX), str(texthash or Field.ZERO_HASH_HEX)]
        for recipientkey, amount in pairs:
            parts.extend([str(recipientkey), str(int(amount))])
        out = sha256hex('|'.join(parts).encode('utf-8'))
        return out

    def lockhash(self, lockbody: Field.Lock) -> str:
        out = Field.LockHash(lockbody)
        return out

    def nextlock(self, glyph: Glyph, texthash: str, pairs: Sequence[TransferPair]) -> Field.Lock:
        currentlock = self.baselock(glyph)
        parent = str(currentlock.child or Field.ZERO_HASH_HEX)
        child = self.childlockhex(glyph.kind, parent, texthash, pairs)
        out = Field.Lock(parent=parent, child=child)
        return out

    def lawfulsalt(self, glyph: Glyph) -> Field.SaltGlyph:
        pairs = self.normalizepairs(glyph.pairs)
        self.assertglyphshape(glyph, pairs)
        textbody = self.textbody(self.taggedtext(glyph))
        texthash = self.texthash(textbody)
        saltbody = self.saltbody(pairs)
        salthash = self.salthash(saltbody)
        lockbody = self.nextlock(glyph, texthash, pairs)
        lockhash = self.lockhash(lockbody)
        locksign = self.signdigesthex(lockhash)
        proto = Field.SaltGlyph(key=self.publickeyhex, saltbody=saltbody, lockbody=lockbody, textbody=textbody, salthash=salthash, lockhash=lockhash, texthash=texthash, sign=Field.NULL_SIGN_HEX, locksign=locksign)
        sign = self.signdigesthex(Field.SaltGlyphHash(proto))
        body = replace(proto, sign=sign)
        Field.VerifySalt(body)
        return body

    def lawfulpurge(self, glyph: Glyph) -> Dict[str, Any]:
        targetkey = str(glyph.key or '').strip()
        Field.VerifyKey(targetkey)
        return {'kind': PURGE, 'key': targetkey}

    def whisper(self, glyph: Glyph) -> Field.SaltGlyph:
        return self.lawfulsalt(glyph)

    def rally(self, glyph: Glyph) -> Field.SaltGlyph:
        return self.lawfulsalt(glyph)

    def wrath(self, glyph: Glyph) -> Field.SaltGlyph:
        return self.lawfulsalt(glyph)

    def purge(self, glyph: Glyph) -> Dict[str, Any]:
        return self.lawfulpurge(glyph)

    def defect(self, glyph: Glyph) -> Field.SaltGlyph:
        return self.lawfulsalt(glyph)

    def refine(self, glyph: Any) -> Any:
        baton = Glyph.fromany(glyph)
        if baton.kind == PURGE:
            return self.purge(baton)
        if baton.kind == WHISPER:
            return self.whisper(baton)
        if baton.kind == RALLY:
            return self.rally(baton)
        if baton.kind == WRATH:
            return self.wrath(baton)
        if baton.kind == DEFECT:
            return self.defect(baton)
        raise ValueError(f'unsupported glyph kind: {baton.kind!r}')

    def intent(self, value: Any) -> Any:
        return self.glyph(value)

    def submit(self, value: Any) -> Any:
        return self.glyph(value)

    def transact(self, value: Any) -> Any:
        return self.glyph(value)

    def glyph(self, glyph: Any) -> Any:
        payload = self.refine(glyph)
        try:
            if hasattr(payload, 'key'):
                pass
        except Exception:
            pass
        self.emit(payload)
        self.wake()
        return payload
__all__ = ['State', 'Glyph', 'Vault', 'TransferPair', 'PURGE', 'WHISPER', 'RALLY', 'WRATH', 'DEFECT', 'SALT', 'textmaxlen', 'derivesigningkeyfromstate']
