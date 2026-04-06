from __future__ import annotations
from dataclasses import dataclass, replace
import re
from typing import Any
import Citadel
import Crypt
import Field

PurgeGlyph = 'PURGE'
DreamGlyph = 'DREAM'
SaltGlyph = 'SALT'
MONUMENT_SLOTS = 3
MONUMENT_NAME_W = 8
MONUMENT_RE = re.compile(r'^(.{0,8})\s+([+-]?[\d,]+):\s*(.*)$')

def _seat(state: Field.State, key: str) -> int | None:
    key = str(key or '').strip()
    if not key:
        return None
    for q, cell in enumerate(state.cells):
        if str(cell.key or '').strip() == key:
            return q
    return None

def _samefile(a: int, b: int) -> bool:
    return int(a) // Field.SEATS_PER_FILE == int(b) // Field.SEATS_PER_FILE

def _defectshape(glyph: Field.SaltGlyph) -> bool:
    return len(tuple(glyph.saltbody or ())) == 6

def _defectparts(glyph: Field.SaltGlyph) -> tuple[tuple[Field.Salt, ...], Field.Salt | None]:
    legs = tuple(glyph.saltbody or ())
    if len(legs) != 6:
        return (legs, None)
    return (legs[:-1], legs[-1])

def _monumentparse(line: Any) -> tuple[str, int | None, str]:
    text = str(line or '').rstrip()
    match = MONUMENT_RE.match(text)
    if not match:
        head = text[:MONUMENT_NAME_W].strip()
        tail = text[MONUMENT_NAME_W:].strip() if len(text) > MONUMENT_NAME_W else ''
        return (head, None, tail or text)
    head = str(match.group(1) or '').strip()
    score_text = str(match.group(2) or '').replace(',', '').strip()
    body = str(match.group(3) or '')
    try:
        score = int(score_text)
    except Exception:
        score = None
    return (head, score, body)

def _monumentscore(line: Any) -> int:
    _head, score, _body = _monumentparse(line)
    return abs(int(score or 0))

def _monumentclean(text: Any, n: int=59) -> str:
    body = str(text or '').replace('\r', ' ').replace('\n', ' ').strip()
    return body[:n]

def _monumentname(name: Any) -> str:
    raw = str(name or '').strip()
    if not raw:
        return 'UNKNOWN'
    return raw[:MONUMENT_NAME_W].ljust(MONUMENT_NAME_W)

def _monumentline(name: Any, total: int, text: Any) -> str:
    score = f'{int(total):+,}'
    body = _monumentclean(text)
    return f'{_monumentname(name)} {score}:{body}' if body else f'{_monumentname(name)} {score}:'

def _ashkind(text: Any) -> str:
    raw = str(text or '')
    head, sep, _ = raw.partition('|')
    return head.strip().upper() if sep else ''

def _ashbroadcasttotal(glyph: Field.SaltGlyph, viewer: str, sender: str, kind: str) -> int:
    legs = tuple(getattr(glyph, 'saltbody', ()) or ())
    if viewer == sender:
        return sum((int(getattr(leg, 'salt', 0) or 0) for leg in legs))
    direct = sum((int(getattr(leg, 'salt', 0) or 0) for leg in legs if str(getattr(leg, 'key', '') or '').strip() == viewer))
    if direct > 0:
        return direct
    if kind == 'DEFECT':
        return max((int(getattr(leg, 'salt', 0) or 0) for leg in legs), default=0)
    return 0

@dataclass
class Box:
    vault: Any = None
    crypt: Any = None
    ashfall: Any = None

class Dream:

    def __init__(self, citadel: Any=None, crypt: Any=None):
        self.box = Box()
        self.state: Field.State | None = None
        self.citadel = citadel
        self.crypt = crypt
        self.glyph: Any = None
        self.changed = False
        self.bootflare = False

    def wakecitadel(self):
        if self.citadel is not None:
            return self.citadel
        self.citadel = Citadel.citadel
        return self.citadel

    def wakecrypt(self):
        if self.crypt is not None:
            return self.crypt
        live = getattr(Crypt, 'crypt', None)
        if live is not None:
            self.crypt = live
            return self.crypt
        live = getattr(Crypt, '_CRYPT', None)
        if live is not None:
            self.crypt = live
            return self.crypt
        return None

    def Genesis(self, state: Any):
        self.crypt = Crypt.Crypt(state=state)
        Crypt.crypt = self.crypt
        Crypt._CRYPT = self.crypt
        if hasattr(self.crypt, 'Genesis'):
            return self.crypt.Genesis(state)
        return self.crypt

    def wake(self):
        self.changed = False
        self.routevault()
        self.routecrypt()
        if self.changed:
            self.publish()
        return self.state

    def awake(self):
        return self.wake()

    def route(self):
        return self.wake()

    def monumentname(self, key: str) -> str:
        key = str(key or '').strip()
        if self.state is not None:
            cell = Field.FindCell(self.state, key)
            if cell is not None:
                soul = str(getattr(cell, 'soul', '') or '').strip()
                if soul:
                    return soul
        if self.state is not None:
            selfkey = str(self.state.self[1] or '').strip()
            selfsoul = str(self.state.self[0] or '').strip()
            if key and key == selfkey and selfsoul:
                return selfsoul
        return key or 'UNKNOWN'

    def monumenttotal(self, glyph: Field.SaltGlyph) -> int:
        return sum((int(getattr(leg, 'salt', 0) or 0) for leg in tuple(glyph.saltbody or ())))

    def monumentline(self, glyph: Field.SaltGlyph) -> str:
        return _monumentline(self.monumentname(glyph.key), self.monumenttotal(glyph), glyph.textbody.text)

    def updatemonument(self, state: Field.State, glyph: Field.SaltGlyph) -> Field.State:
        candidate = self.monumentline(glyph)
        entries = [str(line) for line in tuple(getattr(state, 'monument', ()) or ()) if str(line or '').strip()]
        _head, score, _body = _monumentparse(candidate)
        if score is None or score <= 0:
            return state
        pool = list(entries)
        if candidate not in pool:
            pool.append(candidate)
        ranked = sorted(pool, key=lambda line: (_monumentscore(line), str(line)), reverse=True)[:MONUMENT_SLOTS]
        monument = tuple(ranked)
        if monument == tuple(getattr(state, 'monument', ()) or ()):
            return state
        return Field.State(cells=state.cells, self=state.self, monument=monument)

    def acceptstate(self, state: Any, *, publish: bool=True):
        if not isinstance(state, Field.State):
            raise TypeError('Dream.acceptstate expects Field.State')
        firstreal = self.state is None and bool(getattr(state, 'cells', ()) or ())
        self.state = state
        if firstreal and (not self.bootflare):
            self.bootflare = True
            flare = self.purgeflare()
            self.forward(flare)
        if publish:
            self.publish()
        return self.state

    def publish(self):
        if self.state is None:
            return None
        citadel = self.wakecitadel()
        citadel.state = self.state
        if self.box.ashfall is not None:
            try:
                citadel.ashfall(self.box.ashfall)
            except Exception:
                pass
            self.box.ashfall = None
        return self.state

    def scrub(self, state: Field.State | None=None) -> Field.State | None:
        body = self.state if state is None else state
        if body is None:
            return None
        return Field.Scrub(body)

    def routevault(self):
        glyph = self.box.vault
        if glyph is None:
            return self.state
        self.box.vault = None
        if self.state is None:
            if isinstance(glyph, Field.State):
                self.acceptstate(glyph, publish=True)
                return self.state
            raise TypeError('Dream.routevault expected Field.State during bootstrap')
        self.mutate(glyph, source='vault')
        return self.state

    def routecrypt(self):
        glyph = self.box.crypt
        if glyph is None:
            return self.state
        self.box.crypt = None
        if self.state is None:
            if isinstance(glyph, Field.State):
                self.acceptstate(glyph, publish=True)
                return self.state
            raise TypeError('Dream.routecrypt expected Field.State during bootstrap')
        self.mutate(glyph, source='crypt')
        return self.state

    def ashfall(self, glyph: Field.SaltGlyph) -> Any:
        if self.state is None:
            return None
        viewer = str(self.state.self[1] or '').strip()
        sender = str(glyph.key or '').strip()
        if not viewer or not sender:
            return None
        text = str(getattr(getattr(glyph, 'textbody', None), 'text', '') or '')
        actionkind = _ashkind(text)
        total = _ashbroadcasttotal(glyph, viewer, sender, actionkind)
        if viewer != sender and total <= 0:
            return None
        sendercell = Field.FindCell(self.state, sender)
        sendername = str(sendercell.soul or '') if sendercell is not None else str(self.state.self[0] or '')
        payload = {'sender': sendername or sender, 'kind': SaltGlyph, 'text': text, 'total': int(total)}
        self.box.ashfall = payload
        return payload

    def selfkey(self) -> str:
        if self.state is None:
            return ''
        return str(self.state.self[1] or '').strip()

    def purgekey(self, glyph: Any) -> str:
        if isinstance(glyph, dict):
            return str(glyph.get('key', '') or '').strip()
        return str(getattr(glyph, 'key', '') or '').strip()

    def pristine(self, state: Field.State | None=None) -> bool:
        body = self.state if state is None else state
        if body is None:
            return True
        return len(tuple(getattr(body, 'monument', ()) or ())) == 0

    def _withchainbit(self, cell: Field.Cell, chainbit: int) -> Field.Cell:
        cb = 1 if int(chainbit) else 0
        lb = 1 if int(cell.purge.lockbit) or cb else 0
        return replace(cell, purge=Field.Purge(chainbit=cb, lockbit=lb))

    def _latchonly(self, cell: Field.Cell) -> Field.Cell:
        return replace(cell, purge=Field.Purge(chainbit=0, lockbit=1))

    def _clearpurge(self, cell: Field.Cell) -> Field.Cell:
        return replace(cell, purge=Field.Purge(chainbit=0, lockbit=0))

    def _activechain(self, cell: Field.Cell, chain: Field.Chain) -> int:
        child = str(getattr(cell.lock, 'child', '') or '').strip()
        return 1 if chain.linked and child not in ('', Field.ZERO_HASH_HEX) else 0

    def _idempotentsalt(self, current: Field.Cell | None, glyph: Field.SaltGlyph) -> bool:
        if current is None:
            return False
        same_parent = str(current.lock.parent or '') == str(glyph.lockbody.parent or '')
        same_child = str(current.lock.child or '') == str(glyph.lockbody.child or '')
        same_sign = str(current.sign or '') == str(glyph.locksign or '')
        return bool(same_parent and same_child and same_sign)

    def _stampchainsalt(self, state: Field.State, glyph: Field.SaltGlyph, chains: tuple[Field.Chain, ...]) -> Field.State:
        cells = {cell.key: cell for cell in state.cells}
        signer = str(glyph.key or '').strip()
        if signer and signer in cells and (len(chains) >= 1):
            cells[signer] = self._withchainbit(cells[signer], 1 if chains[0].linked else 0)
        seen = set()
        credit_keys = []
        for leg in glyph.saltbody:
            key = str(getattr(leg, 'key', '') or '').strip()
            if not key or key == signer or key in seen:
                continue
            seen.add(key)
            credit_keys.append(key)
        for idx, key in enumerate(credit_keys, start=1):
            if key in cells and idx < len(chains):
                cells[key] = self._withchainbit(cells[key], 1) if chains[idx].linked else self._withchainbit(cells[key], 0)
        stamped = Field.State(cells=tuple((cells[cell.key] for cell in state.cells)), self=state.self, monument=state.monument)
        return stamped

    def _stampchaindream(self, state: Field.State, chains: tuple[Field.Chain, ...]) -> Field.State:
        if len(chains) != len(state.cells):
            return state
        cells = []
        for cell, chain in zip(state.cells, chains):
            if chain.linked:
                active = self._activechain(cell, chain)
                cells.append(self._withchainbit(cell, 1) if active else self._latchonly(cell))
            else:
                cells.append(self._withchainbit(cell, 0))
        stamped = Field.State(cells=tuple(cells), self=state.self, monument=state.monument)
        return stamped

    def _applypurgekey(self, state: Field.State, key: str) -> Field.State:
        key = str(key or '').strip()
        if not key:
            return state
        target = Field.FindCell(state, key)
        if target is None:
            return state
        cleared = self._clearpurge(target)
        nextstate = Field.ReplaceCell(state, cleared)
        return nextstate

    def _scrubpurge(self, state: Field.State) -> Field.State:
        cleared = tuple((replace(cell, purge=Field.Purge(chainbit=0, lockbit=0)) for cell in state.cells))
        out = Field.State(cells=cleared, self=state.self, monument=state.monument)
        return out

    def purgeflare(self) -> dict[str, Any]:
        return {'kind': PurgeGlyph, 'key': self.selfkey()}

    def mutate(self, glyph: Any, source: str=''):
        kind = self.kind(glyph)
        if kind == PurgeGlyph:
            mutated = self.mutatepurge(glyph, source=source)
        elif kind == DreamGlyph:
            mutated = self.mutatedream(glyph, source=source)
        else:
            mutated = self.mutatesalt(glyph, source=source)
        if mutated:
            self.changed = True
            if kind == SaltGlyph:
                self.ashfall(glyph)
            echo = True
            if kind == PurgeGlyph:
                echo = False
            elif source == 'crypt' and kind == DreamGlyph:
                echo = False
            if echo:
                self.forward(glyph)
        return mutated

    def defectviable(self, state: Field.State, glyph: Field.SaltGlyph) -> bool:
        spend, victim = _defectparts(glyph)
        if victim is None:
            return False
        signerq = _seat(state, glyph.key)
        victimq = _seat(state, victim.key)
        if signerq is None or victimq is None:
            return False
        if signerq == victimq:
            return False
        if _samefile(signerq, victimq):
            return False
        signer = state.cells[signerq]
        target = state.cells[victimq]
        if int(target.salt) >= int(signer.salt):
            return False
        total = sum((int(leg.salt) for leg in spend))
        floor = 10000 if signerq % Field.SEATS_PER_FILE == 0 else 1000
        if total != floor:
            return False
        if any((int(leg.salt) <= 0 for leg in spend)):
            return False
        filekeys = {cell.key for i, cell in enumerate(state.cells) if i != signerq and _samefile(i, signerq)}
        spendkeys = {str(leg.key or '').strip() for leg in spend}
        if spendkeys != filekeys:
            return False
        if int(victim.salt) != 0:
            return False
        return True

    def defectswap(self, state: Field.State, glyph: Field.SaltGlyph) -> Field.State:
        _spend, victim = _defectparts(glyph)
        if victim is None:
            return state
        signerq = _seat(state, glyph.key)
        victimq = _seat(state, victim.key)
        if signerq is None or victimq is None:
            return state
        cells = list(state.cells)
        cells[signerq], cells[victimq] = (cells[victimq], cells[signerq])
        out = Field.State(cells=tuple(cells), self=state.self, monument=state.monument)
        return out

    def mutatesalt(self, glyph: Any, source: str='') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.SaltGlyph):
            raise TypeError('Dream.mutatesalt expects Field.SaltGlyph')
        defect = _defectshape(glyph)
        try:
            Field.VerifySalt(glyph)
            current = Field.FindCell(self.state, glyph.key)
            if defect and (not self.defectviable(self.state, glyph)):
                return False
            if source == 'crypt' and self._idempotentsalt(current, glyph):
                return False
            nextstate, chains = Field.MutateReceipt(self.state, glyph)
            if defect:
                nextstate = self.defectswap(nextstate, glyph)
            nextstate = self.updatemonument(nextstate, glyph)
        except Exception as exc:
            return False
        changed = self.commit(nextstate)
        if not changed or self.state is None:
            return False
        self.state = self._stampchainsalt(self.state, glyph, chains)
        return True

    def mutatedream(self, glyph: Any, source: str='') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.State):
            raise TypeError('Dream.mutatedream expects Field.State')
        try:
            nextstate, chains = Field.MutateState(self.state, glyph)
        except Exception as exc:
            return False
        changed = self.commit(nextstate)
        if not changed or self.state is None:
            return False
        self.state = self._stampchaindream(self.state, chains)
        return True

    def mutatepurge(self, glyph: Any, source: str='') -> bool:
        if self.state is None:
            return False
        key = self.purgekey(glyph)
        if source == 'vault':
            if key and key == self.selfkey():
                nextstate = self._scrubpurge(self.state)
                changed = self.commit(nextstate)
                flare = self.purgeflare()
                self.forward(flare)
                return changed or True
            nextstate = self._applypurgekey(self.state, key)
            changed = self.commit(nextstate)
            flare = self.purgeflare()
            self.forward(flare)
            return changed or True
        if source == 'crypt':
            if key and key == self.selfkey():
                return False
            if self.pristine(self.state):
                return False
            dream = self.scrub(self.state)
            if dream is None:
                return False
            self.forward(dream)
            return True
        return False

    def commit(self, nextstate: Field.State) -> bool:
        nextstate = Field.Stasis(nextstate)
        if self.state is None:
            self.state = nextstate
            return True
        if nextstate == self.state:
            return False
        before = self.state.saltTotal
        after = nextstate.saltTotal
        self.state = nextstate
        return True

    def forward(self, glyph: Any):
        self.glyph = glyph
        crypt = self.wakecrypt()
        if crypt is None:
            return glyph
        packet = self.packet(glyph)
        try:
            crypt.glyph = packet
        except Exception:
            pass
        if hasattr(crypt, 'emitGlyph'):
            try:
                crypt.emitGlyph(packet)
            except Exception:
                pass
        return packet

    def packet(self, glyph: Any) -> Any:
        kind = self.kind(glyph)
        if kind == DreamGlyph:
            return self.boxdream(glyph)
        if kind == SaltGlyph:
            return self.boxsalt(glyph)
        if isinstance(glyph, dict):
            return dict(glyph)
        if isinstance(glyph, str):
            return str(glyph)
        return glyph

    def kind(self, glyph: Any) -> str:
        if isinstance(glyph, str):
            out = PurgeGlyph
            return out
        if isinstance(glyph, Field.State):
            out = DreamGlyph
            return out
        if isinstance(glyph, Field.SaltGlyph):
            out = SaltGlyph
            return out
        if isinstance(glyph, dict):
            kind = str(glyph.get('kind', '') or '').strip().upper()
            if kind:
                return kind
            if 'cells' in glyph:
                return DreamGlyph
            if 'saltbody' in glyph and 'lockbody' in glyph and ('textbody' in glyph):
                return SaltGlyph
        return SaltGlyph

    def boxsalt(self, glyph: Field.SaltGlyph) -> dict[str, Any]:
        return {'key': glyph.key, 'saltbody': [self.boxleg(leg) for leg in glyph.saltbody], 'lockbody': self.boxlock(glyph.lockbody), 'textbody': {'text': glyph.textbody.text}, 'salthash': glyph.salthash, 'lockhash': glyph.lockhash, 'texthash': glyph.texthash, 'sign': glyph.sign, 'locksign': glyph.locksign}

    def boxleg(self, leg: Field.Salt) -> dict[str, Any]:
        return {'key': leg.key, 'salt': leg.salt}

    def boxlock(self, lock: Field.Lock) -> dict[str, Any]:
        return {'parent': lock.parent, 'child': lock.child}

    def boxpurge(self, purge: Field.Purge) -> dict[str, Any]:
        return {'chainbit': purge.chainbit, 'lockbit': purge.lockbit}

    def boxcell(self, cell: Field.Cell) -> dict[str, Any]:
        return {'soul': cell.soul, 'key': cell.key, 'salt': cell.salt, 'purge': self.boxpurge(cell.purge), 'lock': self.boxlock(cell.lock), 'sign': cell.sign}

    def boxdream(self, state: Field.State) -> dict[str, Any]:
        scrubbed = Field.Scrub(state)
        return {'kind': DreamGlyph, 'self': [scrubbed.self[0], scrubbed.self[1]], 'monument': list(scrubbed.monument), 'cells': [self.boxcell(cell) for cell in scrubbed.cells]}

dream = Dream()
_DREAM = dream
