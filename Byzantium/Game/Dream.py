from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Any

import Citadel
import Crypt
import Field

PurgeGlyph = 'purge'
DreamGlyph = 'dream'
SaltGlyph = 'salt'
MonumentSlots = 3
MonumentNameWidth = 8
MonumentPattern = re.compile(r'^(.{0,8})\s+([+-]?[\d,]+):\s*(.*)$')


def Seat(state: Field.State, key: str) -> int | None:
    key = str(key or '').strip()
    if not key:
        return None
    for q, cell in enumerate(state.cells):
        if str(cell.key or '').strip() == key:
            return q
    return None


def SameFile(a: int, b: int) -> bool:
    return int(a) // Field.SeatsPerFile == int(b) // Field.SeatsPerFile


def DefectShape(glyph: Field.SaltGlyph) -> bool:
    return len(tuple(glyph.saltbody or ())) == 6


def DefectParts(glyph: Field.SaltGlyph) -> tuple[tuple[Field.Salt, ...], Field.Salt | None]:
    legs = tuple(glyph.saltbody or ())
    if len(legs) != 6:
        return (legs, None)
    return (legs[:-1], legs[-1])


def MonumentParse(line: Any) -> tuple[str, int | None, str]:
    text = str(line or '').rstrip()
    match = MonumentPattern.match(text)
    if not match:
        head = text[:MonumentNameWidth].strip()
        tail = text[MonumentNameWidth:].strip() if len(text) > MonumentNameWidth else ''
        return (head, None, tail or text)
    head = str(match.group(1) or '').strip()
    scoretext = str(match.group(2) or '').replace(',', '').strip()
    body = str(match.group(3) or '')
    try:
        score = int(scoretext)
    except Exception:
        score = None
    return (head, score, body)


def MonumentScore(line: Any) -> int:
    head, score, body = MonumentParse(line)
    return abs(int(score or 0))


def AshSplit(text: Any) -> tuple[str, str]:
    raw = str(text or '').replace('\r', ' ').replace('\n', ' ').strip()
    body, sep, kind = raw.rpartition('|')
    if sep:
        return (kind.strip().lower(), body.strip())
    return ('', raw)


def MonumentClean(text: Any, n: int = 60) -> str:
    kind, body = AshSplit(text)
    return body[:n]


def MonumentName(name: Any) -> str:
    raw = str(name or '').strip()
    if not raw:
        return 'UNKNOWN'
    return raw[:MonumentNameWidth].ljust(MonumentNameWidth)


def MonumentLine(name: Any, total: int, text: Any) -> str:
    score = f'{int(total):+,}'
    body = MonumentClean(text)
    return f'{MonumentName(name)} {score}:{body}' if body else f'{MonumentName(name)} {score}:'


def AshKind(text: Any) -> str:
    kind, body = AshSplit(text)
    return kind


def AshBroadcastTotal(glyph: Field.SaltGlyph, viewer: str, sender: str, kind: str) -> int:
    legs = tuple(getattr(glyph, 'saltbody', ()) or ())
    if viewer == sender:
        return sum((int(getattr(leg, 'salt', 0) or 0) for leg in legs))
    direct = sum((int(getattr(leg, 'salt', 0) or 0) for leg in legs if str(getattr(leg, 'key', '') or '').strip() == viewer))
    if direct > 0:
        return direct
    if kind == 'defect':
        return max((int(getattr(leg, 'salt', 0) or 0) for leg in legs), default=0)
    return 0


@dataclass
class Box:
    vault: Any = None
    crypt: Any = None
    ashfall: Any = None


class Dream:

    def __init__(self, citadel: Any = None, crypt: Any = None):
        self.box = Box()
        self.state: Field.State | None = None
        self.citadel = citadel
        self.crypt = crypt
        self.glyph: Any = None
        self.changed = False
        self.bootflare = False

    def WakeCitadel(self):
        if self.citadel is not None:
            return self.citadel
        self.citadel = Citadel.Citadel
        return self.citadel

    def WakeCrypt(self):
        if self.crypt is not None:
            return self.crypt
        live = getattr(Crypt, 'crypt', None)
        if live is not None:
            self.crypt = live
            return self.crypt
        return None

    def Genesis(self, state: Any):
        self.crypt = Crypt.Crypt(state=state)
        Crypt.crypt = self.crypt
        return self.crypt

    def Wake(self):
        self.changed = False
        self.RouteVault()
        self.RouteCrypt()
        if self.changed:
            self.Publish()
        return self.state

    def Awake(self):
        return self.Wake()

    def Route(self):
        return self.Wake()

    def MonumentName(self, key: str) -> str:
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

    def MonumentTotal(self, glyph: Field.SaltGlyph) -> int:
        return sum((int(getattr(leg, 'salt', 0) or 0) for leg in tuple(glyph.saltbody or ())))

    def MonumentEntry(self, glyph: Field.SaltGlyph) -> str:
        return MonumentLine(self.MonumentName(glyph.key), self.MonumentTotal(glyph), glyph.textbody.text)

    def UpdateMonument(self, state: Field.State, glyph: Field.SaltGlyph) -> Field.State:
        candidate = self.MonumentEntry(glyph)
        entries = [str(line) for line in tuple(getattr(state, 'monument', ()) or ()) if str(line or '').strip()]
        head, score, body = MonumentParse(candidate)
        if score is None or score <= 0:
            return state
        pool = list(entries)
        if candidate not in pool:
            pool.append(candidate)
        ranked = sorted(pool, key=lambda line: (MonumentScore(line), str(line)), reverse=True)[:MonumentSlots]
        monument = tuple(ranked)
        if monument == tuple(getattr(state, 'monument', ()) or ()):
            return state
        return Field.State(cells=state.cells, self=state.self, monument=monument)

    def AcceptState(self, state: Any, *, publish: bool = True):
        if not isinstance(state, Field.State):
            raise TypeError('Dream.AcceptState expects Field.State')
        firstreal = self.state is None and bool(getattr(state, 'cells', ()) or ())
        self.state = state
        if firstreal and (not self.bootflare):
            self.bootflare = True
            flare = self.PurgeFlare()
            self.Forward(flare)
        if publish:
            self.Publish()
        return self.state

    def Publish(self):
        if self.state is None:
            return None
        citadel = self.WakeCitadel()
        citadel.State = self.state
        if self.box.ashfall is not None:
            try:
                citadel.Ashfall(self.box.ashfall)
            except Exception:
                pass
            self.box.ashfall = None
        return self.state

    def Scrub(self, state: Field.State | None = None) -> Field.State | None:
        body = self.state if state is None else state
        if body is None:
            return None
        return Field.Scrub(body)

    def RouteVault(self):
        glyph = self.box.vault
        if glyph is None:
            return self.state
        self.box.vault = None
        if self.state is None:
            if isinstance(glyph, Field.State):
                self.AcceptState(glyph, publish=True)
                return self.state
            raise TypeError('Dream.RouteVault expected Field.State during bootstrap')
        self.Mutate(glyph, source='vault')
        return self.state

    def RouteCrypt(self):
        glyph = self.box.crypt
        if glyph is None:
            return self.state
        self.box.crypt = None
        if self.state is None:
            if isinstance(glyph, Field.State):
                self.AcceptState(glyph, publish=True)
                return self.state
            raise TypeError('Dream.RouteCrypt expected Field.State during bootstrap')
        self.Mutate(glyph, source='crypt')
        return self.state

    def Ashfall(self, glyph: Field.SaltGlyph) -> Any:
        if self.state is None:
            return None
        viewer = str(self.state.self[1] or '').strip()
        sender = str(glyph.key or '').strip()
        if not viewer or not sender:
            return None
        rawtext = str(getattr(getattr(glyph, 'textbody', None), 'text', '') or '')
        action, body = AshSplit(rawtext)
        total = AshBroadcastTotal(glyph, viewer, sender, action)
        if viewer != sender and total <= 0:
            return None
        sendercell = Field.FindCell(self.state, sender)
        sendername = str(sendercell.soul or '') if sendercell is not None else str(self.state.self[0] or '')
        payload = {
            'sender': sendername or sender,
            'kind': SaltGlyph,
            'action': action,
            'text': body[:60],
            'rawtext': rawtext,
            'total': int(total),
        }
        self.box.ashfall = payload
        return payload

    def SelfKey(self) -> str:
        if self.state is None:
            return ''
        return str(self.state.self[1] or '').strip()

    def PurgeKey(self, glyph: Any) -> str:
        if isinstance(glyph, dict):
            return str(glyph.get('key', '') or '').strip()
        return str(getattr(glyph, 'key', '') or '').strip()

    def Pristine(self, state: Field.State | None = None) -> bool:
        body = self.state if state is None else state
        if body is None:
            return True
        return len(tuple(getattr(body, 'monument', ()) or ())) == 0

    def WithChainbit(self, cell: Field.Cell, chainbit: int) -> Field.Cell:
        cb = 1 if int(chainbit) else 0
        lb = 1 if int(cell.purge.lockbit) or cb else 0
        return replace(cell, purge=Field.Purge(chainbit=cb, lockbit=lb))

    def LatchOnly(self, cell: Field.Cell) -> Field.Cell:
        return replace(cell, purge=Field.Purge(chainbit=0, lockbit=1))

    def ClearPurge(self, cell: Field.Cell) -> Field.Cell:
        return replace(cell, purge=Field.Purge(chainbit=0, lockbit=0))

    def ActiveChain(self, cell: Field.Cell, chain: Field.Chain) -> int:
        child = str(getattr(cell.lock, 'child', '') or '').strip()
        return 1 if chain.linked and child not in ('', Field.ZeroHashHex) else 0

    def IdempotentSalt(self, current: Field.Cell | None, glyph: Field.SaltGlyph) -> bool:
        if current is None:
            return False
        sameparent = str(current.lock.parent or '') == str(glyph.lockbody.parent or '')
        samechild = str(current.lock.child or '') == str(glyph.lockbody.child or '')
        samesign = str(current.sign or '') == str(glyph.locksign or '')
        return bool(sameparent and samechild and samesign)

    def StampChainSalt(self, state: Field.State, glyph: Field.SaltGlyph, chains: tuple[Field.Chain, ...]) -> Field.State:
        cells = {cell.key: cell for cell in state.cells}
        signer = str(glyph.key or '').strip()
        if signer and signer in cells and (len(chains) >= 1):
            cells[signer] = self.WithChainbit(cells[signer], 1 if chains[0].linked else 0)
        seen = set()
        creditkeys = []
        for leg in glyph.saltbody:
            key = str(getattr(leg, 'key', '') or '').strip()
            if not key or key == signer or key in seen:
                continue
            seen.add(key)
            creditkeys.append(key)
        for index, key in enumerate(creditkeys, start=1):
            if key in cells and index < len(chains):
                cells[key] = self.WithChainbit(cells[key], 1) if chains[index].linked else self.WithChainbit(cells[key], 0)
        stamped = Field.State(cells=tuple((cells[cell.key] for cell in state.cells)), self=state.self, monument=state.monument)
        return stamped

    def StampChainDream(self, state: Field.State, chains: tuple[Field.Chain, ...]) -> Field.State:
        if len(chains) != len(state.cells):
            return state
        cells = []
        for cell, chain in zip(state.cells, chains):
            if chain.linked:
                active = self.ActiveChain(cell, chain)
                cells.append(self.WithChainbit(cell, 1) if active else self.LatchOnly(cell))
            else:
                cells.append(self.WithChainbit(cell, 0))
        stamped = Field.State(cells=tuple(cells), self=state.self, monument=state.monument)
        return stamped

    def ApplyPurgeKey(self, state: Field.State, key: str) -> Field.State:
        key = str(key or '').strip()
        if not key:
            return state
        target = Field.FindCell(state, key)
        if target is None:
            return state
        cleared = self.ClearPurge(target)
        nextstate = Field.ReplaceCell(state, cleared)
        return nextstate

    def ScrubPurge(self, state: Field.State) -> Field.State:
        cleared = tuple((replace(cell, purge=Field.Purge(chainbit=0, lockbit=0)) for cell in state.cells))
        out = Field.State(cells=cleared, self=state.self, monument=state.monument)
        return out

    def PurgeFlare(self) -> dict[str, Any]:
        return {'kind': PurgeGlyph, 'key': self.SelfKey()}

    def Mutate(self, glyph: Any, source: str = ''):
        kind = self.Kind(glyph)
        if kind == PurgeGlyph:
            mutated = self.MutatePurge(glyph, source=source)
        elif kind == DreamGlyph:
            mutated = self.MutateDream(glyph, source=source)
        else:
            mutated = self.MutateSalt(glyph, source=source)
        if mutated:
            self.changed = True
            if kind == SaltGlyph:
                self.Ashfall(glyph)
            echo = True
            if kind == PurgeGlyph:
                echo = False
            elif source == 'crypt' and kind == DreamGlyph:
                echo = False
            if echo:
                self.Forward(glyph)
        return mutated

    def DefectViable(self, state: Field.State, glyph: Field.SaltGlyph) -> bool:
        spend, victim = DefectParts(glyph)
        if victim is None:
            return False
        signerq = Seat(state, glyph.key)
        victimq = Seat(state, victim.key)
        if signerq is None or victimq is None:
            return False
        if signerq == victimq:
            return False
        if SameFile(signerq, victimq):
            return False
        signer = state.cells[signerq]
        target = state.cells[victimq]
        if int(target.salt) >= int(signer.salt):
            return False
        total = sum((int(leg.salt) for leg in spend))
        floor = 10000 if signerq % Field.SeatsPerFile == 0 else 1000
        if total != floor:
            return False
        if any((int(leg.salt) <= 0 for leg in spend)):
            return False
        filekeys = {cell.key for index, cell in enumerate(state.cells) if index != signerq and SameFile(index, signerq)}
        spendkeys = {str(leg.key or '').strip() for leg in spend}
        if spendkeys != filekeys:
            return False
        if int(victim.salt) != 0:
            return False
        return True

    def DefectSwap(self, state: Field.State, glyph: Field.SaltGlyph) -> Field.State:
        spend, victim = DefectParts(glyph)
        if victim is None:
            return state
        signerq = Seat(state, glyph.key)
        victimq = Seat(state, victim.key)
        if signerq is None or victimq is None:
            return state
        cells = list(state.cells)
        cells[signerq], cells[victimq] = (cells[victimq], cells[signerq])
        out = Field.State(cells=tuple(cells), self=state.self, monument=state.monument)
        return out

    def MutateSalt(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.SaltGlyph):
            raise TypeError('Dream.MutateSalt expects Field.SaltGlyph')
        defect = DefectShape(glyph)
        try:
            Field.VerifySalt(glyph)
            current = Field.FindCell(self.state, glyph.key)
            if defect and (not self.DefectViable(self.state, glyph)):
                return False
            if source == 'crypt' and self.IdempotentSalt(current, glyph):
                return False
            nextstate, chains = Field.MutateReceipt(self.state, glyph)
            if defect:
                nextstate = self.DefectSwap(nextstate, glyph)
            nextstate = self.UpdateMonument(nextstate, glyph)
        except Exception:
            return False
        changed = self.Commit(nextstate)
        if not changed or self.state is None:
            return False
        self.state = self.StampChainSalt(self.state, glyph, chains)
        return True

    def MutateDream(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.State):
            raise TypeError('Dream.MutateDream expects Field.State')
        try:
            nextstate, chains = Field.MutateState(self.state, glyph)
        except Exception:
            return False
        changed = self.Commit(nextstate)
        if not changed or self.state is None:
            return False
        self.state = self.StampChainDream(self.state, chains)
        return True

    def MutatePurge(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if source == 'vault':
            key = self.PurgeKey(glyph)
            if key and key == self.SelfKey():
                nextstate = self.ScrubPurge(self.state)
                changed = self.Commit(nextstate)
                flare = self.PurgeFlare()
                self.Forward(flare)
                return changed or True
            nextstate = self.ApplyPurgeKey(self.state, key)
            changed = self.Commit(nextstate)
            flare = self.PurgeFlare()
            self.Forward(flare)
            return changed or True
        if source == 'crypt':
            if self.Pristine(self.state):
                return False
            key = self.PurgeKey(glyph)
            if key and key == self.SelfKey():
                return False
            self.Forward(self.Scrub(self.state))
            return True
        return False

    def Commit(self, nextstate: Field.State) -> bool:
        nextstate = Field.Stasis(nextstate)
        if self.state is None:
            self.state = nextstate
            return True
        if nextstate == self.state:
            return False
        before = self.state.SaltTotal
        after = nextstate.SaltTotal
        self.state = nextstate
        return True

    def Forward(self, glyph: Any):
        self.glyph = glyph
        crypt = self.WakeCrypt()
        if crypt is None:
            return glyph
        packet = self.Packet(glyph)
        try:
            crypt.glyph = packet
        except Exception:
            pass
        if hasattr(crypt, 'EmitGlyph'):
            try:
                crypt.EmitGlyph(packet)
            except Exception:
                pass
        elif hasattr(crypt, 'emitGlyph'):
            try:
                crypt.emitGlyph(packet)
            except Exception:
                pass
        return packet

    def Packet(self, glyph: Any) -> Any:
        kind = self.Kind(glyph)
        if kind == DreamGlyph:
            return self.BoxDream(glyph)
        if kind == SaltGlyph:
            return self.BoxSalt(glyph)
        if isinstance(glyph, dict):
            return dict(glyph)
        if isinstance(glyph, str):
            return str(glyph)
        return glyph

    def Kind(self, glyph: Any) -> str:
        if isinstance(glyph, str):
            return PurgeGlyph
        if isinstance(glyph, Field.State):
            return DreamGlyph
        if isinstance(glyph, Field.SaltGlyph):
            return SaltGlyph
        if isinstance(glyph, dict):
            kind = str(glyph.get('kind', '') or '').strip().lower()
            if kind:
                return kind
            if 'cells' in glyph:
                return DreamGlyph
            if 'saltbody' in glyph and 'lockbody' in glyph and 'textbody' in glyph:
                return SaltGlyph
        return SaltGlyph

    def BoxSalt(self, glyph: Field.SaltGlyph) -> dict[str, Any]:
        return {
            'key': glyph.key,
            'saltbody': [self.BoxLeg(leg) for leg in glyph.saltbody],
            'lockbody': self.BoxLock(glyph.lockbody),
            'textbody': {'text': glyph.textbody.text},
            'salthash': glyph.salthash,
            'lockhash': glyph.lockhash,
            'texthash': glyph.texthash,
            'sign': glyph.sign,
            'locksign': glyph.locksign,
        }

    def BoxLeg(self, leg: Field.Salt) -> dict[str, Any]:
        return {'key': leg.key, 'salt': leg.salt}

    def BoxLock(self, lock: Field.Lock) -> dict[str, Any]:
        return {'parent': lock.parent, 'child': lock.child}

    def BoxPurge(self, purge: Field.Purge) -> dict[str, Any]:
        return {'chainbit': purge.chainbit, 'lockbit': purge.lockbit}

    def BoxCell(self, cell: Field.Cell) -> dict[str, Any]:
        return {
            'soul': cell.soul,
            'key': cell.key,
            'salt': cell.salt,
            'purge': self.BoxPurge(cell.purge),
            'lock': self.BoxLock(cell.lock),
            'sign': cell.sign,
        }

    def BoxDream(self, state: Field.State) -> dict[str, Any]:
        scrubbed = Field.Scrub(state)
        return {
            'kind': DreamGlyph,
            'self': [scrubbed.self[0], scrubbed.self[1]],
            'monument': list(scrubbed.monument),
            'cells': [self.BoxCell(cell) for cell in scrubbed.cells],
        }


dream = Dream()
