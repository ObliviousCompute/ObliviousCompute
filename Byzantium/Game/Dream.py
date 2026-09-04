from __future__ import annotations

from dataclasses import dataclass, replace
import threading
from typing import Any

import Citadel
import Crypt
import Field

PurgeGlyph = 'purge'
DreamGlyph = 'dream'
SaltGlyph = 'salt'
NightmareGlyph = 'nightmare'
WireDream = 0
WireNightmare = 1
WirePurge = 2
WireWhisper = 3
WireRally = 4
WireDefect = 5
WireWrath = 6
@dataclass(frozen=True)
class Ash:
    sender: str = ''
    text: str = ''
    total: int = 0


@dataclass(frozen=True)
class Surface:
    cells: tuple[Field.Cell, ...]
    self: tuple[str, str]
    pristine: int
    ash: Ash | None = None
    dreamfall: bool = False
    ashfall: bool = False


def AshTotal(glyph: Field.SaltGlyph, viewer: str, sender: str) -> int:
    legs = tuple(glyph.lockbody.payout or ())
    viewertag = Field.PlayerTag(viewer) if len(str(viewer or '').strip()) == Field.KeyHexLen else str(viewer or '').strip().lower()
    sendertag = str(sender or '').strip().lower()
    if viewertag == sendertag:
        return sum(int(getattr(leg, 'salt', 0) or 0) for leg in legs)
    direct = sum(int(getattr(leg, 'salt', 0) or 0) for leg in legs if str(leg.tag or '').strip().lower() == viewertag)
    if direct > 0:
        return direct
    if int(glyph.lockbody.kind) == Field.KindDefect:
        return max((int(getattr(leg, 'salt', 0) or 0) for leg in legs), default=0)
    return 0


def MutatePurge(cell: Field.Cell, *, chainbit: int | None = None, lockbit: int | None = None) -> Field.Cell:
    Field.VerifyCell(cell)
    cb = cell.purge.chainbit if chainbit is None else int(chainbit)
    lb = cell.purge.lockbit if lockbit is None else int(lockbit)
    return replace(cell, purge=Field.PurgeLocks(chainbit=cb, lockbit=lb))


def OpenCell(cell: Field.Cell) -> Field.Cell:
    return MutatePurge(cell, chainbit=0)


def CloseCell(cell: Field.Cell) -> Field.Cell:
    return MutatePurge(cell, chainbit=1)


def Link(current: Field.Cell, candidate: Field.Cell) -> Field.Chain:
    Field.VerifyCell(current)
    Field.VerifyCell(candidate)
    if current == candidate:
        return Field.Chain(linked=True, relation='Link')
    if candidate.key != current.key:
        return Field.Chain(linked=False, relation='reject', open=True, reason='key changed')
    if Field.Burned(current):
        if int(candidate.salt) != 0:
            return Field.Chain(linked=False, relation='reject', open=True, reason='burned actor resurrection')
        if candidate.lowlock is None:
            return Field.Chain(linked=False, relation='reject', open=True, reason='burned actor lost competing pair')
        currentlocks = Field.LockSet(current)
        candidatelocks = Field.LockSet(candidate)
        if currentlocks == candidatelocks:
            return Field.Chain(linked=True, relation='Link', reason='burned pair')
        try:
            canonical = Field.CanonicalLocks(*(currentlocks + candidatelocks))
        except Exception:
            canonical = tuple()
        if canonical == candidatelocks:
            return Field.Chain(linked=True, relation='Link', reason='lower burned pair')
        return Field.Chain(linked=False, relation='reject', open=True, reason='burned pair mismatch')
    if candidate.lock.parent == Field.ContinuationChild(current):
        if candidate.salt < current.salt and candidate.lock.sign == Field.NullSignHex:
            return Field.Chain(linked=False, relation='reject', open=True, reason='debit without sign')
        return Field.Chain(linked=True, relation='Link')
    if Field.LockSet(current) == Field.LockSet(candidate):
        return Field.Chain(linked=True, relation='Link')
    return Field.Chain(linked=False, relation='reject', open=True, reason='no Link')


def BurnRecipients(state: Field.State, *locks: Field.Lock) -> set[str]:
    keys: set[str] = set()
    for lock in locks:
        for leg in lock.payout:
            if int(leg.salt) <= 0:
                continue
            target = Field.FindCell(state, leg.tag)
            if target is not None:
                keys.add(target.key)
    return keys


def ReconcileCellLocks(
    working: Field.State,
    candidate: Field.Cell,
) -> tuple[Field.State, set[str], set[str], set[str]]:
    current = Field.FindCell(working, candidate.key)
    if current is None or current.purge.lockbit == 0:
        return (working, set(), set(), set())
    incoming = Field.LockSet(candidate)
    currentlocks = Field.LockSet(current)
    if not incoming or incoming == currentlocks:
        return (working, set(), set(), set())

    # A complete zero-Salt Dream may carry a lower burned settlement. The Dream
    # supplies that whole settlement; do not locally re-burn the exhausted actor.
    if Field.Burned(current):
        if int(candidate.salt) != 0 or candidate.lowlock is None:
            return (working, set(), set(), set())
        try:
            canonical = Field.CanonicalLocks(*(currentlocks + incoming))
        except Exception:
            return (working, set(), set(), set())
        if canonical != incoming:
            return (working, set(), set(), set())
        settlement = BurnRecipients(working, *(currentlocks + incoming))
        settlement.discard(current.key)
        return (working, set(), {current.key}, settlement)

    evidence: Field.Lock | tuple[Field.Lock, ...]
    evidence = incoming[0] if len(incoming) == 1 else incoming
    try:
        repaired, _chains = Field.Adopt(working, evidence)
    except Exception:
        return (working, set(), set(), set())
    if repaired == working:
        return (working, set(), set(), set())
    before = {cell.key: cell for cell in working.cells}
    beforepos = {cell.key: index for index, cell in enumerate(working.cells)}
    afterpos = {cell.key: index for index, cell in enumerate(repaired.cells)}
    changed = {
        cell.key for cell in repaired.cells
        if before.get(cell.key) != cell or beforepos.get(cell.key) != afterpos.get(cell.key)
    }
    refreshed = Field.FindCell(repaired, candidate.key)
    if refreshed is not None and refreshed.lowlock is not None:
        changed.update(BurnRecipients(working, *(currentlocks + incoming + Field.LockSet(refreshed))))
    return (repaired, changed, set(), set())


def Assimilate(local: Field.State, incoming: Field.State) -> tuple[Field.State, tuple[Field.Chain, ...]]:
    Field.VerifyState(local)
    keys = Field.FindKeys(local)
    Field.VerifyDream(incoming, expectedkeys=keys)
    scrubbed = Field.Scrub(incoming)

    # A closed burned surface cannot be resurrected by a whole Dream. Lower
    # competing siblings may improve its tombstone, but Salt stays zero and the
    # fork stays on the same parent until that local PurgeLock is cleared.
    incomingmap = {cell.key: cell for cell in scrubbed.cells}
    for current in local.cells:
        if not Field.Burned(current) or current.purge.lockbit == 0:
            continue
        candidate = incomingmap.get(current.key)
        if candidate is None:
            raise ValueError('burned actor missing from incoming Dream')
        if int(candidate.salt) != 0:
            raise ValueError('burned actor cannot be resurrected while PurgeLocked')
        if candidate.lowlock is None:
            raise ValueError('burned actor must retain competing children while PurgeLocked')
        if candidate.lock.parent != current.lock.parent:
            raise ValueError('burned actor parent cannot change while PurgeLocked')

    # Reconcile signed child evidence on one complete working projection first.
    # Revoke/Trace/Adopt may touch arbitrary cells, so every touched cell remains
    # anchored to that coherent repair for the rest of this assimilation pass.
    working = local
    protected: set[str] = set()
    burnedpairs: set[str] = set()
    burnrecipients: set[str] = set()
    for candidate in scrubbed.cells:
        working, changed, pairkeys, recipientkeys = ReconcileCellLocks(working, candidate)
        protected.update(changed)
        burnedpairs.update(pairkeys)
        burnrecipients.update(recipientkeys)

    selected: dict[str, Field.Cell] = {}
    chainmap: dict[str, Field.Chain] = {}
    for candidate in scrubbed.cells:
        current = Field.FindCell(working, candidate.key)
        if current is None:
            raise ValueError('incoming state key missing from local state')
        if current.key in protected:
            selected[current.key] = current
            chainmap[current.key] = Field.Chain(linked=True, relation='Link', open=False, reason='reconciled')
            continue
        if current.key in burnedpairs:
            selected[current.key] = CloseCell(candidate)
            chainmap[current.key] = Field.Chain(linked=True, relation='Link', open=False, reason='burned Dream pair')
            continue
        if current.key in burnrecipients and Field.LockSet(current) == Field.LockSet(candidate):
            if Field.Burned(current) and int(candidate.salt) != 0:
                selected[current.key] = OpenCell(current)
                chainmap[current.key] = Field.Chain(linked=False, relation='reject', open=True, reason='burned recipient resurrection')
            else:
                selected[current.key] = CloseCell(candidate)
                chainmap[current.key] = Field.Chain(linked=True, relation='Link', open=False, reason='burn settlement')
            continue
        if current.purge.lockbit == 0:
            selected[current.key] = CloseCell(candidate)
            chainmap[current.key] = Field.Chain(linked=True, relation='Link', open=False, reason='open')
            continue
        outcome = Link(current, candidate)
        chainmap[current.key] = outcome
        selected[current.key] = CloseCell(candidate) if outcome.linked else OpenCell(current)

    order = working.cells if working != local else scrubbed.cells
    cells = tuple(selected[cell.key] for cell in order)
    chains = tuple(chainmap[cell.key] for cell in order)
    nextstate = Field.State(
        cells=cells,
        self=local.self,
        pristine=incoming.pristine,
    )
    Field.VerifyState(nextstate, expectedkeys=keys)
    return (nextstate, chains)


@dataclass
class Box:
    vault: Any = None
    crypt: Any = None


class Dream:

    def __init__(self, citadel: Any = None, crypt: Any = None):
        self.box = Box()
        self.state: Field.State | None = None
        self.citadel = citadel
        self.crypt = crypt
        self.glyph: Any = None
        self.ash: Ash | None = None
        self.dreamfall = False
        self.ashfall = False
        self.bootflare = False
        self.Sleepwalk = threading.Lock()
        self.Dreaming = False
        self.Snooze = False

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
        self.crypt = Crypt.Crypt(state=state, dream=self)
        Crypt.crypt = self.crypt
        return self.crypt

    def Sleep(self):
        crypt = self.WakeCrypt()
        if crypt is None:
            return None
        return crypt.Sleep()

    def Empty(self) -> bool:
        vaultempty = self.box.vault is None
        cryptlane = self.box.crypt
        cryptempty = cryptlane is None or (isinstance(cryptlane, list) and len(cryptlane) == 0)
        return vaultempty and cryptempty

    def Wake(self):
        with self.Sleepwalk:
            if self.Dreaming:
                self.Snooze = True
                return self.state
            self.Dreaming = True
            self.Snooze = False

        try:
            while True:
                self.RouteVault()
                if self.dreamfall:
                    self.Publish()
                self.RouteCrypt()
                if self.dreamfall:
                    self.Publish()

                with self.Sleepwalk:
                    more = self.Snooze or not self.Empty()
                    self.Snooze = False
                    if not more:
                        self.Dreaming = False
                        break
        except Exception:
            with self.Sleepwalk:
                self.Dreaming = False
            raise
        finally:
            crypt = self.WakeCrypt()
            if crypt is not None:
                try:
                    crypt.Wake()
                except Exception:
                    pass
        return self.state

    def Awake(self):
        return self.Wake()

    def Route(self):
        return self.Wake()

    def AcceptState(self, state: Any, *, publish: bool = True):
        if not isinstance(state, Field.State):
            raise TypeError('Dream.AcceptState expects Field.State')
        firstreal = self.state is None and bool(getattr(state, 'cells', ()) or ())
        self.state = state
        self.dreamfall = True
        self.ashfall = False
        if firstreal and (not self.bootflare):
            self.bootflare = True
            flare = self.PurgeFlare()
            self.Forward(flare)
        if publish:
            self.Publish()
        return self.state

    def Publish(self):
        if self.state is None or not self.dreamfall:
            return self.state
        surface = Surface(
            cells=self.state.cells,
            self=self.state.self,
            pristine=self.state.pristine,
            ash=self.ash,
            dreamfall=True,
            ashfall=bool(self.ashfall),
        )
        citadel = self.WakeCitadel()
        citadel.State = surface
        self.dreamfall = False
        self.ashfall = False
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
                self.AcceptState(glyph, publish=False)
                return self.state
            raise TypeError('Dream.RouteVault expected Field.State during bootstrap')
        self.Mutate(glyph, source='vault')
        return self.state

    def RouteCrypt(self):
        lane = self.box.crypt
        if lane is None:
            return self.state
        if isinstance(lane, list):
            if len(lane) == 0:
                self.box.crypt = None
                return self.state
            glyph = lane.pop(0)
            if len(lane) == 0:
                self.box.crypt = None
        else:
            glyph = lane
            self.box.crypt = None
        if self.state is None:
            if isinstance(glyph, Field.State):
                if int(getattr(glyph, 'pristine', 1) or 0) != 0:
                    glyph = replace(glyph, pristine=0)
                self.AcceptState(glyph, publish=False)
                return self.state
            raise TypeError('Dream.RouteCrypt expected Field.State during bootstrap')
        self.Mutate(glyph, source='crypt')
        return self.state

    def SetAsh(self, glyph: Field.SaltGlyph) -> Ash | None:
        if self.state is None:
            return None
        viewer = str(self.state.self[1] or '').strip()
        sender = str(glyph.lockbody.tag or '').strip()
        rawtext = str(getattr(getattr(glyph, 'textbody', None), 'text', '') or '')
        sendercell = Field.FindCell(self.state, sender) if sender else None
        sendername = str(sendercell.soul or '') if sendercell is not None else sender
        total = AshTotal(glyph, viewer, sender)
        self.ash = Ash(sender=sendername or sender, text=rawtext, total=int(total))
        self.ashfall = True
        return self.ash

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
        return int(getattr(body, 'pristine', 1) or 0) == 1

    def WithChainbit(self, cell: Field.Cell, chainbit: int) -> Field.Cell:
        cb = 1 if int(chainbit) else 0
        return replace(cell, purge=Field.PurgeLocks(chainbit=cb, lockbit=1))

    def LatchOnly(self, cell: Field.Cell) -> Field.Cell:
        return replace(cell, purge=Field.PurgeLocks(chainbit=0, lockbit=1))

    def ClearPurge(self, cell: Field.Cell) -> Field.Cell:
        return replace(cell, purge=Field.PurgeLocks(chainbit=0, lockbit=0))

    def StampKeys(self, state: Field.State, keys: set[str], chainbit: int) -> Field.State:
        wanted = {str(key or '').strip() for key in keys if str(key or '').strip()}
        cells = tuple(self.WithChainbit(cell, chainbit) if cell.key in wanted else cell for cell in state.cells)
        return Field.State(cells=cells, self=state.self, pristine=state.pristine)

    def SaltFootprint(self, state: Field.State, glyph: Field.SaltGlyph) -> set[str]:
        keys: set[str] = set()
        signer = Field.FindCell(state, str(glyph.lockbody.tag or '').strip())
        if signer is not None:
            keys.add(signer.key)
        for leg in glyph.lockbody.payout:
            target = Field.FindCell(state, str(leg.tag or '').strip())
            if target is not None:
                keys.add(target.key)
        return keys

    def MutationKeys(self, before: Field.State, after: Field.State) -> set[str]:
        beforemap = {cell.key: cell for cell in before.cells}
        beforepos = {cell.key: i for i, cell in enumerate(before.cells)}
        afterpos = {cell.key: i for i, cell in enumerate(after.cells)}
        changed: set[str] = set()
        for cell in after.cells:
            old = beforemap.get(cell.key)
            if old is None:
                changed.add(cell.key)
                continue
            oldbody = replace(old, purge=Field.PurgeLocks(chainbit=0, lockbit=0))
            newbody = replace(cell, purge=Field.PurgeLocks(chainbit=0, lockbit=0))
            if oldbody != newbody or beforepos.get(cell.key) != afterpos.get(cell.key):
                changed.add(cell.key)
        return changed

    def StampChainSalt(self, state: Field.State, glyph: Field.SaltGlyph, chainbit: int, extra: set[str] | None = None) -> Field.State:
        keys = self.SaltFootprint(state, glyph)
        keys.update(extra or set())
        return self.StampKeys(state, keys, chainbit)

    def StampChainDream(self, state: Field.State, chains: tuple[Field.Chain, ...]) -> Field.State:
        if len(chains) != len(state.cells):
            return state
        cells = tuple(self.WithChainbit(cell, 1 if chain.linked else 0) for cell, chain in zip(state.cells, chains))
        return Field.State(cells=cells, self=state.self, pristine=state.pristine)

    def StampAll(self, state: Field.State, chainbit: int) -> Field.State:
        return self.StampKeys(state, {cell.key for cell in state.cells}, chainbit)

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

    def PurgeFlare(self) -> dict[str, Any]:
        return {'kind': PurgeGlyph, 'key': self.SelfKey()}

    def Mutate(self, glyph: Any, source: str = ''):
        kind = self.Kind(glyph)
        before = self.state
        if kind == PurgeGlyph:
            mutated = self.MutatePurge(glyph, source=source)
        elif kind == DreamGlyph:
            mutated = self.MutateDream(glyph, source=source)
        elif kind == NightmareGlyph:
            mutated = self.MutateNightmare(glyph, source=source)
        else:
            mutated = self.MutateSalt(glyph, source=source)
        if mutated:
            self.dreamfall = True
            if kind == SaltGlyph:
                self.SetAsh(glyph)
            if kind == PurgeGlyph:
                return mutated
            if source == 'crypt' and kind == DreamGlyph:
                return mutated
            if kind in (SaltGlyph, NightmareGlyph) and self.state is not None:
                tag = glyph.lockbody.tag if kind == SaltGlyph else glyph.lowlock.tag
                current = Field.FindCell(self.state, tag)
                previous = Field.FindCell(before, tag) if before is not None else None
                if current is not None and current.lowlock is not None and (previous is None or Field.LockSet(previous) != Field.LockSet(current)):
                    self.Forward(Field.NightmareGlyph(lowlock=current.lowlock, lock=current.lock))
                    return mutated
            self.Forward(glyph)
        return mutated


    def MutateSalt(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.SaltGlyph):
            raise TypeError('Dream.MutateSalt expects Field.SaltGlyph')
        before = self.state
        verified = False
        try:
            Field.VerifySalt(glyph, before)
            verified = True
            nextstate, _chains = Field.Adopt(before, glyph.lockbody)
            if nextstate == before:
                observed = self.StampChainSalt(before, glyph, 0)
                if observed != before:
                    self.state = observed
                    self.dreamfall = True
                return False
        except Exception:
            if verified:
                observed = self.StampChainSalt(before, glyph, 0)
                if observed != before:
                    self.state = observed
                    self.dreamfall = True
            return False
        changed = self.Commit(nextstate)
        if not changed or self.state is None:
            return False
        touched = self.MutationKeys(before, self.state)
        self.state = self.StampChainSalt(self.state, glyph, 1, touched)
        return True

    def MutateNightmare(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.NightmareGlyph):
            raise TypeError('Dream.MutateNightmare expects NightmareGlyph')
        before = self.state
        verified = False
        signer = Field.FindCell(before, glyph.lowlock.tag)
        footprint = set() if signer is None else {signer.key}
        try:
            Field.VerifyNightmare(glyph, before)
            verified = True
            nextstate, _chains = Field.Adopt(before, (glyph.lowlock, glyph.lock))
        except Exception:
            if verified and footprint:
                observed = self.StampKeys(before, footprint, 0)
                if observed != before:
                    self.state = observed
                    self.dreamfall = True
            return False
        if nextstate == before:
            if footprint:
                observed = self.StampKeys(before, footprint, 0)
                if observed != before:
                    self.state = observed
                    self.dreamfall = True
            return False
        changed = self.Commit(nextstate)
        if not changed or self.state is None:
            return False
        touched = self.MutationKeys(before, self.state) | footprint
        self.state = self.StampKeys(self.state, touched, 1)
        return True

    def MutateDream(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.State):
            raise TypeError('Dream.MutateDream expects Field.State')
        before = self.state
        try:
            nextstate, chains = Assimilate(before, glyph)
        except Exception:
            observed = self.StampAll(before, 0)
            if observed != before:
                self.state = observed
                self.dreamfall = True
            return False
        changed = self.Commit(nextstate)
        if self.state is None:
            return False
        stamped = self.StampChainDream(self.state, chains)
        if stamped != self.state:
            self.state = stamped
            self.dreamfall = True
        return bool(changed)

    def MutatePurge(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if source == 'vault':
            key = self.PurgeKey(glyph)
            if key and key == self.SelfKey():
                nextstate = Field.Purge(self.state)
                changed = self.Commit(nextstate)
                flare = self.PurgeFlare()
                self.Forward(flare)
                return changed
            nextstate = self.ApplyPurgeKey(self.state, key)
            changed = self.Commit(nextstate)
            flare = self.PurgeFlare()
            self.Forward(flare)
            return changed
        if source == 'crypt':
            if self.Pristine(self.state):
                return False
            key = self.PurgeKey(glyph)
            if key and key == self.SelfKey():
                return False
            if key:
                target = Field.FindCell(self.state, key)
                if target is not None and Field.Burned(target):
                    return False
            self.Forward(self.Scrub(self.state))
            return False
        return False

    def Commit(self, nextstate: Field.State) -> bool:
        nextstate = Field.Stasis(nextstate)
        if self.state is None:
            self.state = nextstate
            return True
        if nextstate == self.state:
            return False
        if int(getattr(nextstate, 'pristine', 1) or 0) != 0:
            nextstate = replace(nextstate, pristine=0)
        self.state = nextstate
        return True

    def Forward(self, glyph: Any):
        self.glyph = glyph
        crypt = self.WakeCrypt()
        if crypt is None:
            return glyph
        try:
            crypt.glyph = glyph
        except Exception:
            pass
        try:
            crypt.EmitGlyph(glyph)
        except Exception:
            pass
        return glyph

    def Kind(self, glyph: Any) -> str:
        if isinstance(glyph, str):
            return PurgeGlyph
        if isinstance(glyph, Field.State):
            return DreamGlyph
        if isinstance(glyph, Field.SaltGlyph):
            return SaltGlyph
        if isinstance(glyph, Field.NightmareGlyph):
            return NightmareGlyph
        if isinstance(glyph, dict):
            rawkind = glyph.get('kind', '')
            if isinstance(rawkind, int) and not isinstance(rawkind, bool):
                if rawkind == WireDream:
                    return DreamGlyph
                if rawkind == WireNightmare:
                    return NightmareGlyph
                if rawkind == WirePurge:
                    return PurgeGlyph
                if rawkind in (WireWhisper, WireRally, WireDefect, WireWrath):
                    return SaltGlyph
            kind = str(rawkind or '').strip().lower()
            if kind:
                return kind
            if 'cells' in glyph:
                return DreamGlyph
            if ('lock' in glyph or 'lockbody' in glyph) and 'textbody' in glyph:
                return SaltGlyph
        return SaltGlyph


dream = Dream()
