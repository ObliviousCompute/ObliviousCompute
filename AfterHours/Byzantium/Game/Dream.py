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
    # Locked cells are never replaced from a complete Dream merely because the
    # remote frontier is cryptographic kin. Ordinary continuations are computed
    # through Reconcile first; by the time Assemble reaches an unprotected
    # locked cell, the incoming economic coordinate must already be reproduced
    # locally. Purge metadata is deliberately ignored here because Seal owns it.
    if Field.LockSet(current) == Field.LockSet(candidate):
        if int(candidate.salt) != int(current.salt):
            return Field.Chain(linked=False, relation='reject', open=True, reason='unexplained Salt')
        return Field.Chain(linked=True, relation='Link', reason='reproduced')
    return Field.Chain(linked=False, relation='reject', open=True, reason='unreproduced frontier')


Kindred = Field.Kindred

def Seal(state: Field.State) -> Field.State:
    return Field.State(
        cells=tuple(
            replace(cell, purge=Field.PurgeLocks(chainbit=1, lockbit=1))
            for cell in state.cells
        ),
        self=state.self,
        pristine=state.pristine,
    )


class Underworld:
    """Scratch surface used only while an equivocation is being resolved."""

    def __init__(self, state: Field.State) -> None:
        Field.VerifyState(state)
        self.template = state
        self.cells = {cell.key: cell for cell in state.cells}
        self.tags = {Field.PlayerTag(cell.key): cell.key for cell in state.cells}
        self.order = [cell.key for cell in state.cells]
        self.balances = {cell.key: int(cell.salt) for cell in state.cells}
        self.touched: set[str] = set()

    def Key(self, identity: str) -> str:
        token = str(identity or '').strip().lower()
        if token in self.cells:
            return token
        key = self.tags.get(token)
        if key is None:
            raise ValueError('Underworld identity not found')
        return key

    def Swap(self, first: str, second: str) -> None:
        a = self.Key(first)
        b = self.Key(second)
        ia = self.order.index(a)
        ib = self.order.index(b)
        self.order[ia], self.order[ib] = self.order[ib], self.order[ia]
        self.touched.update((a, b))

    def Dissolve(self, lock: Field.Lock) -> None:
        signer = self.Key(lock.tag)
        self.touched.add(signer)
        self.balances[signer] += Field.LockTotal(lock)
        for leg in lock.payout:
            target = self.Key(leg.tag)
            self.touched.add(target)
            self.balances[target] -= int(leg.salt)
        if lock.kind == Field.KindDefect:
            _spend, victim = Field.DefectParts(lock)
            if victim is None:
                raise ValueError('defect requires swap victim')
            self.Swap(lock.tag, victim.tag)

    def Apply(self, lock: Field.Lock) -> None:
        signer = self.Key(lock.tag)
        self.touched.add(signer)
        self.balances[signer] -= Field.LockTotal(lock)
        for leg in lock.payout:
            target = self.Key(leg.tag)
            self.touched.add(target)
            self.balances[target] += int(leg.salt)
        if lock.kind == Field.KindDefect:
            _spend, victim = Field.DefectParts(lock)
            if victim is None:
                raise ValueError('defect requires swap victim')
            self.Swap(lock.tag, victim.tag)

    def SetFrontier(self, identity: str, locks: tuple[Field.Lock, ...]) -> None:
        key = self.Key(identity)
        self.touched.add(key)
        cell = self.cells[key]
        if len(locks) == 1:
            self.cells[key] = replace(cell, lowlock=None, lock=locks[0])
            return
        if len(locks) == 2:
            low, high = Field.CanonicalLocks(*locks)
            self.cells[key] = replace(cell, lowlock=low, lock=high)
            return
        raise ValueError('frontier must contain one or two receipts')

    def State(self, *, pristine: int | None = None) -> Field.State:
        if any(int(value) < 0 for value in self.balances.values()):
            raise ValueError('negative Salt cannot crystallize')
        cells = tuple(
            replace(self.cells[key], salt=int(self.balances[key]))
            for key in self.order
        )
        state = Field.State(
            cells=cells,
            self=self.template.self,
            pristine=self.template.pristine if pristine is None else int(pristine),
        )
        Field.VerifyState(state, expectedkeys=Field.FindKeys(self.template))
        return state


class Trance:
    """Atomic equivocation treatment. Nothing leaves Trance before crystallization."""

    def __init__(self, before: Field.State, incoming: Field.State | None = None) -> None:
        Field.VerifyState(before)
        self.before = before
        self.incoming = incoming
        self.frontier: dict[str, tuple[Field.Lock, Field.Lock]] = {}
        self.reburn: set[str] = {
            cell.key for cell in before.cells if Field.Burned(cell)
        }
        self.burn: set[str] = set()
        self.damned: set[str] = set()
        self.zeroes: set[str] = set()
        self.touched: set[str] = set()

    def Add(self, *locks: Field.Lock) -> bool:
        pair = Field.Pair(*locks)
        if pair is None:
            return False
        signer = Field.FindCell(self.before, pair[0].tag)
        if signer is None:
            return False
        existing = self.frontier.get(signer.key)
        if existing is not None:
            pair = Field.Pair(*(existing + pair)) or existing
        current = Field.LockSet(signer)
        if len(current) == 2:
            try:
                currentpair = Field.CanonicalLocks(*current)
            except Exception:
                currentpair = tuple()
            if len(currentpair) == 2 and (currentpair[0].child, currentpair[1].child) <= (pair[0].child, pair[1].child):
                pair = (currentpair[0], currentpair[1])
        self.frontier[signer.key] = pair
        return True

    def Scry(self) -> 'Trance':
        if self.incoming is None:
            return self
        incoming = Field.Scrub(self.incoming)
        incomingmap = {cell.key: cell for cell in incoming.cells}
        for mine in self.before.cells:
            theirs = incomingmap.get(mine.key)
            if theirs is None:
                continue
            currentlocks = Field.LockSet(mine)
            incominglocks = Field.LockSet(theirs)
            if currentlocks == incominglocks:
                continue
            pair = Field.Pair(*(currentlocks + incominglocks))
            if pair is not None:
                self.Add(*pair)
                if int(mine.salt) > 0 and int(theirs.salt) == 0:
                    self.zeroes.add(mine.key)
        return self

    @staticmethod
    def Shares(total: int, keys: set[str]) -> dict[str, int]:
        total = int(total)
        if total < 0:
            raise ValueError('delta cannot be negative')
        ordered = sorted(set(keys))
        if not ordered:
            if total:
                raise ValueError('delta has no Damned recipients')
            return {}
        q, r = divmod(total, len(ordered))
        return {
            key: q + (1 if index < r else 0)
            for index, key in enumerate(ordered)
        }

    def FinalFrontier(self, key: str) -> tuple[Field.Lock, ...]:
        if key in self.frontier:
            return self.frontier[key]
        return Field.LockSet(self.before.cells[Field.CellIndex(self.before, key)])

    def Damned(self, burn: set[str]) -> set[str]:
        # Damned is receipt-derived from this Burn only. PurgeLocks never seed it.
        damned: set[str] = set()
        for key in burn:
            pair = self.FinalFrontier(key)
            if len(pair) != 2:
                continue
            damned.update(Field.Recipients(self.before, *pair))
        return damned

    def Expand(self, under: Underworld, burn: set[str]) -> dict[str, set[str]]:
        surfaces = {key: self.Damned({key}) for key in burn}
        absorbed: set[str] = set()
        while True:
            negatives = {key for key, value in under.balances.items() if int(value) < 0}
            if not negatives:
                return surfaces
            key = min(negatives)
            owners = {dog for dog, surface in surfaces.items() if key in surface}
            if not owners or key in self.reburn or key in burn or key in absorbed:
                raise ValueError('Damned surface cannot recover negative Salt')
            locks = self.FinalFrontier(key)
            if not locks:
                raise ValueError('negative Damned cell has no present frontier')
            for lock in locks:
                under.Dissolve(lock)
            targets = Field.Recipients(self.before, *locks) | {key}
            for dog in owners:
                surfaces[dog].update(targets)
            absorbed.add(key)

    def Weigh(self, zeroes: set[str] | None = None) -> Field.State:
        under = Underworld(self.before)
        burn = set(zeroes or ()) - self.reburn
        heretics: set[str] = set()

        for key in sorted(self.frontier):
            if key in self.reburn or key in burn:
                continue
            pair = self.frontier[key]
            current = self.before.cells[Field.CellIndex(self.before, key)]
            old = Field.LockSet(current)
            sameparent = tuple(lock for lock in old if lock.parent == pair[0].parent)
            estate = int(current.salt) + sum(Field.LockTotal(lock) for lock in sameparent)
            claims = sum(Field.LockTotal(lock) for lock in pair)
            if claims > estate:
                burn.add(key)
            else:
                heretics.add(key)

        # Burn closure is global over the signed sibling pairs already present,
        # not only pairs discovered in this Trance. Evidence in the Dream remains live.
        closure = set(heretics) | {
            cell.key for cell in self.before.cells
            if not Field.Burned(cell) and Field.Pair(*Field.LockSet(cell)) is not None
        }
        changed = True
        while changed:
            changed = False
            for key in sorted(closure - self.reburn - burn):
                if Field.Recipients(self.before, *self.FinalFrontier(key)) & (self.reburn | burn):
                    heretics.discard(key)
                    burn.add(key)
                    changed = True
                    break

        # Heretic: dissolve the old frontier, then apply both canonical children
        # on one Underworld surface. Evidence survives; only its old effect dissolves.
        for key in sorted(heretics):
            current = self.before.cells[Field.CellIndex(self.before, key)]
            pair = self.frontier[key]
            for old in Field.LockSet(current):
                if old.parent == pair[0].parent:
                    under.Dissolve(old)
            if under.balances[key] < sum(Field.LockTotal(lock) for lock in pair):
                raise ValueError('Heretic became insolvent during Weigh')
            for lock in reversed(pair):
                under.Apply(lock)
            under.SetFrontier(key, pair)

        # ReBurn was already settled at zero and contributes no new delta.
        # Burn is created in this Trance: dissolve its old effect and free its
        # recoverable estate exactly once.
        for key in sorted(burn):
            current = self.before.cells[Field.CellIndex(self.before, key)]
            for old in Field.LockSet(current):
                under.Dissolve(old)
        for key in sorted(self.reburn | burn):
            pair = self.frontier.get(key)
            if pair is not None:
                under.SetFrontier(key, pair)

        surfaces = self.Expand(under, burn)
        estates = {key: int(under.balances[key]) for key in burn}
        if any(value < 0 for value in estates.values()):
            raise ValueError('Burn has negative estate')
        for key in burn:
            under.balances[key] = 0
        for dog in sorted(burn):
            eligible = surfaces[dog] - (self.reburn | burn)
            for key, share in self.Shares(estates[dog], eligible).items():
                under.balances[key] += int(share)

        damned = set().union(*surfaces.values()) if surfaces else set()
        candidate = under.State(
            pristine=self.incoming.pristine if self.incoming is not None else self.before.pristine
        )
        self.touched = set(under.touched)
        self.burn = burn
        self.damned = damned
        return candidate

    def Crystallize(self, zeroes: set[str] | None = None) -> Field.State:
        candidate = self.Weigh(zeroes)
        Field.VerifyState(candidate, expectedkeys=Field.FindKeys(self.before))
        return candidate


def Reconcile(local: Field.State, evidence: Field.Lock | tuple[Field.Lock, ...]) -> tuple[Field.State, tuple[Field.Chain, ...]]:
    Field.VerifyState(local)
    incoming = (evidence,) if isinstance(evidence, Field.Lock) else tuple(evidence)
    if not incoming:
        raise ValueError('Reconcile requires receipt evidence')
    first = incoming[0]
    signer = Field.FindCell(local, first.tag)
    if signer is None:
        raise ValueError('receipt signer tag not found in Dream')
    for lock in incoming:
        Field.VerifyLock(signer.key, lock)
        if lock.tag != first.tag:
            raise ValueError('receipt evidence must share one signer')

    if signer.purge.lockbit == 0:
        return (local, (Field.Chain(linked=False, relation='reject', open=True, reason='PurgeLock released'),))

    current = Field.LockSet(signer)
    currentchildren = {lock.child for lock in current}
    if len(incoming) == 1 and first.child in currentchildren:
        return (local, (Field.Chain(linked=True, relation='Link', reason='idempotent'),))

    if not Field.Burned(signer) and len(incoming) == 1 and first.parent == Field.ContinuationChild(signer):
        if Field.LockTotal(first) > signer.salt:
            return (local, (Field.Chain(linked=False, relation='reject', open=True, reason='insolvent continuation'),))
        candidate, chains = Field.ApplyEffect(local, first)
        candidate = Field.SetLockSet(candidate, first.tag, (first,))
        return (candidate, chains)

    pair = Field.Pair(*(current + incoming))
    if pair is None:
        return (local, (Field.Chain(linked=False, relation='reject', open=True, reason='no admissible relation'),))
    if len(current) == 2 and tuple(Field.CanonicalLocks(*current)) == tuple(pair):
        return (local, (Field.Chain(linked=True, relation='Link', reason='idempotent nightmare'),))

    trance = Trance(local)
    trance.Add(*pair)
    candidate = trance.Crystallize()
    return (candidate, (Field.Chain(linked=True, relation='Link', reason='Trance'),))


def ReconcileDream(local: Field.State, incoming: Field.State) -> Field.State:
    Field.VerifyState(local)
    keys = Field.FindKeys(local)
    Field.VerifyDream(incoming, expectedkeys=keys)
    scrubbed = Field.Scrub(incoming)

    # First absorb ordinary one-hop continuations from still-trusted signers.
    # Retry because one receipt may fund another signer before its receipt fits.
    working = local
    pending = {cell.key: cell for cell in scrubbed.cells}
    while pending:
        progress = False
        for key, candidate in tuple(pending.items()):
            current = Field.FindCell(working, key)
            if current is None or current.purge.lockbit == 0 or Field.Burned(current):
                pending.pop(key, None)
                continue
            theirs = Field.LockSet(candidate)
            if len(theirs) != 1:
                pending.pop(key, None)
                continue
            lock = theirs[0]
            if lock.child in {item.child for item in Field.LockSet(current)}:
                pending.pop(key, None)
                continue
            if lock.parent != Field.ContinuationChild(current):
                pending.pop(key, None)
                continue
            try:
                nextstate, _chains = Reconcile(working, lock)
            except Exception:
                nextstate = working
            if nextstate != working:
                working = nextstate
                pending.pop(key, None)
                progress = True
        if not progress:
            break

    base = working
    trance = Trance(base, scrubbed).Scry()

    def Assemble(resolved: Field.State, protected: set[str]) -> Field.State:
        selected: dict[str, Field.Cell] = {}
        for candidate in scrubbed.cells:
            current = Field.FindCell(resolved, candidate.key)
            original = Field.FindCell(local, candidate.key)
            if current is None or original is None:
                raise ValueError('incoming Dream key missing from local Dream')
            if current.key in protected:
                selected[current.key] = current
                continue
            if original.purge.lockbit == 0:
                selected[current.key] = candidate
                continue
            outcome = Link(current, candidate)
            selected[current.key] = candidate if outcome.linked else current

        # If local receipt processing changed geometry, preserve that computed
        # geometry. A pure leapfrog may adopt the incoming geometry wholesale.
        order = resolved.cells if resolved != local else scrubbed.cells
        candidate = Field.State(
            cells=tuple(selected[cell.key] for cell in order),
            self=local.self,
            pristine=incoming.pristine,
        )
        Field.VerifyState(candidate, expectedkeys=keys)
        sealed = Seal(Field.Stasis(candidate))
        # A Dream is a proposed complete endpoint, not a bag of independently
        # adoptable coordinates. Locked cells are reproduced locally and open
        # cells may leapfrog, but the canonical complete Dream must equal the
        # proposal exactly before it can assimilate.
        if Field.Scrub(sealed) != scrubbed:
            raise ValueError('incoming Dream is not exactly reproducible')
        return sealed

    if not trance.frontier:
        # Full surrender may replace every released coordinate except a ReBurn.
        # A locally settled Burn remains a terminal lower bound even when all
        # PurgeLocks are open; surrendering anchors does not resurrect it.
        return Assemble(base, set(trance.reburn))

    # Crystallize normally first. If the complete Dream still cannot fit,
    # Scry's incoming zeroes are useful lower-bound information: re-Weigh once
    # with those zeroes as Burn candidates, never as remote authority.
    try:
        resolved = trance.Crystallize()
        protected = set(trance.touched) | set(trance.frontier) | trance.reburn | trance.burn | set(trance.damned)
        return Assemble(resolved, protected)
    except Exception:
        if not trance.zeroes:
            raise
    resolved = trance.Crystallize(trance.zeroes)
    protected = set(trance.touched) | set(trance.frontier) | trance.reburn | trance.burn | set(trance.damned)
    return Assemble(resolved, protected)


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
        self.omens: dict[str, Field.Lock] = {}
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
        # Genesis/bootstrap is already a complete state: it becomes the first
        # trusted frontier, so every PurgeLock begins latched and white.
        self.state = Seal(state) if firstreal else state
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
        # Chainbit is a wire observation, never a latch. An open PurgeLock has
        # deliberately withdrawn this lineage as an anchor, so its chainbit is
        # inactive until a full Dream is assimilated.
        lockbit = int(cell.purge.lockbit)
        cb = (1 if int(chainbit) else 0) if lockbit == 1 else 0
        return replace(cell, purge=Field.PurgeLocks(chainbit=cb, lockbit=lockbit))

    def ClearPurge(self, cell: Field.Cell) -> Field.Cell:
        return replace(cell, purge=Field.PurgeLocks(chainbit=0, lockbit=0))

    def Sample(self, state: Field.State, evidence: dict[str, tuple[Field.Lock, ...]]) -> Field.State:
        cells = []
        for cell in state.cells:
            if int(cell.purge.lockbit) == 0:
                cells.append(self.WithChainbit(cell, 0))
                continue
            locks = evidence.get(cell.key)
            cells.append(self.WithChainbit(cell, 1 if locks is not None and Kindred(cell, locks) else 0))
        return Field.State(cells=tuple(cells), self=state.self, pristine=state.pristine)

    def SampleSalt(self, state: Field.State, glyph: Field.SaltGlyph) -> Field.State:
        signer = Field.FindCell(state, glyph.lockbody.tag)
        evidence = {} if signer is None else {signer.key: (glyph.lockbody,)}
        return self.Sample(state, evidence)

    def SampleNightmare(self, state: Field.State, glyph: Field.NightmareGlyph) -> Field.State:
        signer = Field.FindCell(state, glyph.lowlock.tag)
        evidence = {} if signer is None else {signer.key: (glyph.lowlock, glyph.lock)}
        return self.Sample(state, evidence)

    def SampleDream(self, state: Field.State, incoming: Field.State) -> Field.State:
        incomingmap = {cell.key: Field.LockSet(cell) for cell in Field.Scrub(incoming).cells}
        return self.Sample(state, incomingmap)

    def ApplyPurgeKey(self, state: Field.State, key: str) -> Field.State:
        key = str(key or '').strip()
        if not key:
            return state
        target = Field.FindCell(state, key)
        if target is None:
            return state
        return Field.ReplaceCell(state, self.ClearPurge(target))

    def PurgeFlare(self) -> dict[str, Any]:
        return {'kind': PurgeGlyph, 'key': self.SelfKey()}

    def Assimilate(self, nextstate: Field.State) -> bool:
        # Every projection must cross this boundary first. Receipt, Nightmare,
        # Dream, or Trance crystallization: compute it locally, then project it.
        Field.VerifyState(nextstate, expectedkeys=Field.FindKeys(self.state) if self.state is not None else None)
        return self.Commit(nextstate)

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
            if kind in (SaltGlyph, NightmareGlyph) and self.state is not None:
                tag = glyph.lockbody.tag if kind == SaltGlyph else glyph.lowlock.tag
                current = Field.FindCell(self.state, tag)
                previous = Field.FindCell(before, tag) if before is not None else None
                if current is not None and current.lowlock is not None and (previous is None or Field.LockSet(previous) != Field.LockSet(current)):
                    self.Forward(Field.NightmareGlyph(lowlock=current.lowlock, lock=current.lock))
                    return mutated
            # A full Dream is reprojected as the Dream this node actually
            # assimilated, not blindly relayed as the packet it received.
            self.Forward(self.Scrub(self.state) if kind == DreamGlyph else glyph)
        return mutated


    def MutateSalt(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.SaltGlyph):
            raise TypeError('Dream.MutateSalt expects Field.SaltGlyph')
        before = self.state
        try:
            Field.VerifySalt(glyph, before)
        except Exception:
            return False
        observed = self.SampleSalt(before, glyph)
        if observed != before:
            self.state = observed
            self.dreamfall = True
        lock = glyph.lockbody
        signer = Field.FindCell(observed, lock.tag)
        if signer is None:
            return False
        held = self.omens.get(signer.key)
        parents = {item.parent for item in Field.LockSet(signer)} | {Field.ContinuationChild(signer)}
        if held is not None and held.parent not in parents:
            self.omens.pop(signer.key, None)
            held = None
        pair = Field.Pair(*((held,) if held is not None else ()), lock)
        evidence: Field.Lock | tuple[Field.Lock, ...] = pair or lock
        if pair is not None:
            self.omens.pop(signer.key, None)
        elif lock.parent == Field.ContinuationChild(signer) and Field.Scorched(observed, lock):
            self.omens[signer.key] = lock
            return False
        try:
            nextstate, _chains = Reconcile(observed, evidence)
        except Exception:
            return False
        if nextstate == observed:
            return False
        return self.Assimilate(nextstate)

    def MutateNightmare(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.NightmareGlyph):
            raise TypeError('Dream.MutateNightmare expects NightmareGlyph')
        before = self.state
        try:
            Field.VerifyNightmare(glyph, before)
        except Exception:
            return False
        observed = self.SampleNightmare(before, glyph)
        if observed != before:
            self.state = observed
            self.dreamfall = True
        try:
            nextstate, _chains = Reconcile(observed, (glyph.lowlock, glyph.lock))
        except Exception:
            return False
        if nextstate == observed:
            return False
        return self.Assimilate(nextstate)

    def MutateDream(self, glyph: Any, source: str = '') -> bool:
        if self.state is None:
            return False
        if not isinstance(glyph, Field.State):
            raise TypeError('Dream.MutateDream expects Field.State')
        before = self.state
        try:
            Field.VerifyDream(glyph, expectedkeys=Field.FindKeys(before))
        except Exception:
            return False
        observed = self.SampleDream(before, glyph)
        if observed != before:
            self.state = observed
            self.dreamfall = True
        try:
            nextstate = ReconcileDream(observed, glyph)
        except Exception:
            return False
        if nextstate == observed:
            return False
        return self.Assimilate(nextstate)

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
            key = self.PurgeKey(glyph)
            if key and key == self.SelfKey():
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
