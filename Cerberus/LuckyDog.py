"""LuckyDog: Last Dog Standing in Oblivion.
Nine production Cerberus heads begin with 99 bones.
Ordinary play continues until the dogs get greedy.
The surviving Lucky dog inherits the whole field.
Proof mode exposes the execution for inspection.
"""
from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import termios
import time
from dataclasses import dataclass
from typing import Callable

import Game.BoneYard as BY
from Game.BoneYard import BonePileToWire, BoneYard, DirtyDogs
from Game.Catacomb import Bone, BonePile, Catacomb, ReceiptHash
from Game.Guardian import (
    Clear,
    Guardian,
    Guardians,
    HashRank,
    HeadCountHash,
    Terminal,
)

HEADS = tuple("ABCDEFGHI")
FALLEN = tuple("ABCDEFGH")
CAMERA = "I"
CERBERUS = "LuckyDog"
BONEPILE = "Paradise"
FIXED = "CERBERUS-LUCKY-DOG-LAST-DOG-STANDING"

# Ordinary gameplay runs until the last dogs stop sharing.
# The sources are the names narrated on screen; all moves are production Bones.
BUNDLES = (
    (("B", "C", 2), ("D", "E", 4), ("F", "G", 3)),
    (("B", "D", 6), ("E", "F", 5), ("G", "H", 9)),
    (("C", "E", 10), ("F", "G", 12), ("H", "I", 14)),
    (("D", "F", 17), ("G", "H", 15), ("I", "E", 18)),
    (("E", "G", 12), ("H", "I", 19), ("F", "G", 16)),
)

# Each dog signs two same-parent spends for its entire current estate.  Both are
# individually payable; together they oversubscribe the parent and Razed-to-zero
# reconciliation tightens the live field around the remaining clean dogs.
GREED_TARGETS = {
    "A": ("B", "C"),
    "B": ("C", "D"),
    "C": ("D", "E"),
    "D": ("E", "F"),
    "E": ("F", "G"),
    "F": ("G", "H"),
    "G": ("H", "A"),
    "H": ("I", "A"),
}


def Balances(pile: BonePile) -> tuple[int, ...]:
    return tuple(int(pile[head].bones) for head in HEADS)


def Total(pile: BonePile) -> int:
    return sum(Balances(pile))


def PileHash(pile: BonePile) -> str:
    body = json.dumps(BonePileToWire(pile), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(body).hexdigest()


def FindPortBase() -> int:
    """Find nine clean consecutive localhost UDP mouths for this trial."""
    start = 24000 + ((os.getpid() * 23) % 700) * 9
    for base in range(start, min(start + 5000, 60000), 9):
        held: list[socket.socket] = []
        try:
            for port in range(base, base + len(HEADS)):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.bind((BY.Host, port))
                held.append(sock)
            return base
        except OSError:
            pass
        finally:
            for sock in held:
                sock.close()
    raise RuntimeError("LuckyDog could not find nine clean Oblivion mouths")


@dataclass
class TrialEvent:
    number: int
    text: str
    before: tuple[int, ...]
    after: tuple[int, ...]


@dataclass
class GreedEvidence:
    head: str
    estate: int
    first: Bone
    second: Bone


class LuckyDogTrial:
    """Last Dog Standing, played by nine production Cerberus heads."""

    def __init__(self) -> None:
        self.baseport = FindPortBase()
        BY.BonePilePort = self.baseport
        self.ring = f"LuckyDog|{os.getpid()}|{self.baseport}"
        self.events: list[TrialEvent] = []
        self.greeds: list[GreedEvidence] = []
        self.notices: list[tuple[str, str]] = []
        self.stepindex = 0
        self.previewing = False
        self.ending = 0
        self.final_attempt = None

        secrets = {head: f"{FIXED}|{head}" for head in HEADS}
        self.cats = {head: Catacomb(HEADS, head, secrets[head]) for head in HEADS}
        genesis = BonePile({head: self.cats[head].GenesisCell for head in HEADS})
        for cat in self.cats.values():
            if not cat.Seed(genesis).changed:
                raise RuntimeError("LuckyDog Genesis did not Bury")

        self.yards: dict[str, BoneYard] = {}
        try:
            for head in HEADS:
                cat = self.cats[head]
                yard = BoneYard(
                    self.ring,
                    NoticeOut=lambda message, head=head: self.notices.append((head, message)),
                )
                yard.Open(len(HEADS))
                yard.Attach(
                    HEADS,
                    head,
                    CatacombIn=cat.BoneYard,
                    BonePileIn=cat.FetchBonePile,
                    BonePileOut=lambda cat=cat: cat.BonePile,
                )
                cat.BoneYardOut = yard.Catacomb
                cat.ProjectOut = yard.SendBonePile
                cat.HungerOut = yard.Hunger
                self.yards[head] = yard
        except Exception:
            self.Close()
            raise

        pool = tuple(name for name in Guardians if name != "Lucky")
        ranked = sorted(
            pool,
            key=lambda name: HashRank(FIXED, "LUCKYDOGNAME", name),
            reverse=True,
        )[: len(HEADS) - 1]
        self.names = dict(zip(FALLEN, ranked))
        self.names[CAMERA] = "Lucky"
        self.headcounthash = HeadCountHash({self.cats[h].publickey: self.names[h] for h in HEADS})

        # The visible board is the actual Lucky head's Catacomb + BoneYard pair.
        self.camera = Guardian(CERBERUS)
        self.camera.boneyard.Close()
        self.camera.count = len(HEADS)
        self.camera.dogtag = "Lucky"
        self.camera.bonepile = BONEPILE
        self.camera.publickey = self.cats[CAMERA].publickey
        self.camera.headcounthash = self.headcounthash
        self.camera.heads = list(HEADS)
        self.camera.expected = set(HEADS)
        self.camera.head = CAMERA
        self.camera.target = "H"
        self.camera.amount = 1
        self.camera.names = dict(self.names)
        self.camera.catacomb = self.cats[CAMERA]
        self.camera.boneyard = self.yards[CAMERA]
        self.camera.state = self.cats[CAMERA].BonePile
        self.camera.notice = "Nine dogs share 99 bones."

        self.steps: list[Callable[[], str]] = []
        self.previews: list[str] = []

        def add(preview: str, action: Callable[[], str]) -> None:
            self.previews.append(preview)
            self.steps.append(action)

        leads = ("But ", "But then ", "", "Of course, ", "")
        for index, head in enumerate(FALLEN):
            if index < len(BUNDLES):
                moves = BUNDLES[index]
                sources = [self.names[source] for source, _, _ in moves]
                add(f"{', '.join(sources[:-1])}, and {sources[-1]} share some bones.", lambda moves=moves: self.Bundle(moves))
                add(f"{leads[index]}{self.names[head]} gets sneaky.", lambda head=head: self.Greed(head))
            else:
                add("All dogs get greedy. Nobody moves a bone.", lambda: "")
                add(f"Until {self.names[head]} gets sneaky.", lambda head=head: self.Greed(head))

    def Same(self) -> bool:
        pile = self.cats[HEADS[0]].BonePile
        return all(self.cats[head].BonePile == pile for head in HEADS[1:])

    def Fresh(self) -> None:
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)

    def Pump(self, rounds: int = 9000) -> None:
        for index in range(rounds):
            for yard in self.yards.values():
                yard.Pump()
            if self.Same():
                for _ in range(16):
                    for yard in self.yards.values():
                        yard.Pump()
                if self.Same():
                    self.Fresh()
                    return
            if index % 32 == 31:
                time.sleep(0.0002)
        raise RuntimeError("LuckyDog Oblivion did not settle")

    def Record(self, text: str, before: tuple[int, ...]) -> str:
        after = Balances(self.cats[CAMERA].BonePile)
        self.events.append(TrialEvent(len(self.events) + 1, text, before, after))
        return text

    def Bundle(self, moves: tuple[tuple[str, str, int], ...]) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        names = []
        for source, target, bones in moves:
            result = self.cats[source].Guardian(target, bones)
            if not result.changed:
                raise RuntimeError(f"ordinary LuckyDog play {source}->{target} failed: {result.status}")
            names.append(self.names[source])
        self.Pump()
        return self.Record(f"{', '.join(names)} played normally", before)

    def Greed(self, head: str) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        estate = int(self.cats[head].BonePile[head].bones)
        if estate <= 0:
            raise RuntimeError(f"{head} has no estate left to oversubscribe")
        firsttarget, secondtarget = GREED_TARGETS[head]
        first = self.cats[head].Mint(firsttarget, estate)
        second = self.cats[head].Mint(secondtarget, estate)
        if first.tag.parent != second.tag.parent or first.tag.child == second.tag.child:
            raise RuntimeError(f"{head} did not mint a genuine sibling pair")
        one = self.cats[head].ReceiveBone(first)
        two = self.cats[head].ReceiveBone(second)
        if not one.changed or not two.changed or not two.reproject:
            raise RuntimeError(f"{head} greed did not create an equivocation collapse: {one.status}/{two.status}")
        self.Pump()
        self.greeds.append(GreedEvidence(head, estate, first, second))
        expected = frozenset(FALLEN[: FALLEN.index(head) + 1])
        actual = DirtyDogs(self.cats[CAMERA].BonePile)
        if actual != expected:
            raise RuntimeError(f"{head} greed produced DirtyDogs {sorted(actual)}, expected {sorted(expected)}")
        if Total(self.cats[CAMERA].BonePile) != 99:
            raise RuntimeError("LuckyDog lost the 99 invariant")
        return self.Record(f"{self.names[head]} gets sneaky", before)

    def LuckyAttempt(self) -> None:
        beforepile = self.cats[CAMERA].BonePile
        before = Balances(beforepile)
        result = self.cats[CAMERA].Guardian("A", 100)
        after = Balances(self.cats[CAMERA].BonePile)
        self.final_attempt = result
        self.events.append(TrialEvent(len(self.events) + 1, "Lucky tries to bury 100 bones", before, after))
        if result.changed or after != before or Total(self.cats[CAMERA].BonePile) != 99:
            raise RuntimeError("Lucky's impossible 100-bone attempt changed the last valid field")
        if not self.Same():
            raise RuntimeError("Lucky's impossible attempt disturbed loyal agreement")

    def Step(self) -> bool:
        # Same preview-first cadence as DevilDog: the caption describes what the
        # next key will do while the board still shows the current buried field.
        if not self.previewing:
            self.previewing = True
            self.camera.notice = self.previews[0]
            return False

        if self.stepindex < len(self.steps):
            action = self.steps[self.stepindex]
            self.stepindex += 1
            action()
            if self.stepindex < len(self.steps):
                self.camera.notice = self.previews[self.stepindex]
                return False
            final = self.cats[CAMERA].BonePile
            if Balances(final) != (0, 0, 0, 0, 0, 0, 0, 0, 99):
                raise RuntimeError(f"LuckyDog ended at {Balances(final)} instead of Last Dog Standing")
            if DirtyDogs(final) != frozenset(FALLEN):
                raise RuntimeError("LuckyDog did not leave exactly eight DirtyDogs")
            if any(cat.Hungry for cat in self.cats.values()) or self.notices:
                raise RuntimeError("LuckyDog did not finish cleanly")
            self.camera.notice = "Now, Lucky has all 99 bones."
            return False

        if self.ending == 0:
            self.ending = 1
            self.camera.notice = "But of course, Lucky can't help it."
            return False
        if self.ending == 1:
            self.ending = 2
            self.camera.notice = "Lucky gets sneaky and tries to bury 100 bones."
            return False
        if self.ending == 2:
            self.ending = 3
            self.LuckyAttempt()
            self.camera.notice = self.final_attempt.status
            return False

        return True

    def Proofs(self) -> str:
        final = self.cats[CAMERA].BonePile
        dirty = DirtyDogs(final)
        lines = [
            "CERBERUS LUCKY DOG PROOFS",
            "==========================",
            "",
            "WHAT THIS RUN IS",
            "Nine independent production Cerberus heads maintain separate BonePiles.",
            "Five rounds of ordinary play give way to three still rounds before the final collapses.",
            "All Bones and BonePiles reconcile through production BoneYard / UDP Oblivion.",
            "LuckyDog supplies gameplay actions; it does not assign balances or replace Catacomb logic.",
            "",
            "WHAT TO CHECK",
            "Eight sneaky heads are Razed through signed sibling evidence.",
            "The field converges until Lucky alone holds all 99 bones.",
            "Lucky's final 100-bone attempt is rejected without changing the last buried BonePile.",
            "",
            "TRIAL",
            "LuckyDog — Last Dog Standing gameplay",
            "",
            "COMPUTER",
            f"Production Catacombs: {len(self.cats)}",
            f"Production BoneYards: {len(self.yards)}",
            f"Distinct public keys: {len({cat.publickey for cat in self.cats.values()})}",
            f"UDP Oblivion mouths: {', '.join(str(self.yards[h].bindport) for h in HEADS)}",
            f"Visible head: {CAMERA} / Lucky",
            "",
            "HEADS",
        ]
        for head in HEADS:
            role = "LUCKY" if head == CAMERA else "SNEAKY"
            lines.append(
                f"{head} {self.names[head]:<8} {role:<6} "
                f"key={self.cats[head].publickey[:16]} port={self.yards[head].bindport}"
            )

        lines += ["", "SIGNED SNEAKY SIBLINGS"]
        for evidence in self.greeds:
            first, second = evidence.first, evidence.second
            lines.append(
                f"{evidence.head} {self.names[evidence.head]:<8} estate={evidence.estate:>2} "
                f"claims={first.bones}+{second.bones} targets={first.target}/{second.target}"
            )
            lines.append(
                f"   parent={first.tag.parent[:12]} "
                f"children={first.tag.child[:12]}/{second.tag.child[:12]} "
                f"receipts={ReceiptHash(first)[:12]}/{ReceiptHash(second)[:12]}"
            )

        lines += ["", "OBSERVED GAMEPLAY"]
        for event in self.events:
            lines.append(
                f"{event.number:02d} {event.text}\n"
                f"   before={event.before}\n"
                f"   after ={event.after}"
            )

        lines += [
            "",
            "LAST BURIED FIELD",
            f"balances A-I: {Balances(final)}",
            f"DirtyDogs: {''.join(sorted(dirty))}",
            f"Lucky balance: {final[CAMERA].bones}",
            f"bones: {Total(final)} / 99",
            "",
            "FINAL ATTEMPT",
            "Lucky attempted to spend 100 bones while holding 99.",
            f"Result: {getattr(self.final_attempt, 'status', 'NOT RUN')}",
            f"Changed: {str(bool(getattr(self.final_attempt, 'changed', False))).upper()}",
            f"Last valid BonePile unchanged: {str(Balances(final) == (0,0,0,0,0,0,0,0,99)).upper()}",
            "",
            "FINAL BONEPILE HASHES",
        ]
        for head in HEADS:
            lines.append(f"{head} {PileHash(self.cats[head].BonePile)}")
        hashes = {PileHash(self.cats[head].BonePile) for head in HEADS}
        lines += [
            "",
            f"All nine buried same BonePile: {str(len(hashes) == 1).upper()}",
            f"Hungry heads: {''.join(h for h in HEADS if self.cats[h].Hungry) or 'NONE'}",
            f"Bad packet notices: {self.notices or 'NONE'}",
            f"99 invariant: {str(Total(final) == 99).upper()}",
        ]
        return "\n".join(lines)

    def RunAll(self) -> None:
        # Proof mode executes exactly the same scripted actions without TTY pauses.
        while self.stepindex < len(self.steps):
            self.steps[self.stepindex]()
            self.stepindex += 1
        final = self.cats[CAMERA].BonePile
        if Balances(final) != (0, 0, 0, 0, 0, 0, 0, 0, 99):
            raise RuntimeError("LuckyDog proof mode did not reach Last Dog Standing")
        self.LuckyAttempt()

    def Close(self) -> None:
        for yard in getattr(self, "yards", {}).values():
            yard.Close()


def Run(*, proofs_only: bool = False) -> None:
    trial = LuckyDogTrial()
    try:
        if proofs_only:
            trial.RunAll()
            sys.stdout.write(trial.Proofs() + "\n")
            return
        if not sys.stdin.isatty():
            raise RuntimeError("LuckyDog needs a terminal; use `cerberus LuckyDog proofs` for text output")
        with Terminal() as fd:
            trial.camera.Trial(fd, trial.Step, trial.Proofs)
    finally:
        trial.Close()
    sys.stdout.write(Clear)
    sys.stdout.flush()


def main() -> None:
    proofs = len(sys.argv) > 1 and sys.argv[1] == "proofs"
    Run(proofs_only=proofs)


if __name__ == "__main__":
    main()
