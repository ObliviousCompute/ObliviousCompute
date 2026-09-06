"""DevilDog: five equivocators, four loyal heads.
Nine production Cerberus heads play through Oblivion.
Delayed signed siblings create the Pack Attack.
The field must reconcile without external authority.
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
    Heads,
    Terminal,
)

HEADS = tuple("ABCDEFGHI")
DEVILS = tuple("ABCDE")
LOYAL = tuple("FGHI")
CAMERA = "I"
CERBERUS = "DevilDog"
BONEPILE = "Paradise"
FIXED = "CERBERUS-DEVIL-DOG-FIVE-FOUR-RAGGED"

# Public children drain the five Devil Dogs to 1/2/3/4/5 and touch only
# loyal F/G/H.  I stays outside the eventual DogPile and must remain ossified.
PUBLIC = {
    "A": ("F", 10),
    "B": ("G", 9),
    "C": ("H", 8),
    "D": ("F", 7),
    "E": ("G", 6),
}

# These are minted from the same parents before the public children settle.
# Each is the lower canonical sibling for the fixed production keys.
DELAYED = {
    "A": ("B", 5),
    "B": ("C", 11),
    "C": ("D", 11),
    "D": ("E", 10),
    "E": ("A", 11),
}

# Ordinary loyal gameplay after the public Bone Bucks have entered the field.
BURST = {
    "F": ("G", 10),
    "G": ("H", 7),
    "H": ("I", 5),
    "I": ("F", 2),
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
    start = 20000 + ((os.getpid() * 17) % 800) * 9
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
    raise RuntimeError("DevilDog could not find nine clean Oblivion mouths")


@dataclass
class TrialEvent:
    number: int
    text: str
    before: tuple[int, ...]
    after: tuple[int, ...]


class DevilDogTrial:
    """One readable 5/4 Cerberus game played through production Oblivion."""

    def __init__(self) -> None:
        self.baseport = FindPortBase()
        BY.BonePilePort = self.baseport
        self.ring = f"DevilDog|{os.getpid()}|{self.baseport}"
        self.events: list[TrialEvent] = []
        self.notices: list[tuple[str, str]] = []
        self.stepindex = 0
        self.finished = False

        secrets = {head: f"{FIXED}|{head}" for head in HEADS}
        self.cats = {head: Catacomb(HEADS, head, secrets[head]) for head in HEADS}
        genesis = BonePile({head: self.cats[head].GenesisCell for head in HEADS})
        for cat in self.cats.values():
            if not cat.Seed(genesis).changed:
                raise RuntimeError("DevilDog Genesis did not Bury")

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

            # Both children are genuinely signed by each Devil Dog before either
            # child can advance that dog's parent.
            self.public = {head: self.cats[head].Mint(*PUBLIC[head]) for head in DEVILS}
            self.delayed = {head: self.cats[head].Mint(*DELAYED[head]) for head in DEVILS}
            for head in DEVILS:
                high, low = self.public[head], self.delayed[head]
                if low.tag.parent != high.tag.parent or low.tag.child >= high.tag.child:
                    raise RuntimeError(f"{head} fixed delayed sibling is not the lower child")
        except Exception:
            self.Close()
            raise

        ranked = sorted(
            Guardians,
            key=lambda name: HashRank(FIXED, "DEVILDOGNAME", name),
            reverse=True,
        )[: len(HEADS)]
        self.names = dict(zip(HEADS, ranked))
        self.headcounthash = HeadCountHash({self.cats[h].publickey: self.names[h] for h in HEADS})

        # Guardian is presentation only here; the head underneath is the real I
        # Catacomb + BoneYard pair used by the nine-head trial.
        self.camera = Guardian(CERBERUS)
        self.camera.boneyard.Close()
        self.camera.count = len(HEADS)
        self.camera.dogtag = self.names[CAMERA]
        self.camera.bonepile = BONEPILE
        self.camera.publickey = self.cats[CAMERA].publickey
        self.camera.headcounthash = self.headcounthash
        self.camera.heads = list(HEADS)
        self.camera.expected = set(HEADS)
        self.camera.head = CAMERA
        self.camera.target = "F"
        self.camera.amount = 1
        self.camera.names = dict(self.names)
        self.camera.catacomb = self.cats[CAMERA]
        self.camera.boneyard = self.yards[CAMERA]
        self.camera.state = self.cats[CAMERA].BonePile
        self.camera.notice = "Five Devil Dogs are hiding bones."

        self.steps: list[Callable[[], str]] = []
        self.previews: list[str] = []
        self.previewing = False
        self.buried_screen = False

        def add(preview: str, action: Callable[[], str]) -> None:
            self.previews.append(preview)
            self.steps.append(action)

        for head in DEVILS:
            target, bones = PUBLIC[head]
            add(
                f"{self.names[head]} sends {bones} bones to {self.names[target]}",
                lambda h=head, t=target, b=bones: self.Public(h, t, b),
            )
        for source, (target, count) in BURST.items():
            add(
                f"{self.names[source]} moves {count} bones through {self.names[target]}",
                lambda s=source, t=target, n=count: self.Churn(s, t, n),
            )
        for head in DEVILS:
            target, bones = DELAYED[head]
            add(
                f"{self.names[head]} drops one more bone into Oblivion",
                lambda h=head, t=target, b=bones: self.QueueDelayed(h, t, b),
            )
        add("The delayed Pack Attack hits Oblivion.", self.SettlePack)

    def Same(self) -> bool:
        pile = self.cats[HEADS[0]].BonePile
        return all(self.cats[head].BonePile == pile for head in HEADS[1:])

    def Pump(self, rounds: int = 5000) -> None:
        for index in range(rounds):
            for yard in self.yards.values():
                yard.Pump()
            if self.Same():
                for _ in range(16):
                    for yard in self.yards.values():
                        yard.Pump()
                if self.Same():
                    return
            if index % 32 == 31:
                time.sleep(0.0002)
        raise RuntimeError("DevilDog Oblivion did not settle")

    def Record(self, text: str, before: tuple[int, ...]) -> str:
        after = Balances(self.cats[CAMERA].BonePile)
        self.events.append(TrialEvent(len(self.events) + 1, text, before, after))
        self.camera.notice = text
        return text

    def Public(self, head: str, target: str, bones: int) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        result = self.cats[head].ReceiveBone(self.public[head])
        if not result.changed:
            raise RuntimeError(f"{head} public child did not Bury: {result.status}")
        self.Pump()
        return self.Record(f"{self.names[head]} spends {bones} bones to {self.names[target]}", before)

    def Churn(self, source: str, target: str, count: int) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        for _ in range(count):
            result = self.cats[source].Guardian(target, 1)
            if not result.changed:
                raise RuntimeError(f"loyal churn {source}->{target} stopped: {result.status}")
        self.Pump()
        return self.Record(
            f"{self.names[source]} moves {count} bones through {self.names[target]}",
            before,
        )

    def QueueDelayed(self, head: str, target: str, bones: int) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        result = self.cats[head].ReceiveBone(self.delayed[head])
        if not result.changed or not result.reproject:
            raise RuntimeError(f"{head} delayed sibling was not projected: {result.status}")
        # Deliberately do not pump.  The signed sibling is now in real UDP
        # Oblivion, queued alongside the rest of the delayed pack.
        return self.Record(
            f"{self.names[head]} drops one more bone into Oblivion",
            before,
        )

    def SettlePack(self) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        # This is the expensive boss-fight step.  The preview screen is already
        # visible while the real nine-head Bone Storm settles at machine speed.
        self.Pump(rounds=9000)
        final = self.cats[CAMERA].BonePile
        expected = (0, 0, 0, 0, 0, 29, 28, 28, 14)
        if Balances(final) != expected:
            raise RuntimeError(f"DevilDog settled to {Balances(final)}, expected {expected}")
        if DirtyDogs(final) != frozenset(DEVILS):
            raise RuntimeError(f"wrong DirtyDogs: {sorted(DirtyDogs(final))}")
        if Total(final) != 99 or not self.Same():
            raise RuntimeError("DevilDog did not finish as one conserved BonePile")
        if any(cat.Hungry for cat in self.cats.values()):
            raise RuntimeError("DevilDog left a head Hungry")
        if self.notices:
            raise RuntimeError(f"DevilDog emitted bad-packet notices: {self.notices}")
        self.finished = True
        # The Pack Attack can take long enough for an eager viewer to tap or hold
        # a key while Cerberus is settling. Throw those stale bytes away here so
        # the final buried board waits for a genuinely fresh key before entering
        # the Void.
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        return self.Record("All Devil Dogs are razed to zero.", before)

    def Step(self) -> bool:
        # The trial is intentionally preview-first.  The first key only announces
        # the next move.  Each later key commits the announced move, then leaves
        # the resulting field on screen while announcing what will happen next.
        if not self.previewing:
            self.previewing = True
            self.camera.notice = self.previews[0]
            return False

        if self.stepindex < len(self.steps):
            action = self.steps[self.stepindex]
            self.stepindex += 1
            action()

            # The Pack Attack is the final mutating step.  Hold its settled field
            # on a dedicated screen before the separate Bury screen.
            if self.stepindex == len(self.steps):
                return False

            self.camera.notice = self.previews[self.stepindex]
            return False

        if not self.buried_screen:
            self.buried_screen = True
            self.camera.notice = "Cerberus Buries 99 bones."
            return False

        return True

    def Proofs(self) -> str:
        final = self.cats[CAMERA].BonePile
        dirty = DirtyDogs(final)
        dogpile = {
            receipt.target
            for head in dirty
            for receipt in final[head].receipts
        }
        active = set(dogpile) - set(dirty)
        lines = [
            "CERBERUS DEVIL DOG PROOFS",
            "==========================",
            "",
            "WHAT THIS RUN IS",
            "Nine independent production Cerberus heads maintain separate BonePiles.",
            "Five heads withhold signed sibling receipts while four loyal heads continue normal play.",
            "All Bones and BonePiles reconcile through production BoneYard / UDP Oblivion.",
            "DevilDog supplies gameplay actions; it does not assign balances or replace Catacomb logic.",
            "",
            "WHAT TO CHECK",
            "Five delayed equivocators are Razed through signed evidence.",
            "The four loyal heads independently Bury the same final BonePile.",
            "The field remains conserved at 99 bones.",
            "",
            "TRIAL",
            "DevilDog 5/4 — ragged delayed Pack Attack gameplay",
            "",
            "COMPUTER",
            f"Production Catacombs: {len(self.cats)}",
            f"Production BoneYards: {len(self.yards)}",
            f"Distinct public keys: {len({cat.publickey for cat in self.cats.values()})}",
            f"UDP Oblivion mouths: {', '.join(str(self.yards[h].bindport) for h in HEADS)}",
            f"Visible loyal head: {CAMERA} / {self.names[CAMERA]}",
            "",
            "HEADS",
        ]
        for head in HEADS:
            role = "DEVIL" if head in DEVILS else "LOYAL"
            lines.append(
                f"{head} {self.names[head]:<8} {role:<5} "
                f"key={self.cats[head].publickey[:16]} port={self.yards[head].bindport}"
            )
        lines += ["", "SIGNED DELAYED SIBLINGS"]
        for head in DEVILS:
            bone = self.delayed[head]
            lines.append(
                f"{head}->{bone.target} bones={bone.bones:>2} "
                f"parent={bone.tag.parent[:12]} child={bone.tag.child[:12]} "
                f"receipt={ReceiptHash(bone)[:12]} sign={bone.sign[:12]}"
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
            "FINAL FIELD",
            f"balances A-I: {Balances(final)}",
            f"DirtyDogs: {''.join(sorted(dirty))}",
            f"DogPile: {''.join(sorted(dogpile))}",
            f"Active DogPile: {''.join(sorted(active))}",
            f"I outside DogPile / ossified balance: {final['I'].bones}",
            f"bones: {Total(final)} / 99",
            "",
            "FINAL BONEPILE HASHES",
        ]
        for head in HEADS:
            lines.append(f"{head} {PileHash(self.cats[head].BonePile)}")
        loyalhashes = {PileHash(self.cats[head].BonePile) for head in LOYAL}
        allhashes = {PileHash(self.cats[head].BonePile) for head in HEADS}
        lines += [
            "",
            f"Loyal agreement: {str(len(loyalhashes) == 1).upper()}",
            f"All nine buried same BonePile: {str(len(allhashes) == 1).upper()}",
            f"Hungry heads: {''.join(h for h in HEADS if self.cats[h].Hungry) or 'NONE'}",
            f"Bad packet notices: {self.notices or 'NONE'}",
            f"99 invariant: {str(Total(final) == 99).upper()}",
        ]
        return "\n".join(lines)

    def RunAll(self) -> None:
        while self.stepindex < len(self.steps):
            self.steps[self.stepindex]()
            self.stepindex += 1

    def Close(self) -> None:
        for yard in getattr(self, "yards", {}).values():
            yard.Close()


def Run(*, proofs_only: bool = False) -> None:
    trial = DevilDogTrial()
    try:
        if proofs_only:
            trial.RunAll()
            sys.stdout.write(trial.Proofs() + "\n")
            return
        if not sys.stdin.isatty():
            raise RuntimeError("DevilDog needs a terminal; use `cerberus DevilDog proofs` for text output")
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
