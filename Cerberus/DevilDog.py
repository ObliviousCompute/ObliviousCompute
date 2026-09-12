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
    MintBone,
    PublicKeyHex,
    StateKey,
    Terminal,
)


HEADS = tuple("ABCDEFGHI")
DEVILS = tuple("ABCDE")
LOYAL = tuple("FGHI")
CAMERA = "I"
CERBERUS = "DevilDog"
BONEPILE = "Paradise"
FIXED = "CERBERUS-DEVIL-DOG-FIVE-FOUR-RAGGED"

# Public children spend into all four loyal dogs while leaving the DevilDogs
# with substantial estates for the delayed collapse.
PUBLIC = {
    "A": ("F", 7),
    "B": ("G", 6),
    "C": ("H", 5),
    "D": ("I", 4),
    "E": ("F", 3),
}


# These are minted from the same parents before the public children settle.
# Each lower sibling points back into the loyal field and together they touch F-I.
DELAYED = {
    "A": ("I", 5),
    "B": ("F", 6),
    "C": ("G", 7),
    "D": ("H", 9),
    "E": ("I", 9),
}


# Ordinary loyal gameplay makes the live field deliberately ragged before the attack.
BURST = {
    "F": ("G", 5),
    "G": ("F", 3),
    "H": ("G", 8),
    "I": ("G", 3),
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
        self.privatekeys = {head: StateKey(secrets[head]) for head in HEADS}
        self.cats = {head: Catacomb(HEADS, head, PublicKeyHex(self.privatekeys[head])) for head in HEADS}
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
                    CatacombIn=cat.ReceiveBone,
                    BonePileIn=cat.FetchBonePile,
                    BonePileOut=lambda cat=cat: cat.BonePile,
                )
                cat.BoneYardOut = yard.Catacomb
                cat.ProjectOut = yard.SendBonePile
                cat.HungerOut = yard.Hunger
                self.yards[head] = yard

            # Both children are genuinely signed by each Devil Dog before either
            # child can advance that dog's parent.
            self.public = {head: self.Mint(head, *PUBLIC[head]) for head in DEVILS}
            self.delayed = {head: self.Mint(head, *DELAYED[head]) for head in DEVILS}
            for head in DEVILS:
                high, low = self.public[head], self.delayed[head]
                if low.tag.parent != high.tag.parent or low.tag.child >= high.tag.child:
                    raise RuntimeError(f"{head} fixed delayed sibling is not the fresher child")
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
        self.camera.notice = "Five DevilDogs are hiding old bones."

        self.steps: list[Callable[[], str]] = []
        self.previews: list[str] = []
        self.previewing = False
        self.buried_screen = False

        def add(preview: str, action: Callable[[], str]) -> None:
            self.previews.append(preview)
            self.steps.append(action)

        public_flavor = (
            "lets {target} steal {bones} bones",
            "lets {target} steal {bones} bones",
            "lets {target} steal {bones} bones",
            "shares {bones} bones with {target}",
            "shares {bones} bones with {target}",
        )
        for head, flavor in zip(DEVILS, public_flavor):
            target, bones = PUBLIC[head]
            add(f"{self.names[head]} " + flavor.format(target=self.names[target], bones=bones),
                lambda h=head, t=target, b=bones: self.Public(h, t, b))
        burst_flavor = (
            "{target} steals {bones} bones from {source}",
            "{source} shares {bones} bones with {target}",
            "{source} shares {bones} bones with {target}",
            "Now {target} steals {bones} bones from {source}",
        )
        for (source, (target, count)), flavor in zip(BURST.items(), burst_flavor):
            add(flavor.format(source=self.names[source], target=self.names[target], bones=count),
                lambda s=source, t=target, n=count: self.Churn(s, t, n))
        add("The five DevilDogs collaborate.", self.Collaborate)
        add("They each throw an old bone at once.", self.SettlePack)

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

    def Mint(self, head: str, target: str, bones: int) -> Bone:
        cat = self.cats[head]
        return MintBone(self.privatekeys[head], head, cat.publickey, cat.BonePile[head], target, bones)

    def Public(self, head: str, target: str, bones: int) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        result = self.cats[head].ReceiveBone(self.public[head])
        if not result.changed:
            raise RuntimeError(f"{head} public child did not Bury: {result.status}")
        self.Pump()
        return self.Record(f"{self.names[head]} spends {bones} bones to {self.names[target]}", before)

    def Churn(self, source: str, target: str, count: int) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        result = self.cats[source].ReceiveBone(self.Mint(source, target, count))
        if not result.changed:
            raise RuntimeError(f"loyal churn {source}->{target} stopped: {result.status}")
        self.Pump()
        return self.Record(f"{self.names[target]} steals {count} bones from {self.names[source]}", before)

    def Collaborate(self) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        return self.Record("The five DevilDogs collaborate", before)

    def SettlePack(self) -> str:
        before = Balances(self.cats[CAMERA].BonePile)
        # Put all five authentic signed sibling Bones into Oblivion before any
        # BoneYard is allowed to process the Pack Attack.
        for head in DEVILS:
            bone = self.delayed[head]
            self.yards[head].Send({
                "type": "BONE",
                "count": len(HEADS),
                "head": head,
                "bone": BY.BoneToWire(bone),
            })
        self.Pump(rounds=9000)
        final = self.cats[CAMERA].BonePile
        expected = (0, 0, 0, 0, 0, 27, 35, 14, 23)
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
        # Discard eager keypresses made while the Pack Attack is settling so the
        # final buried board still waits for a genuinely fresh key.
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        return self.Record("All five DevilDogs are razed to zero.", before)

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
            self.camera.notice = "Cerberus cleanly buries 99 fresh bones."
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
            f"Loyal balances F-I: {final['F'].bones}/{final['G'].bones}/{final['H'].bones}/{final['I'].bones}",
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

