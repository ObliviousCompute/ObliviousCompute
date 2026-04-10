from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import Dream
import Field


Files = 4
Ranks = 6
CellCount = Files * Ranks
GeneralCount = Files
CaptainCount = CellCount - GeneralCount
CaptainsPerFile = Ranks - 1
Reserve = 1000000

SoulPair = Tuple[str, str]


def Sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def Rng(root: str, label: str) -> random.Random:
    seed = Sha(f"{root}|{label}")
    return random.Random(int(seed, 16))


def SoulShape(item: Any) -> Optional[SoulPair]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        soul = str(item[0] or "").strip()
        key = str(item[1] or "").strip()
        return (soul, key) if key else None
    if isinstance(item, dict):
        soul = str(item.get("soul", "") or "").strip()
        key = str(item.get("key", "") or "").strip()
        return (soul, key) if key else None
    soul = str(getattr(item, "soul", "") or "").strip()
    key = str(getattr(item, "key", "") or "").strip()
    return (soul, key) if key else None


def SoulState(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        rawself = value.get("self")
        rawsouls = list(value.get("souls", []) or [])
    else:
        rawself = getattr(value, "self", None)
        rawsouls = list(getattr(value, "souls", []) or [])
    myself = SoulShape(rawself)
    souls: list[SoulPair] = []
    for item in rawsouls:
        pair = SoulShape(item)
        if pair is not None:
            souls.append(pair)
    if myself is not None and myself not in souls:
        souls.append(myself)
    result = {"self": myself, "souls": souls}
    return result


def SoulSet(cryptstate: Any) -> list[SoulPair]:
    state = SoulState(cryptstate)
    seen = set()
    roster: list[SoulPair] = []
    for soul, key in state["souls"]:
        pair = (str(soul), str(key))
        if pair in seen:
            continue
        seen.add(pair)
        roster.append(pair)
    roster.sort(key=lambda pair: (pair[1], pair[0]))
    return roster


def RootFromSouls(souls: Sequence[SoulPair]) -> str:
    body = "|".join((f"{soul}:{key}" for soul, key in souls))
    root = Sha(body)
    return root


@dataclass(frozen=True)
class SoulCard:
    soul: str
    key: str
    origin: str


class Genesis:

    def __init__(self, cryptstate: Any):
        self.cryptstate = SoulState(cryptstate)
        self.myself = self.cryptstate["self"]
        self.souls = SoulSet(cryptstate)
        self.root = RootFromSouls(self.souls)

    def ShuffleSouls(self) -> list[SoulCard]:
        cards = [SoulCard(soul=soul, key=key, origin="real") for soul, key in self.souls]
        randomizer = Rng(self.root, "souls")
        randomizer.shuffle(cards)
        return cards

    def GeneralTokens(self) -> list[int]:
        tokens = [1, 2, 3, 4]
        randomizer = Rng(self.root, "generaltokens")
        randomizer.shuffle(tokens)
        return tokens


class TheFallen:
    Names: list[str] = [
        "Rigel   ", "Null    ", "Dimon   ", "Nyx     ", "Paradox ", "Hector  ",
        "Yellen  ", "Echo    ", "Marcus  ", "Shadow  ", "Plato   ", "Burry   ",
        "Atlas   ", "Ghost   ", "Lucian  ", "Drift   ", "Powell  ", "Selene  ",
        "Flux    ", "Tesla   ", "AFKLord ", "Cicero  ", "Void    ", "Soros   ",
        "Altair  ", "TryHard ", "Orion   ", "Thales  ", "Alias   ", "Bezos   ",
        "Deneb   ", "Laglord ", "Zeno    ", "Balance ", "Krugman ", "Apollo  ",
        "Vector  ", "Sappho  ", "Respawn ", "Metis   ", "Boreas  ", "Galen   ",
        "Proxy   ", "Draco   ", "Minsky  ", "Castor  ", "Collapse", "Rhea    ",
        "Logos   ", "FragGod ", "Aurora  ", "Friedman", "Hydra   ", "Solon   ",
        "Unknown ", "Iris    ", "NoScope ", "Nemesis ", "Zephyrus", "Hera    ",
        "Pressure", "Ovid    ", "Zucker  ", "Cygnus  ", "Pericles", "Gaia    ",
        "Glitch  ", "Seneca  ", "Sirius  ", "Bernanke", "Leonidas", "Eos     ",
        "Anchor  ", "Animus  ", "Andreas ", "Lagarde ", "Volcker ", "Anima   ",
        "DeadEye ", "Mythos  ", "Vesta   ", "Oblivion", "Ptolemy ", "Janus   ",
        "Thiel   ", "Hemera  ", "Fortuna ", "Spica   ", "Self    ", "Pegasus ",
        "NoName  ", "Aion    ", "Wright  ", "Demeter ", "Minerva ", "Saylor  ",
        "HitScan ", "Antares ", "Juno    ", "Musk    ", "Erebus  ", "Bankman ",
        "Clutch  ", "Melinoe ", "Vega    ", "Default ", "LryJnkns", "GRNSPAN ",
        "Achilles", "Virgil  ", "Keynes  ", "Dalio   ", "Hypatia ", "Boreas  ",
    ]

    def __init__(self, root: str):
        self.root = str(root or "")
        self.randomizer = Rng(self.root, "TheFallen")

    def Take(self, count: int, usedkeys: Sequence[str]) -> list[SoulCard]:
        names = list(self.Names)
        self.randomizer.shuffle(names)
        used = {str(value or "").strip() for value in usedkeys if str(value or "").strip()}
        out: list[SoulCard] = []
        index = 0
        while len(out) < int(count):
            soul = names[index] if index < len(names) else f"fallen{index:02d}"
            key = Sha(f"THEFALLEN|{self.root}|{soul}")
            if key not in used:
                used.add(key)
                out.append(SoulCard(soul=soul, key=key, origin="fallen"))
            index += 1
        return out


class Spoils:

    def __init__(self, totalreserve: int, root: str):
        self.totalreserve = int(totalreserve)
        self.root = str(root or "")

    def Allocate(self, total: int, count: int, label: str, *, lo: float = 1.0, hi: float = 1.7) -> list[int]:
        randomizer = Rng(self.root, f"spoils:{label}")
        weights = [randomizer.uniform(lo, hi) for _ in range(int(count))]
        totalweight = sum(weights)
        raw = [total * (weight / totalweight) for weight in weights]
        ints = [int(value) for value in raw]
        remainder = total - sum(ints)
        if remainder:
            fracs = sorted(((raw[i] - ints[i], i) for i in range(len(ints))), reverse=True)
            for index in range(remainder):
                ints[fracs[index % len(fracs)][1]] += 1
        if sum(ints) != total and ints:
            ints[0] += total - sum(ints)
        return ints

    def AllocateCaptainFile(self, total: int, filetoken: int) -> list[int]:
        randomizer = Rng(self.root, f"captainfile:{int(filetoken)}")
        base = [1.6, 1.3, 1.0, 0.7, 0.4]
        jittered = [weight * randomizer.uniform(0.92, 1.08) for weight in base]
        totalweight = sum(jittered)
        raw = [total * (weight / totalweight) for weight in jittered]
        ints = [int(value) for value in raw]
        remainder = total - sum(ints)
        if remainder:
            fracs = sorted(((raw[i] - ints[i], i) for i in range(len(ints))), reverse=True)
            for index in range(remainder):
                ints[fracs[index % len(fracs)][1]] += 1
        ints.sort(reverse=True)
        if sum(ints) != total:
            ints[0] += total - sum(ints)
        return ints

    def Split(self) -> Tuple[list[int], list[int]]:
        generalpool = self.totalreserve // 2
        captainpool = self.totalreserve - generalpool
        generalsalts = self.Allocate(generalpool, GeneralCount, "general", lo=1.0, hi=1.9)
        captainbaseperfile = captainpool // Files
        captainremainder = captainpool - captainbaseperfile * Files
        captainsalts: list[int] = []
        for filetoken in range(1, Files + 1):
            filetotal = captainbaseperfile + (1 if filetoken <= captainremainder else 0)
            captainsalts.extend(self.AllocateCaptainFile(filetotal, filetoken))
        return (generalsalts, captainsalts)


class Sanctum:

    def __init__(self, dream: Any = None):
        self.dream = dream
        self.state = None

    def ResolveDream(self):
        if self.dream is not None:
            return self.dream
        live = getattr(Dream, "dream", None)
        if live is not None:
            self.dream = live
            return self.dream
        self.dream = Dream.Dream()
        return self.dream

    def Genesis(self, cryptstate: Any):
        self.state = State(cryptstate)
        dream = self.ResolveDream()
        if hasattr(dream, "AcceptState"):
            dream.AcceptState(self.state, publish=True)
        else:
            dream.state = self.state
            if hasattr(dream, "Publish"):
                dream.Publish()
        return self.state


def AssignGeneralRoster(roster: Sequence[SoulCard], generaltokens: Sequence[int]) -> Tuple[list[Tuple[int, SoulCard]], list[SoulCard]]:
    generalroster: list[Tuple[int, SoulCard]] = []
    captainroster: list[SoulCard] = list(roster)
    count = min(len(captainroster), GeneralCount)
    for index in range(count):
        token = int(generaltokens[index])
        card = captainroster.pop(0)
        generalroster.append((token, card))
    return (generalroster, captainroster)


def BackfillGeneralRoster(generalroster: list[Tuple[int, SoulCard]], fallen: TheFallen, usedkeys: Sequence[str]) -> list[Tuple[int, SoulCard]]:
    presenttokens = {token for token, _card in generalroster}
    missingtokens = [token for token in [1, 2, 3, 4] if token not in presenttokens]
    if not missingtokens:
        out = list(sorted(generalroster, key=lambda item: item[0]))
        return out
    extras = fallen.Take(len(missingtokens), usedkeys)
    for token, card in zip(missingtokens, extras):
        generalroster.append((int(token), card))
    out = list(sorted(generalroster, key=lambda item: item[0]))
    return out


def BuildCaptainRoster(captainroster: Sequence[SoulCard], fallen: TheFallen, usedkeys: Sequence[str]) -> list[SoulCard]:
    out = list(captainroster)
    if len(out) < CaptainCount:
        extras = fallen.Take(CaptainCount - len(out), usedkeys)
        out.extend(extras)
    out = out[:CaptainCount]
    return out


def BuildCells(
    generalroster: Sequence[Tuple[int, SoulCard]],
    captainroster: Sequence[SoulCard],
    generalsalts: Sequence[int],
    captainsalts: Sequence[int],
) -> Tuple[Field.Cell, ...]:
    cells: list[Field.Cell] = []
    captainindex = 0
    generalsbytoken = {int(token): card for token, card in generalroster}
    for filetoken in range(1, Files + 1):
        generalcard = generalsbytoken[filetoken]
        generalsalt = int(generalsalts[filetoken - 1])
        cells.append(
            Field.Cell(
                soul=str(generalcard.soul),
                key=str(generalcard.key),
                salt=int(generalsalt),
                purge=Field.Clean.purge(),
                lock=Field.Clean.lock(),
                sign=Field.Clean.sign(),
            )
        )
        for _rank in range(2, Ranks + 1):
            card = captainroster[captainindex]
            salt = int(captainsalts[captainindex])
            captainindex += 1
            cells.append(
                Field.Cell(
                    soul=str(card.soul),
                    key=str(card.key),
                    salt=int(salt),
                    purge=Field.Clean.purge(),
                    lock=Field.Clean.lock(),
                    sign=Field.Clean.sign(),
                )
            )
    return tuple(cells)


def State(cryptstate: Any) -> Field.State:
    genesis = Genesis(cryptstate)
    root = genesis.root
    roster = genesis.ShuffleSouls()
    generaltokens = genesis.GeneralTokens()
    generalroster, captainroster = AssignGeneralRoster(roster, generaltokens)
    usedkeys = [card.key for card in roster]
    fallen = TheFallen(root)
    generalroster = BackfillGeneralRoster(generalroster, fallen, usedkeys)
    usedkeys.extend((card.key for _token, card in generalroster if card.key not in usedkeys))
    captainroster = BuildCaptainRoster(captainroster, fallen, usedkeys)
    spoils = Spoils(totalreserve=Reserve, root=root)
    generalsalts, captainsalts = spoils.Split()
    cells = BuildCells(
        generalroster=generalroster,
        captainroster=captainroster,
        generalsalts=generalsalts,
        captainsalts=captainsalts,
    )
    finalstate = Field.State(self=genesis.myself or Field.Clean.self(), monument=(), cells=cells)
    return finalstate
