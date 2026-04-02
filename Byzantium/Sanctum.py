from __future__ import annotations
import hashlib
import json
import os
import random
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
import Field
import Dream
files = 4
ranks = 6
cellCount = files * ranks
generalCount = files
captainCount = cellCount - generalCount
captainsPerFile = ranks - 1
reserve = 1000000
SoulPair = Tuple[str, str]

def sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def rng(root: str, label: str) -> random.Random:
    seed = sha(f'{root}|{label}')
    return random.Random(int(seed, 16))

def coerceSoul(item: Any) -> Optional[SoulPair]:
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        soul = str(item[0] or '').strip()
        key = str(item[1] or '').strip()
        return (soul, key) if key else None
    if isinstance(item, dict):
        soul = str(item.get('soul', '') or '').strip()
        key = str(item.get('key', '') or '').strip()
        return (soul, key) if key else None
    soul = str(getattr(item, 'soul', '') or '').strip()
    key = str(getattr(item, 'key', '') or '').strip()
    return (soul, key) if key else None

def coerceCryptState(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        rawself = value.get('self')
        rawsouls = list(value.get('souls', []) or [])
    else:
        rawself = getattr(value, 'self', None)
        rawsouls = list(getattr(value, 'souls', []) or [])
    myself = coerceSoul(rawself)
    souls: List[SoulPair] = []
    for item in rawsouls:
        pair = coerceSoul(item)
        if pair is not None:
            souls.append(pair)
    else:
        pass
    if myself is not None and myself not in souls:
        souls.append(myself)
    result = {'self': myself, 'souls': souls}
    return result

def canonicalSouls(cryptState: Any) -> List[SoulPair]:
    state = coerceCryptState(cryptState)
    seen = set()
    roster: List[SoulPair] = []
    for soul, key in state['souls']:
        pair = (str(soul), str(key))
        if pair in seen:
            continue
        seen.add(pair)
        roster.append(pair)
    roster.sort(key=lambda pair: (pair[1], pair[0]))
    return roster

def rootFromSouls(souls: Sequence[SoulPair]) -> str:
    body = '|'.join((f'{soul}:{key}' for soul, key in souls))
    root = sha(body)
    return root

@dataclass(frozen=True)
class SoulCard:
    soul: str
    key: str
    origin: str

class Genesis:

    def __init__(self, cryptState: Any):
        self.cryptState = coerceCryptState(cryptState)
        self.myself = self.cryptState['self']
        self.souls = canonicalSouls(cryptState)
        self.root = rootFromSouls(self.souls)

    def shuffleSouls(self) -> List[SoulCard]:
        cards = [SoulCard(soul=soul, key=key, origin='real') for soul, key in self.souls]
        r = rng(self.root, 'souls')
        r.shuffle(cards)
        return cards

    def generalTokens(self) -> List[int]:
        tokens = [1, 2, 3, 4]
        r = rng(self.root, 'generalTokens')
        r.shuffle(tokens)
        return tokens

class TheFallen:
    names: List[str] = [
        'Rigel   ', 'Null    ', 'Dimon   ', 'Nyx     ', 'Paradox ', 'Hector  ',
        'Yellen  ', 'Echo    ', 'Marcus  ', 'Shadow  ', 'Plato   ', 'Burry   ',
        'Atlas   ', 'Ghost   ', 'Lucian  ', 'Drift   ', 'Powell  ', 'Selene  ',
        'Flux    ', 'Tesla   ', 'AFKLord ', 'Cicero  ', 'Void    ', 'Soros   ',
        'Altair  ', 'TryHard ', 'Orion   ', 'Thales  ', 'Alias   ', 'Bezos   ',
        'Deneb   ', 'Laglord ', 'Zeno    ', 'Balance ', 'Krugman ', 'Apollo  ',
        'Vector  ', 'Sappho  ', 'Respawn ', 'Metis   ', 'Boreas  ', 'Galen   ',
        'Proxy   ', 'Draco   ', 'Minsky  ', 'Castor  ', 'Collapse', 'Rhea    ',
        'Logos   ', 'FragGod ', 'Aurora  ', 'Friedman', 'Hydra   ', 'Solon   ',
        'Unknown ', 'Iris    ', 'NoScope ', 'Nemesis ', 'Zephyrus', 'Hera    ',
        'Pressure', 'Ovid    ', 'Zucker  ', 'Cygnus  ', 'Pericles', 'Gaia    ',
        'Glitch  ', 'Seneca  ', 'Sirius  ', 'Bernanke', 'Leonidas', 'Eos     ',
        'Anchor  ', 'Animus  ', 'Andreas ', 'Lagarde ', 'Volcker ', 'Anima   ',
        'DeadEye ', 'Mythos  ', 'Vesta   ', 'Oblivion', 'Ptolemy ', 'Janus   ',
        'Thiel   ', 'Hemera  ', 'Fortuna ', 'Spica   ', 'Self    ', 'Pegasus ',
        'NoName  ', 'Aion    ', 'Wright  ', 'Demeter ', 'Minerva ', 'Saylor  ',
        'HitScan ', 'Antares ', 'Juno    ', 'Musk    ', 'Erebus  ', 'Bankman ',
        'Clutch  ', 'Melinoe ', 'Vega    ', 'Default ', 'LryJnkns', 'GRNSPAN ',
        'Achilles', 'Virgil  ', 'Keynes  ', 'Dalio   ', 'Hypatia ', 'Boreas  ',
    ]

    def __init__(self, root: str):
        self.root = str(root or '')
        self.r = rng(self.root, 'TheFallen')

    def take(self, count: int, usedKeys: Sequence[str]) -> List[SoulCard]:
        names = list(self.names)
        self.r.shuffle(names)
        used = {str(x or '').strip() for x in usedKeys if str(x or '').strip()}
        out: List[SoulCard] = []
        index = 0
        while len(out) < int(count):
            soul = names[index] if index < len(names) else f'FALLEN{index:02d}'
            key = sha(f'THEFALLEN|{self.root}|{soul}')
            if key not in used:
                used.add(key)
                out.append(SoulCard(soul=soul, key=key, origin='fallen'))
            index += 1
        return out

class Spoils:

    def __init__(self, totalReserve: int, root: str):
        self.totalReserve = int(totalReserve)
        self.root = str(root or '')

    def allocate(self, total: int, count: int, label: str, *, lo: float=1.0, hi: float=1.7) -> List[int]:
        r = rng(self.root, f'spoils:{label}')
        weights = [r.uniform(lo, hi) for _ in range(int(count))]
        totalWeight = sum(weights)
        raw = [total * (weight / totalWeight) for weight in weights]
        ints = [int(value) for value in raw]
        remainder = total - sum(ints)
        if remainder:
            fracs = sorted(((raw[i] - ints[i], i) for i in range(len(ints))), reverse=True)
            for i in range(remainder):
                ints[fracs[i % len(fracs)][1]] += 1
        else:
            pass
        if sum(ints) != total and ints:
            ints[0] += total - sum(ints)
        return ints

    def allocateCaptainFile(self, total: int, fileToken: int) -> List[int]:
        r = rng(self.root, f'captainFile:{int(fileToken)}')
        base = [1.6, 1.3, 1.0, 0.7, 0.4]
        jittered = [w * r.uniform(0.92, 1.08) for w in base]
        totalWeight = sum(jittered)
        raw = [total * (weight / totalWeight) for weight in jittered]
        ints = [int(value) for value in raw]
        remainder = total - sum(ints)
        if remainder:
            fracs = sorted(((raw[i] - ints[i], i) for i in range(len(ints))), reverse=True)
            for i in range(remainder):
                ints[fracs[i % len(fracs)][1]] += 1
        else:
            pass
        ints.sort(reverse=True)
        if sum(ints) != total:
            ints[0] += total - sum(ints)
        return ints

    def split(self) -> Tuple[List[int], List[int]]:
        generalPool = self.totalReserve // 2
        captainPool = self.totalReserve - generalPool
        generalSalts = self.allocate(generalPool, generalCount, 'general', lo=1.0, hi=1.9)
        captainBasePerFile = captainPool // files
        captainRemainder = captainPool - captainBasePerFile * files
        captainSalts: List[int] = []
        for fileToken in range(1, files + 1):
            fileTotal = captainBasePerFile + (1 if fileToken <= captainRemainder else 0)
            captainSalts.extend(self.allocateCaptainFile(fileTotal, fileToken))
        return (generalSalts, captainSalts)

class Sanctum:

    def __init__(self, dream: Any=None):
        self.dream = dream
        self.state = None

    def resolveDream(self):
        if self.dream is not None:
            return self.dream
        live = getattr(Dream, 'dream', None)
        if live is not None:
            self.dream = live
            return self.dream
        live = getattr(Dream, '_DREAM', None)
        if live is not None:
            self.dream = live
            return self.dream
        raise AttributeError('Sanctum could not resolve Dream')

    def genesis(self, cryptState: Any):
        try:
            self.state = state(cryptState)
            dream = self.resolveDream()
            if hasattr(dream, 'acceptstate'):
                dream.acceptstate(self.state, publish=True)
                publish_mode = 'acceptstate'
            else:
                dream.state = self.state
                if hasattr(dream, 'publish'):
                    dream.publish()
                    publish_mode = 'publish'
                else:
                    publish_mode = 'assign_only'
            return self.state
        except Exception as exc:
            raise

def assignGeneralRoster(roster: Sequence[SoulCard], generalTokens: Sequence[int]) -> Tuple[List[Tuple[int, SoulCard]], List[SoulCard]]:
    generalRoster: List[Tuple[int, SoulCard]] = []
    captainRoster: List[SoulCard] = list(roster)
    count = min(len(captainRoster), generalCount)
    for i in range(count):
        token = int(generalTokens[i])
        card = captainRoster.pop(0)
        generalRoster.append((token, card))
    return (generalRoster, captainRoster)

def backfillGeneralRoster(generalRoster: List[Tuple[int, SoulCard]], fallen: TheFallen, usedKeys: Sequence[str]) -> List[Tuple[int, SoulCard]]:
    presentTokens = {token for token, _card in generalRoster}
    missingTokens = [token for token in [1, 2, 3, 4] if token not in presentTokens]
    if not missingTokens:
        out = list(sorted(generalRoster, key=lambda item: item[0]))
        return out
    extras = fallen.take(len(missingTokens), usedKeys)
    for token, card in zip(missingTokens, extras):
        generalRoster.append((int(token), card))
    out = list(sorted(generalRoster, key=lambda item: item[0]))
    return out

def buildCaptainRoster(captainRoster: Sequence[SoulCard], fallen: TheFallen, usedKeys: Sequence[str]) -> List[SoulCard]:
    out = list(captainRoster)
    if len(out) < captainCount:
        extras = fallen.take(captainCount - len(out), usedKeys)
        out.extend(extras)
    out = out[:captainCount]
    return out

def buildCells(generalRoster: Sequence[Tuple[int, SoulCard]], captainRoster: Sequence[SoulCard], generalSalts: Sequence[int], captainSalts: Sequence[int]) -> Tuple[Field.Cell, ...]:
    cells: List[Field.Cell] = []
    captainIndex = 0
    generalsByToken = {int(token): card for token, card in generalRoster}
    for fileToken in range(1, files + 1):
        generalCard = generalsByToken[fileToken]
        generalSalt = int(generalSalts[fileToken - 1])
        cells.append(Field.Cell(soul=str(generalCard.soul), key=str(generalCard.key), salt=int(generalSalt), purge=Field.Clean.purge(), lock=Field.Clean.lock(), sign=Field.Clean.sign()))
        for _rank in range(2, ranks + 1):
            card = captainRoster[captainIndex]
            salt = int(captainSalts[captainIndex])
            captainIndex += 1
            cells.append(Field.Cell(soul=str(card.soul), key=str(card.key), salt=int(salt), purge=Field.Clean.purge(), lock=Field.Clean.lock(), sign=Field.Clean.sign()))
    else:
        pass
    return tuple(cells)

def state(cryptState: Any):
    genesis = Genesis(cryptState)
    root = genesis.root
    roster = genesis.shuffleSouls()
    generalTokens = genesis.generalTokens()
    generalRoster, captainRoster = assignGeneralRoster(roster, generalTokens)
    usedKeys = [card.key for card in roster]
    fallen = TheFallen(root)
    generalRoster = backfillGeneralRoster(generalRoster, fallen, usedKeys)
    usedKeys.extend((card.key for _token, card in generalRoster if card.key not in usedKeys))
    captainRoster = buildCaptainRoster(captainRoster, fallen, usedKeys)
    spoils = Spoils(totalReserve=reserve, root=root)
    generalSalts, captainSalts = spoils.split()
    cells = buildCells(generalRoster=generalRoster, captainRoster=captainRoster, generalSalts=generalSalts, captainSalts=captainSalts)
    final_state = Field.State(self=genesis.myself or Field.Clean.self(), monument=(), cells=cells)
    return final_state
__all__ = ['files', 'ranks', 'cellCount', 'generalCount', 'captainCount', 'captainsPerFile', 'reserve', 'SoulPair', 'SoulCard', 'Genesis', 'TheFallen', 'Spoils', 'Sanctum', 'LOG_PATH', 'coerceSoul', 'coerceCryptState', 'canonicalSouls', 'rootFromSouls', 'assignGeneralRoster', 'backfillGeneralRoster', 'buildCaptainRoster', 'buildCells', 'state']
