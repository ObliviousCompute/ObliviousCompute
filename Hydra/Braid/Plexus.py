from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

Gems = {1: "Onyx", 2: "Jade", 3: "Opal"}

def GemName(g: int) -> str:
    return Gems.get(int(g or 1), "G?")

def CrownNext(c: int) -> int:
    return 1 if int(c) >= 3 else int(c) + 1

IntentType = Literal["Propagate", "RequestSync", "Envy"]

@dataclass(frozen=True)
class Intent:
    type: IntentType
    payload: Dict[str, Any]

@dataclass
class Tetron:
    tallies: Dict[str, int]

    def Snapshot(self) -> Dict[str, Any]:
        return {
            "tallies": dict(self.tallies),
            "is_dream": True,
        }

@dataclass
class PlexusState:
    tallies: Dict[str, int]
    crown: int
    head: Optional[str] = None

class Plexus:
    def __init__(self, head: str, heads: List[str]) -> None:
        self.head = str(head)
        self.heads = list(heads)

        tallies = {h: 10 for h in self.heads}

        self.tetron = Tetron(tallies=dict(tallies))
        self.state = PlexusState(
            tallies=dict(tallies),
            crown=1,
            head=None,
        )

        self.tail: Optional[Dict[str, Any]] = None
        self.envy: bool = False

    def Snapshot(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "tallies": dict(self.state.tallies),
            "crown": int(self.state.crown),
            "is_dream": False,
        }

    def Emotions(self) -> Dict[str, Any]:
        return {"envy": bool(self.envy)}

    def DreamState(self) -> Dict[str, Any]:
        d = self.tetron.Snapshot()
        d["crown"] = int(self.state.crown)
        return d

    def EnvyReanchor(self) -> Dict[str, Any]:
        return {
            "head": self.head,
            "tallies": dict(self.tetron.tallies),
            "crown": int(self.state.crown),
            "is_dream": True,
            "mode": "Envy",
        }

    def ExpectedTotal(self) -> int:
        return 10 * len(self.heads)

    def TalliesTotal(self, tallies: Dict[str, Any]) -> int:
        return sum(int(value or 0) for value in dict(tallies or {}).values())

    def ValidTotal(self, tallies: Dict[str, Any]) -> bool:
        return self.TalliesTotal(tallies) == self.ExpectedTotal()

    def Propose(self, tohead: str, amount: int) -> Dict[str, Any]:
        if self.envy:
            return self.EnvyReanchor()

        tallies = dict(self.state.tallies)
        tallies[self.head] = tallies.get(self.head, 0) - int(amount)
        tallies[tohead] = tallies.get(tohead, 0) + int(amount)

        return {
            "head": self.head,
            "tallies": tallies,
            "crown": CrownNext(self.state.crown),
        }

    def Ingest(self, tailin: Dict[str, Any]) -> List[Intent]:
        intents: List[Intent] = []

        if tailin.get("is_dream"):
            dreamtallies = dict(tailin.get("tallies", {}) or {})

            # ================= LINCHPIN ================= #
            if dreamtallies and not self.ValidTotal(dreamtallies):
            # ============================================ #
                if not self.envy:
                    self.envy = True
                    intents.append(Intent("Envy", {
                        "expectedtotal": self.ExpectedTotal(),
                        "incomingtotal": self.TalliesTotal(dreamtallies),
                    }))

                intents.append(Intent("RequestSync", {
                    "crown": int(self.state.crown),
                    "gem": GemName(self.state.crown),
                    "needtail": True,
                }))
                return intents

            if dreamtallies and dreamtallies != self.state.tallies:
                self.state.tallies = dict(dreamtallies)
                self.tetron.tallies = dict(dreamtallies)

            if self.envy:
                self.envy = False

            return intents

        incomingtallies = dict(tailin.get("tallies", {}))
        inccrown = int(tailin.get("crown", self.state.crown))

        cur = int(self.state.crown)
        exp = CrownNext(cur)

        # ================= LINCHPIN ================= #
        if inccrown not in (cur, exp):
        # ============================================ #
            if not self.envy:
                self.envy = True
                intents.append(Intent("Envy", {
                    "currentcrown": cur,
                    "incomingcrown": inccrown,
                }))

            intents.append(Intent("RequestSync", {
                "crown": cur,
                "gem": GemName(cur),
            }))
            return intents

        # ================= LINCHPIN ================= #
        if not self.ValidTotal(incomingtallies):
        # ============================================ #
            if not self.envy:
                self.envy = True
                intents.append(Intent("Envy", {
                    "expectedtotal": self.ExpectedTotal(),
                    "incomingtotal": self.TalliesTotal(incomingtallies),
                }))

            intents.append(Intent("RequestSync", {
                "crown": cur,
                "gem": GemName(cur),
                "needtail": True,
            }))
            return intents

        if self.envy:
            self.envy = False

        if incomingtallies == self.state.tallies:
            return intents

        self.state.tallies = dict(incomingtallies)
        self.state.crown = inccrown
        self.state.head = str(tailin.get("head", "")) or None

        self.tetron.tallies = dict(incomingtallies)
        self.tail = dict(tailin)

        intents.append(Intent("Propagate", {"tail": dict(self.tail)}))

        if inccrown == exp:
            intents.append(Intent("RequestSync", {
                "crown": inccrown,
                "gem": GemName(inccrown),
            }))

        return intents
