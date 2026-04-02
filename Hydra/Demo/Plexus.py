# ============================================
# Plexus (Heart) — Truth Through Erasure
# No time. No replay. No logs.
# ============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Literal

# =========================
# Crown Membrane (3-gem plane)
# =========================
GEMS = {1: "ONYX", 2: "JADE", 3: "OPAL"}

def gem_name(g: int) -> str:
    return GEMS.get(int(g or 1), "G?")

def crown_next(c: int) -> int:
    return 1 if int(c) >= 3 else int(c) + 1

# =========================
# Intent (Body Interface)
# =========================
IntentType = Literal["PROPAGATE", "REQUEST_SYNC", "ENVY"]

@dataclass(frozen=True)
class Intent:
    type: IntentType
    payload: Dict[str, Any]

# =========================
# Tetron (Crystalline Soul Seat)
# =========================
@dataclass
class Tetron:
    tallies: Dict[str, int]

    def snapshot(self) -> Dict[str, Any]:
        # Dream geometry: no authority, no memory
        return {
            "tallies": dict(self.tallies),
            "is_dream": True,
        }

# =========================
# Plexus State
# =========================
@dataclass
class plexusState:
    tallies: Dict[str, int]
    crown: int
    head: Optional[str] = None

# =========================
# Plexus (Heart)
# =========================
class plexus:
    def __init__(
        self,
        head_id: str,
        initial_tallies: Optional[Dict[str, int]] = None,
        initial_crown: int = 1,
    ) -> None:
        self.head_id = str(head_id)

        tallies = dict(initial_tallies or {
            "A": 10, "B": 10, "C": 10, "D": 10, "E": 10
        })

        # Tetron = crystalline soul geometry
        self.tetron = Tetron(tallies=tallies)

        # Embodiment
        self.state = plexusState(
            tallies=dict(tallies),
            crown=int(initial_crown),
            head=None,
        )

        self.tail: Optional[Dict[str, Any]] = None
        self.envy: bool = False

    # =========================
    # Echo
    # =========================
    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.head_id,
            "tallies": dict(self.state.tallies),
            "crown": int(self.state.crown),
            "is_dream": False,
        }

    def emotions(self) -> Dict[str, Any]:
        return {"envy": bool(self.envy)}

    # =========================
    # DreamState (Tetron Projection)
    # =========================
    def dream_state(self) -> Dict[str, Any]:
        d = self.tetron.snapshot()
        # Crown is pacing only; never authority
        d["crown"] = int(self.state.crown)
        return d

    # =========================
    # Envy
    # =========================
    def _envy_reanchor(self) -> Dict[str, Any]:
        # Drop authority, redraw identity, do nothing else
        return {
            "id": self.head_id,
            "tallies": dict(self.tetron.tallies),
            "crown": int(self.state.crown),
            "is_dream": True,
            "mode": "ENVY",
        }

    # =========================
    # Proposal
    # =========================
    def propose(self, to_head: str, amount: int) -> Dict[str, Any]:
        if self.envy:
            return self._envy_reanchor()

        tallies = dict(self.state.tallies)
        tallies[self.head_id] = tallies.get(self.head_id, 0) - int(amount)
        tallies[to_head] = tallies.get(to_head, 0) + int(amount)

        return {
            "id": self.head_id,
            "tallies": tallies,
            "crown": crown_next(self.state.crown),
        }

    # =========================
    # Ingest
    # =========================
    def ingest(self, tail_in: Dict[str, Any]) -> List[Intent]:
        intents: List[Intent] = []

        # Dream is orientation AND hydration (never a competitor)
        if tail_in.get("is_dream"):
            dream_tallies = dict(tail_in.get("tallies", {}) or {})

            # Idempotent hydration
            if dream_tallies and dream_tallies != self.state.tallies:
                self.state.tallies = dict(dream_tallies)
                self.tetron.tallies = dict(dream_tallies)

            # Envy resolves through stillness
            if self.envy:
                self.envy = False

            return intents

        inc_tallies = dict(tail_in.get("tallies", {}))
        inc_crown = int(tail_in.get("crown", self.state.crown))

        cur = int(self.state.crown)
        exp = crown_next(cur)

        # Envy gate
        if inc_crown not in (cur, exp):
            if not self.envy:
                self.envy = True
                intents.append(Intent("ENVY", {
                    "current_crown": cur,
                    "incoming_crown": inc_crown,
                }))

            intents.append(Intent("REQUEST_SYNC", {
                "crown": cur,
                "gem": gem_name(cur),
            }))
            return intents

        # Calm again
        if self.envy:
            self.envy = False

        # Idempotent no-op
        if inc_tallies == self.state.tallies:
            return intents

        # Adopt witnessed reality
        self.state.tallies = dict(inc_tallies)
        self.state.crown = inc_crown
        self.state.head = str(tail_in.get("id", "")) or None

        # Stabilize the Tetron
        self.tetron.tallies = dict(inc_tallies)

        self.tail = dict(tail_in)

        intents.append(Intent("PROPAGATE", {"tail": dict(self.tail)}))

        if inc_crown == exp:
            intents.append(Intent("REQUEST_SYNC", {
                "crown": inc_crown,
                "gem": gem_name(inc_crown),
            }))

        return intents
