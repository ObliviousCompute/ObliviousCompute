# ==============================================
# ObliviousHeart v0.1 — Truth Through Erasure
# No time. No replay. No logs.
# ==============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

Proof = Dict[str, Any]
ROCK, PAPER, SCISSORS = 1, 2, 3

def NextRPS(rps: int) -> int:
    rps = int(rps or ROCK)
    return ROCK if rps >= SCISSORS else rps + 1

def RPSName(rps: int) -> str:
    names = {ROCK: "ROCK", PAPER: "PAPER", SCISSORS: "SCISSORS"}
    return names.get(int(rps), "RPS?")

@dataclass(frozen=True)
class Intent:
    type: str
    payload: Dict[str, Any]

@dataclass
class ProofState:
    tallies: Dict[str, int]
    rps: int
    head: Optional[str] = None

class ObliviousHeart:
    def __init__(self, node_id: str,
                 initial_tallies: Optional[Dict[str, int]] = None,
                 initial_rps: int = ROCK) -> None:
        self.node_id = str(node_id)
        tallies = dict(initial_tallies or {
            "A": 10, "B": 10, "C": 10, "D": 10, "E": 10
        })
        self.state = ProofState(
            tallies=dict(tallies), rps=int(initial_rps), head=None
        )
        self.best: Proof = {
            "id": self.node_id, "tallies": dict(tallies),
            "rps": int(initial_rps)
        }
        self.envy = False

    def snapshot(self) -> Proof:
        return {"id": self.node_id, "tallies": dict(self.state.tallies),
                "rps": int(self.state.rps), "is_dream": False}

    def emotions(self) -> Dict[str, Any]:
        return {"envy": bool(self.envy)}

    def seed_proof(self) -> Proof:
        return {"id": self.node_id,
                "tallies": dict(self.best.get("tallies", self.state.tallies)),
                "rps": int(self.state.rps), "is_seed": True}

    def _envy_reanchor(self) -> Proof:
        return {"id": self.node_id,
                "tallies": dict(self.best.get("tallies", self.state.tallies)),
                "rps": int(self.state.rps), "is_dream": True, "mode": "ENVY"}

    def propose(self, to_node: str, amount: int) -> Proof:
        if self.envy:
            return self._envy_reanchor()
        frm, to, amt = self.node_id, str(to_node), int(amount)
        tallies = dict(self.state.tallies)
        tallies[frm] = tallies.get(frm, 0) - amt
        tallies[to] = tallies.get(to, 0) + amt
        return {"id": self.node_id, "tallies": tallies,
                "rps": NextRPS(self.state.rps)}

    def ingest(self, incoming: Proof) -> List[Intent]:
        intents: List[Intent] = []
        if not isinstance(incoming, dict):
            return intents
        rps_in = int(incoming.get("rps", incoming.get("crown", self.state.rps)))
        tallies_in = dict(incoming.get("tallies", {}) or {})
        head_in = str(incoming.get("id", "")) or None
        if incoming.get("is_seed") or incoming.get("is_dream") or incoming.get(
            "is_snapshot"
        ):
            if tallies_in and tallies_in != self.state.tallies:
                self.state.tallies = dict(tallies_in)
                self.best = {"id": head_in or self.node_id,
                             "tallies": dict(tallies_in),
                             "rps": int(self.state.rps)}
            if self.envy:
                self.envy = False
            return intents
        rps_cur = int(self.state.rps or ROCK)
        rps_next = NextRPS(rps_cur)
        if rps_in not in (rps_cur, rps_next):
               # ↑↑↑↑↑ LinchPin ↑↑↑↑↑
            if not self.envy:
                self.envy = True
                intents.append(Intent("ENVY", {"current": RPSName(rps_cur),
                                              "incoming": RPSName(rps_in)}))
            intents.append(Intent("REQUEST_SYNC",
                                 {"rps": rps_cur, "label": RPSName(rps_cur)}))
            return intents
        if self.envy:
            self.envy = False
        if tallies_in == self.state.tallies:
            return intents
        self.state = ProofState(
            tallies=dict(tallies_in), rps=int(rps_in), head=head_in
        )
        self.best = {"id": head_in or self.node_id,
                     "tallies": dict(tallies_in), "rps": int(rps_in)}
        intents.append(Intent("PROPAGATE", {"proof": dict(self.best)}))
        if rps_in == rps_next:
            intents.append(Intent("REQUEST_SYNC",
                                 {"rps": rps_in, "label": RPSName(rps_in)}))
        return intents
