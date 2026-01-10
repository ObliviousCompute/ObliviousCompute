# ============================================
# Oblivious Heart v0 — Truth Through Erasure
# No time. No replay. No logs.
# ============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import hashlib, json

Proof = Dict[str, Any]

# --- Rock / Paper / Scissors partition (1→2→3→1) ---
ROCK, PAPER, SCISSORS = 1, 2, 3

def NextRPS(rps: int) -> int:
    rps = int(rps or ROCK)
    return ROCK if rps >= SCISSORS else rps + 1

def RPSName(rps: int) -> str:
    return {ROCK: "ROCK", PAPER: "PAPER", SCISSORS: "SCISSORS"}.get(int(rps), "RPS?")

# --- Hash / Weight ---
def StateHash(tallies: Dict[str, int], rps: int) -> str:
    b = json.dumps({"tallies": dict(tallies), "rps": int(rps)},
                   sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b).hexdigest()[:12]

def Weight(h: str) -> int:
    return int(h, 16)

def Canonical(proof: Proof) -> Proof:
    tallies = dict(proof.get("tallies", {}) or {})
    rps = int(proof.get("rps", proof.get("crown", ROCK)))  # accept either key
    h = StateHash(tallies, rps)
    p = dict(proof)
    p.update({"tallies": tallies, "rps": rps, "h": h, "weight": Weight(h)})
    return p

# --- Dominance ---
def DomKey(p: Proof) -> Tuple[int, str, str]:
    # Same ordering as your original: (weight, id, h)
    return (int(p.get("weight", 0)), str(p.get("id", "")), str(p.get("h", "")))

# --- Intent / State ---
@dataclass(frozen=True)
class Intent:
    type: str
    payload: Dict[str, Any]

@dataclass
class ProofState:
    tallies: Dict[str, int]
    rps: int
    h: Optional[str] = None
    head: Optional[str] = None

class ObliviousHeart:
    def __init__(self, node_id: str, initial_tallies: Optional[Dict[str, int]] = None, initial_rps: int = ROCK):
        self.node_id = str(node_id)
        self.state = ProofState(
            tallies=dict(initial_tallies or {"A": 10, "B": 10, "C": 10, "D": 10, "E": 10}),
            rps=int(initial_rps),
            h=None,
            head=None,
        )
        self.best: Optional[Proof] = None
        self.envy: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return {"tallies": dict(self.state.tallies), "rps": int(self.state.rps), "h": self.state.h, "head": self.state.head}

    def emotions(self) -> Dict[str, Any]:
        return {"envy": bool(self.envy)}

    def propose(self, to_node: str, amount: int) -> Proof:
        frm, to, amt = self.node_id, str(to_node), int(amount)
        tallies = dict(self.state.tallies)
        tallies[frm] = tallies.get(frm, 0) - amt
        tallies[to] = tallies.get(to, 0) + amt
        rps = NextRPS(self.state.rps)
        h = StateHash(tallies, rps)
        return {"id": self.node_id, "tallies": tallies, "rps": rps, "h": h, "weight": Weight(h)}

    def _incumbent_fallback(self) -> Proof:
        tallies = dict(self.state.tallies)
        rps = int(self.state.rps)
        h = StateHash(tallies, rps)
        return {"id": (self.state.head or self.node_id), "tallies": tallies, "rps": rps, "h": h, "weight": Weight(h)}

    def ingest(self, incoming: Proof) -> List[Intent]:
        intents: List[Intent] = []
        p = Canonical(incoming)

        t_in = dict(p["tallies"])
        rps_in = int(p["rps"])
        h_in = str(p["h"])
        id_in = str(p.get("id", "")) or None

        # Dedup
        if h_in and h_in == self.state.h:
            return intents

        # Bootstrap
        if self.state.h is None:
            self.state = ProofState(t_in, rps_in, h_in, id_in)
            self.best = dict(p); self.best["h"] = self.state.h
            self.envy = False
            intents.append(Intent("PROPAGATE", {"proof": dict(self.best)}))
            intents.append(Intent("SYNC_REQUEST", {"rps": self.state.rps, "need_proof": False, "label": RPSName(self.state.rps)}))
            return intents

        rps_cur = int(self.state.rps or ROCK)
        rps_next = NextRPS(rps_cur)

        # RPS Gate: only CURRENT or NEXT is admissible
        if rps_in not in (rps_cur, rps_next):
            self.envy = True
            intents.append(Intent("ENVY", {"current": RPSName(rps_cur), "incoming": RPSName(rps_in)}))
            intents.append(Intent("SYNC_REQUEST", {"rps": rps_cur, "need_proof": True, "label": RPSName(rps_cur)}))
            return intents

        incumbent = self.best or self._incumbent_fallback()

        # LINCHPIN:
        adopt = (rps_in, DomKey(p)) > (rps_cur, DomKey(incumbent))

        if not adopt:
            return intents

        advanced = (rps_in == rps_next)
        self.state = ProofState(t_in, rps_in, h_in, id_in)
        self.best = dict(p); self.best["h"] = self.state.h
        intents.append(Intent("PROPAGATE", {"proof": dict(self.best)}))
        if advanced:
            intents.append(Intent("SYNC_REQUEST", {"rps": rps_in, "need_proof": False, "label": RPSName(rps_in)}))
        return intents

