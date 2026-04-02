# ============================================
# Hydra (Soul) — Truth Through Erasure
# No time. No replay. No logs.
# ============================================
from __future__ import annotations
import argparse, random
from typing import List, Tuple
from Plexus import plexus, gem_name
from Body import run_body, parse_peer

HEADS_DEFAULT = ["A", "B", "C", "D", "E"]


def _run_loopback(heads: List[str], steps: int, seed: int) -> None:
    rng = random.Random(seed)
    heads = [h.upper() for h in heads]

    nodes = {h: plexus(head_id=h) for h in heads}

    base = next(iter(nodes.values())).snapshot()
    for n in nodes.values():
        n.state.tallies = dict(base["tallies"])
        n.state.crown = int(base["crown"] or 1)

    for _ in range(max(0, int(steps))):
        frm = rng.choice(heads)
        to = rng.choice([h for h in heads if h != frm])
        amt = rng.randint(1, 3)

        tail = nodes[frm].propose(to, amt)

        for n in nodes.values():
            n.ingest(dict(tail))

    print("\nLoopback results (no UDP):\n")
    for h in heads:
        snap = nodes[h].snapshot()
        print(f"{h}: crown={snap['crown']}({gem_name(snap['crown'])}) tallies={snap['tallies']}")
    print("")


def main() -> None:
    ap = argparse.ArgumentParser(prog="Hydra.py")
    ap.add_argument("--id")
    ap.add_argument("--port", type=int)
    ap.add_argument("--peers", nargs="*", default=[])

    ap.add_argument("--loopback", action="store_true")
    ap.add_argument("--heads", nargs="*", default=HEADS_DEFAULT)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    if args.loopback:
        _run_loopback(list(args.heads), args.steps, args.seed)
        return

    if not args.id or args.port is None:
        ap.error("UDP run requires --id and --port (or use --loopback)")

    head_id = str(args.id).upper()
    port = int(args.port)
    peers: List[Tuple[str, int]] = [parse_peer(p) for p in args.peers]

    heart = plexus(head_id=head_id)
    run_body(heart=heart, head_id=head_id, port=port, peers=peers)


if __name__ == "__main__":
    main()
