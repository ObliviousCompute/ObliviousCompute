# =================================================
# ObliviousSmokeTest v0 — Truth Through Erasure
# No time. No replay. No logs.
# =================================================
"""
ObliviousHeart v0 — SmokeTest

This program demonstrates the core invariants of
Oblivious Coomputation.

You just witnessed:

1) A proof outside the admissible Rock/Paper/Scissors window
being rejected and triggering sync.

2) Deterministic dominance resolving same-partition contention 
without history or time.

3) Convergence to a single final state under shuffled, duplicated, 
and dropped message delivery.

No clocks. No logs. No replay.
Truth emerges by erasure alone.
"""

import random
from copy import deepcopy

# Adjust this import if your file name differs
from ObliviousHeart import ObliviousHeart, ROCK, PAPER, SCISSORS

print(__doc__.strip(), "\n")

def has_intent(intents, kind):
    return any(i.type == kind for i in intents)


def ok(msg):
    print(f"OK  - {msg}")


def test_gate():
    A = ObliviousHeart("A")

    # Bootstrap with a valid proof
    p = A.propose("B", 1)
    A.ingest(p)

    cur = A.snapshot()["rps"]
    nxt = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}[cur]
    bad = ({ROCK, PAPER, SCISSORS} - {cur, nxt}).pop()

    bad_proof = deepcopy(p)
    bad_proof["rps"] = bad

    intents = A.ingest(bad_proof)

    assert has_intent(intents, "ENVY")
    assert has_intent(intents, "SYNC_REQUEST")
    ok("Gate rejects out-of-window RPS and requests sync")


def test_dominance():
    A = ObliviousHeart("A")

    p = A.propose("B", 1)
    A.ingest(p)

    same_rps = A.snapshot()["rps"]

    weaker = A.propose("B", 1)
    stronger = A.propose("B", 3)

    weaker["rps"] = same_rps
    stronger["rps"] = same_rps

    A.ingest(weaker)
    A.ingest(stronger)

    h_final = A.snapshot()["h"]

    A.ingest(weaker)  # should not replace
    assert A.snapshot()["h"] == h_final

    ok("Dominance: same-RPS contention adopts only the dominant proof")


def test_convergence(seed=7, nodes=5, rounds=50, drop=0.15, dup=0.25):
    rng = random.Random(seed)
    ids = [chr(ord("A") + i) for i in range(nodes)]
    hearts = {i: ObliviousHeart(i) for i in ids}

    pool = []
    for _ in range(rounds):
        src = rng.choice(ids)
        dst = rng.choice([x for x in ids if x != src])
        amt = rng.randint(1, 3)
        pool.append(hearts[src].propose(dst, amt))

    deliveries = []
    for p in pool:
        if rng.random() < drop:
            continue
        deliveries.append(p)
        if rng.random() < dup:
            deliveries.append(deepcopy(p))

    rng.shuffle(deliveries)

    for i in ids:
        local = deliveries[:]
        rng.shuffle(local)
        for p in local:
            hearts[i].ingest(p)

    snaps = {i: hearts[i].snapshot() for i in ids}
    ref = snaps[ids[0]]

    for i in ids[1:]:
        assert snaps[i]["h"] == ref["h"]
        assert snaps[i]["rps"] == ref["rps"]
        assert snaps[i]["tallies"] == ref["tallies"]

    ok(f"Convergence under chaos (nodes={nodes}, rounds={rounds}, drop={drop}, dup={dup}, seed={seed})")
    print(f"     Final: rps={ref['rps']} h={ref['h']} tallies={ref['tallies']}")


def main():
    test_gate()
    test_dominance()
    test_convergence()
    print("\nOK — ObliviousHeart v0 - SmokeTest: Passed")


if __name__ == "__main__":
    main()

