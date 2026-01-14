"""
# ==============================================
# ObliviousSmokeTest v0.1 — Truth Through Erasure
# No time. No replay. No logs.
# ==============================================

This smoke test validates what the Heart *actually* guarantees.

It proves:

1) Gate + Envy: proofs outside the admissible membrane window are rejected,
   ENVY is raised (once), and REQUEST_SYNC is emitted.

2) Envy clears: envy resolves when valid in-window reality arrives, or via
   dream/seed hydration.

3) No-op rule: if tallies are unchanged, ingest is a no-op even if rps differs.
   (This matches Plexus mechanics.)

4) Idempotency: duplicate delivery of the same proof does not change state.

5) Convergence (ordered): if all nodes process the same delivery order (even
   with drops/dups), they converge to the same final state.

Note: This Heart intentionally does NOT provide shuffled-order convergence.
That property requires a deterministic dominance rule, which was removed by
design.
"""

import random
from copy import deepcopy

from ObliviousHeart import (
    ObliviousHeart,
    NextRPS,
    ROCK,
    PAPER,
    SCISSORS,
)

print(__doc__.strip(), "\n")


def has_intent(intents, kind):
    return any(i.type == kind for i in intents)


def ok(msg):
    print(f"OK  - {msg}")


def rps_next_of(rps):
    return {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}[rps]


def rps_bad_for(cur):
    nxt = rps_next_of(cur)
    return ({ROCK, PAPER, SCISSORS} - {cur, nxt}).pop()


def test_gate_and_sync():
    A = ObliviousHeart("A")
    p = A.propose("B", 1)
    A.ingest(p)

    cur = A.snapshot()["rps"]
    bad = rps_bad_for(cur)
    bad_proof = deepcopy(p)
    bad_proof["rps"] = bad

    intents = A.ingest(bad_proof)
    assert has_intent(intents, "ENVY")
    assert has_intent(intents, "REQUEST_SYNC")
    assert A.emotions()["envy"] is True
    ok("Gate: out-of-window rps => ENVY + REQUEST_SYNC, envy latched")


def test_envy_emits_once():
    A = ObliviousHeart("A")
    p = A.propose("B", 1)
    A.ingest(p)

    cur = A.snapshot()["rps"]
    bad = rps_bad_for(cur)
    bad_proof = deepcopy(p)
    bad_proof["rps"] = bad

    first = A.ingest(bad_proof)
    assert has_intent(first, "ENVY")
    assert has_intent(first, "REQUEST_SYNC")

    for _ in range(6):
        nxt_intents = A.ingest(deepcopy(bad_proof))
        assert not has_intent(nxt_intents, "ENVY")
        assert has_intent(nxt_intents, "REQUEST_SYNC")

    ok("Envy: ENVY emits once; repeated invalid proofs don't spam ENVY")


def test_envy_clears_on_valid_or_dream():
    A = ObliviousHeart("A")
    p = A.propose("B", 1)
    A.ingest(p)

    cur = A.snapshot()["rps"]
    bad = rps_bad_for(cur)
    bad_proof = deepcopy(p)
    bad_proof["rps"] = bad
    A.ingest(bad_proof)
    assert A.emotions()["envy"] is True

    # Clear envy via dream hydration
    dream = A.seed_proof()
    dream["is_dream"] = True
    A.ingest(dream)
    assert A.emotions()["envy"] is False

    # Re-enter envy, then clear via valid in-window reality
    A.ingest(bad_proof)
    assert A.emotions()["envy"] is True
    good = deepcopy(p)
    good["rps"] = cur
    A.ingest(good)
    assert A.emotions()["envy"] is False

    ok("Envy clears: via dream/seed hydration or valid in-window reality")


def test_noop_ignores_rps_when_tallies_same():
    A = ObliviousHeart("A")
    p = A.propose("B", 2)
    A.ingest(p)
    snap = A.snapshot()

    q = {
        "id": "X",
        "tallies": dict(snap["tallies"]),
        "rps": NextRPS(snap["rps"]),
    }
    A.ingest(q)
    snap2 = A.snapshot()

    assert snap2["tallies"] == snap["tallies"]
    assert snap2["rps"] == snap["rps"]
    ok("No-op: same tallies => ignore rps-only changes")


def test_idempotent_duplicates():
    A = ObliviousHeart("A")
    p = A.propose("B", 3)

    A.ingest(p)
    snap1 = A.snapshot()

    A.ingest(deepcopy(p))
    snap2 = A.snapshot()

    assert snap2["tallies"] == snap1["tallies"]
    assert snap2["rps"] == snap1["rps"]
    ok("Idempotent: duplicate proof delivery is a no-op")


def test_convergence_ordered(seed=7, nodes=5, rounds=60, drop=0.15, dup=0.25):
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

    # Everyone processes the same delivery order => determinism + convergence
    for i in ids:
        for p in deliveries:
            hearts[i].ingest(p)

    snaps = {i: hearts[i].snapshot() for i in ids}
    ref = snaps[ids[0]]

    for i in ids[1:]:
        assert snaps[i]["rps"] == ref["rps"]
        assert snaps[i]["tallies"] == ref["tallies"]

    ok(f"Convergence (ordered) under drops/dups "
       f"(nodes={nodes}, rounds={rounds}, drop={drop}, dup={dup}, seed={seed})")
    print(f"     Final: rps={ref['rps']} tallies={ref['tallies']}")


def main():
    tests = [
        ("gate_and_sync", test_gate_and_sync,
         "Reject invalid phase; raise envy; request sync"),
        ("envy_emits_once", test_envy_emits_once,
         "ENVY is transition-only; repeated invalid inputs don't spam"),
        ("envy_clears", test_envy_clears_on_valid_or_dream,
         "Envy resolves via dream/seed hydration or valid in-window proof"),
        ("noop_rps", test_noop_ignores_rps_when_tallies_same,
         "Tallies-only no-op: rps-only changes are ignored"),
        ("idempotent_dups", test_idempotent_duplicates,
         "Duplicate delivery is harmless"),
        ("convergence_ordered", test_convergence_ordered,
         "Same delivery order => all nodes converge (even with drops/dups)"),
    ]

    for _, fn, _ in tests:
        fn()

    print("\nSummary:")
    for name, _, desc in tests:
        print(f" - {name}: {desc}")

    print("\nOK — ObliviousHeart SmokeTest: Passed")


if __name__ == "__main__":
    main()
