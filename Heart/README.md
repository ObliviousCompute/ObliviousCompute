# ObliviousHeart v0.1

Truth Through Erasure  
No time. No replay. No logs.

---

## What This Is

This folder contains the canonical heart of the Oblivious Compute primitive.

The Oblivious Heart defines a minimal, deterministic law for how state survives in a distributed system without relying on time, history, logs, or replay protection. There are no clocks, no sequence numbers, and no stored past—only a rule for what survives now.

If you are looking for the law itself, you are in the right place.

---

## Files

### ObliviousHeart.py — The Law

This is the full, readable, auditable implementation of the Oblivious Heart.

It implements:
- A Rock / Paper / Scissors partition (three-phase admissibility window)
- A deterministic dominance ordering within a partition
- A single overwrite rule for truth selection

This file is intended to be:
- Small enough to audit
- Clear enough to reason about
- Stable enough to freeze

---

### ObliviousHeartSkeleton.py — The Essence

This file is a non-executable skeleton of the law.

It compresses the entire system down to its irreducible form:
- The gate
- The ordering
- The overwrite

If you want to understand the system in under a minute, read this file first.

---

### ObliviousSmokeTest.py — The Witness

This is a runnable smoke test that demonstrates the law under disorder.

Running it shows, in one execution:

1) Proofs outside the admissible window are rejected and trigger sync  
2) Same-partition contention resolves deterministically by dominance  
3) Multiple nodes converge to a single final state under shuffled, duplicated, and dropped message delivery  

## How to Run

From the folder containing the heart files, run the smoke test directly with Python.
The program will print a short declaration of what is being demonstrated.

If the program exits cleanly and reports success, the ObliviousHeart invariants hold.

---

## The Law, Informally

Given:
- a current state S
- an incoming proof P

The next state is determined by:

1) Admissibility  
Only proofs at the current or next partition may compete.

2) Ordering  
Among admissible proofs, the greater under deterministic ordering wins.

3) Overwrite  
The surviving state replaces the current state. All others are forgotten.

There is no appeal to history.

---

## What This Is Not

- Not a blockchain
- Not a consensus log
- Not a replay-based protocol
- Not time-ordered
- Not dependent on transport, networking, or threads

Those concerns live outside the heart.

---

## How to Read This Code

1) Start with ObliviousHeartSkeleton.py  
2) Read the linchpin overwrite line  
3) Then read ObliviousHeart.py  
4) Finally, run the smoke test  

If you understand the skeleton, the rest will feel obvious.

---

## Status

This is v0.1

Future changes, if any, should be additive and external.

---

## License

See the repository root for licensing and notices.
