# ObliviousHeart v0.1  
**Truth Through Erasure**  
_No time. No replay. No logs._

---

## What This Is

This folder contains the canonical heart of the **Oblivious Compute** primitive.

The Oblivious Heart defines a minimal, deterministic law for how state survives in a
distributed system **without relying on time, history, logs, or replay protection**.
There are no clocks, no sequence numbers, and no stored past—only a rule for what
survives **now**.

If you are looking for the law itself, you are in the right place.

---

## Files

### ObliviousHeart.py — *The Law*

This is the full, readable, auditable implementation of the Oblivious Heart.

It implements:

- A Rock / Paper / Scissors partition (three-phase admissibility window)
- A strict admissibility gate (current or next phase only)
- A single overwrite rule for truth selection

This file is intended to be:

- Small enough to audit  
- Clear enough to reason about  
- Stable enough to freeze  

---

### ObliviousSkeleton.py — *The Essence*

This file is a non-executable skeleton of the law.

It compresses the entire system down to its irreducible form:

- The gate (admissibility window)
- The overwrite rule
- The propagation trigger

If you want to understand the system in under a minute, read this file first.

---

### ObliviousSmokeTest.py — *The Witness*

This is a runnable smoke test that demonstrates the law in action.

Running it shows, in one execution:

- Proofs outside the admissible window are rejected and trigger sync
- Duplicate and replayed proofs are harmless (idempotent)
- Envy is raised exactly once per violation and clears correctly
- Multiple nodes converge to a single final state **when processing the same
  delivery order**, even with dropped and duplicated messages

The smoke test intentionally does **not** assume convergence under arbitrary,
independently shuffled delivery orders. That property requires a dominance or
tie-breaking rule, which the Oblivious Heart deliberately does not include.

---

## How to Run

From the folder containing the heart files, run the smoke test directly with Python.

The program will print a short declaration of what is being demonstrated, followed
by a summary of the invariants that were proven.

If the program exits cleanly and reports success, the ObliviousHeart invariants hold.

---

## The Law, Informally

Given:

- a current state **S**
- an incoming proof **P**

The next state is determined by:

### Admissibility  
Only proofs at the **current** or **next** partition may compete.

### Overwrite  
Among admissible proofs, the most recently delivered proof overwrites the current
state. No ordering, ranking, or history is consulted.

All non-surviving states are forgotten.

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

1. Start with **ObliviousSkeleton.py**  
2. Identify the linchpin admissibility gate  
3. Read **ObliviousHeart.py**  
4. Run the smoke test  

If you understand the skeleton, the rest will feel obvious.

---

## Status

This is **v0.1**.

Future changes, if any, should be **additive and external** to the heart.

---

## License

See the repository root for licensing and notices.
