# Oblivious Compute (OC)

**Oblivious Compute (OC)** is a distributed computation primitive that determines correctness through **admissibility and erasure** rather than agreement and historical coordination.

Instead of preserving logs, ordering messages, or reconstructing the past, **OC** allows multiple candidate states to briefly exist and then admits only one.

The admitted state is truth.  
Everything else is erased.

**OC is a primitive, not a product.**

---

## The Code

Oblivious Compute is best understood by interacting with it.

Start with the surface, then move inward:

### Byzantium
A live, networked terminal system built on OC.

This is the primitive made visible—a shared projection where interaction directly reshapes state.

→ [`Byzantium/`](./Byzantium)

*(the 10-minute flight)*

---

### Hydra
A minimal distributed demonstration of the same primitive.

Hydra shows how state moves and collapses across a small network without logs or coordination.

→ [`Hydra/`](./Hydra)

*(the 1-minute flight)*

---

### Skeleton
A stripped-down, hyper-legible expression of the invariant.

No abstraction, no narrative—just the structure that makes the system lawful.

→ [`Skeleton/`](./Skeleton)

*(the lift diagram)*

---

Run Byzantium.  
Watch Hydra.  
Read Skeleton.

> OC doesn’t need to be explained—it can be observed.

---

## Fragments

This repository includes a set of short write-ups exploring the ideas behind OC.

[Oblivious-Compute.pdf](./Fragments/Oblivious-Compute.pdf) describes the core compute primitive itself.

[Forward-Compute.pdf](./Fragments/Forward-Compute.pdf) explains the forward-compute model used by Hydra and the Oblivious Heart.

[Ambient-Compute.pdf](./Fragments/Ambient-Compute.pdf) questions whether consensus and history are foundational at all.

---

## What Problem Does OC Address?

Most distributed systems assume correctness requires memory:

- message ordering  
- logs and replay  
- consensus and reconciliation  
- long-lived historical state  

These assumptions introduce complexity, latency, and coordination overhead.

Oblivious Compute removes the requirement to remember the past.

Correctness is defined operationally as:

> what survives

Not how it was reached.
