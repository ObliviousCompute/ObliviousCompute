# Oblivious Compute (OC)

**Oblivious Compute (OC)** is a distributed computation primitive that determines correctness through **admissibility and erasure** rather than agreement and historical coordination.

Instead of preserving logs, ordering messages, or reconstructing the past, **OC** allows multiple candidate states to briefly exist and then admits only one.

The admitted state is truth.  
Everything else is erased.

If you want to wrap your head around the primitive, read the [Admissibility](./Admissibility.md) document.

---

## The Code

Oblivious Compute is best understood by interacting with it.

Start with the surface, then move inward:

### Byzantium
A live, networked terminal system built on OC.

[`Byzantium`](./Byzantium) is the primitive made visible—a shared projection where interaction directly reshapes state.

*(the 10-minute flight)*

---

### Hydra
A minimal distributed demonstration of the same primitive.

[`Hydra`](./Hydra) shows how state moves and collapses across a small network without logs or traditional coordination.

*(the 1-minute flight)*

---

### Skeleton
A stripped-down, hyper-legible expression of the invariant.

[`Skeleton`](./Skeleton) is structure that makes the system lawful. No abstraction, no narrative.

*(the lift diagram)*

---

Run Byzantium.  
Watch Hydra.  
Read Skeleton.

> OC doesn’t need to be explained—it can be observed.

---

## What OC Removes

Most distributed systems assume correctness requires memory:

- message ordering  
- logs and replay  
- consensus and reconciliation  
- long-lived historical state  

These assumptions introduce complexity, latency, and coordination overhead.

Oblivious Compute removes the requirement to remember the past.

Correctness is defined operationally as:

> What survives  
>Not how it was reached
