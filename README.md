# Oblivious Compute (OC)

**Oblivious Compute (OC)** is a distributed computation primitive that determines correctness through **admissibility and erasure** rather than agreement and historical coordination.

Instead of preserving logs, ordering messages, or reconstructing the past, **OC** allows multiple candidate states to briefly exist and then admits only one.

The admitted state is truth.  
Everything else is erased.

If you want to wrap your head around the primitive, read the [Admissibility](./Admissibility.md) document.

---

## Byzantium
Is a live, networked terminal game for up to 24 players.

[`Byzantium`](./Byzantium) is the primary expression of Oblivious Compute—a shared projection where multiple participants interact with a single, continuously evolving state.

Connect multiple terminals on the same machine or across a network and you’re immediately inside the same system.

No servers.  
No history.  
No replay.

Just a live board, shaped in real time by the people inside it.

> *(the 10-minute flight)*

---

## Hydra
The minimal distributed demonstration of the same primitive.

[`Hydra`](./Hydra) shows how state moves and collapses across a small network without logs or traditional coordination.

> *(the 1-minute flight)*

---

## Skeleton
Pure, hyper-legible expression of the primitive.

[`Skeleton`](./Skeleton) is the structure that makes the system lawful. No abstraction, no narrative.

> *(the lift diagram)*

---

Play Byzantium.  
Run Hydra.  
Study Skeleton.

> OC doesn’t need to be explained—it can be observed.

---

## What Remains

Oblivious Compute does not carry the past forward.

There are no logs to replay.  
No messages to order.  
No history to reconcile.

Multiple states may appear—  
only one persists.

The system does not remember how it arrived.  
It only admits what can exist next.

Everything else falls away.

> What survives is the shape.
