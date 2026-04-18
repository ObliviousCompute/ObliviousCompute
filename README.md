# Oblivious Compute (OC)

[`Oblivious Compute`](https://github.com/ObliviousCompute/ObliviousCompute) **(OC)** is an open-source distributed computation primitive that determines correctness through admissibility rather than agreement and historical coordination.  

**State ≡ State**.

Instead of preserving logs, ordering messages, or reconstructing the past, **OC** allows multiple candidate states to briefly exist and then admits only one.

**The admitted state is truth.**  
*Everything else falls into oblivion.*

Start with [`Byzantium`](./Byzantium) if you want to experience shared state

To wrap your head around the primitive, read the [`Admissibility`](./Admissibility.md) document.

If you want to see where this is going, [`Mowsie`](./Mowsie/README.md) is the next implementation.

---

## The Stack

Oblivious Compute is expressed across four layers:

- [`Skeleton`](./Skeleton/README.md) — the primitive. The rule that determines what can exist.
- [`Hydra`](./Hydra) — the demonstration. Multiple nodes applying the rule and converging.
- [`Byzantium`](./Byzantium) — the system. A full environment built around the rule. **(testing)**
- [`Mowsie`](./Mowsie/README.md) — the application. A real-world system built on the primitive. **(building)**

*Each layer is the same idea, expressed at a different scale—from pure rule to real-world application.*

---

## Skeleton

Pure, *hyper-legible* expression of the primitive.

[`Skeleton`](./Skeleton/README.md) is the structure that makes the system lawful.

**The structure is the explanation.**

> *(the lift diagram)*

---

## Hydra

The minimal distributed demonstration of the same primitive.

[`Hydra`](./Hydra) shows how state moves and collapses across a small network without logs or traditional coordination.

A simple admissibility gate, expressed in under a thousand lines of code.

> *(the 1-minute flight)*

---

## Byzantium

A live, networked terminal game for up to 24 players.

[`Byzantium`](./Byzantium) is the primary expression of Oblivious Compute—a **shared projection** where multiple participants interact with a single, continuously evolving state.

Connect multiple terminals on the same machine or across a network and you’re *immediately inside the same system*.

**No servers.**  
**No history.**  
**No replay.**

Just a live board, shaped in real time by the people inside it.

> *(the 10-minute flight)*

---

## Mowsie

The first real-world application of Oblivious Compute.

[`Mowsie`](./Mowsie/README.md) is a shared-state system for value—designed to replace punch cards, gift cards, and local loyalty systems with a **cache of distributed truth**.

Users don’t create accounts.  
*They receive value.*

Vendors don’t manage infrastructure.  
*They define an invariant.*

It is the simplest expression of the primitive in the real world.

> *(the 1-hour flight)*

---

## What Survives Is the Shape

Oblivious Compute does not carry the past forward.

There are **no logs to replay.**  
**No messages to order.**  
**No history to reconcile.**

Multiple states may appear—  
*only one persists.*

The system does not remember how it arrived.  
It only admits what can exist next.

**Everything else falls away.**

---

## 📜 License

This project is released under the terms of the [`LICENSE`](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
