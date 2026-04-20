# Oblivious Compute (OC)

**No History. No Logs. No Problems.**

---

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute) **(OC)** is an open-source distributed computation primitive that determines correctness through admissibility rather than agreement and historical coordination.  

**State ≡ State**.

Instead of preserving logs, ordering messages, or reconstructing the past, **OC** allows multiple candidate states to briefly exist and then admits only one.

**The admitted state is truth.**  
*Everything else falls into oblivion.*

## The Path

Start with [**`Byzantium`**](./Byzantium/README.md) to see this in motion — a live system with no backend.

Then check out [**`Mowsie`**](./Mowsie/README.md) to understand where it’s going.

If it clicks, move to [**`Hydra`**](./Hydra/README.md) to see how shared state behaves at a smaller scale.

From there, [**`Skeleton`**](./Skeleton/README.md) shows the core mechanism stripped down.

If you want to go deeper, [**`Admissibility`**](./Admissibility/README.md) explains the idea underneath it all.

---

## Mowsie

The first real-world application of Oblivious Compute.

[**`Mowsie`**](./Mowsie/README.md) is a shared-state system for value—designed to replace punch cards, gift cards, and local loyalty systems with a **cache of distributed truth**.

Users don’t create accounts.  
*They receive value.*

Vendors don’t manage infrastructure.  
*They define an invariant.*

It is the simplest expression of the primitive in the real world.

> *(the 1-hour flight)*

---

## Byzantium

A live, networked terminal game for up to 24 players.

[**`Byzantium`**](./Byzantium) is the primary expression of Oblivious Compute—a **shared projection** where multiple participants interact with a single, continuously evolving state.

Connect multiple terminals on the same machine or across a network and you’re *immediately inside the same system*.

Just a live board, shaped in real time by the people inside it.

> *(the 10-minute flight)*

---

## Hydra

The minimal distributed demonstration of the same primitive.

[**`Hydra`**](./Hydra) shows how state moves and collapses across a small network without logs or traditional coordination.

A simple admissibility gate, expressed in under a thousand lines of code.

> *(the 1-minute flight)*

---

## Skeleton

Pure, *hyper-legible* expression of the primitive.

[**`Skeleton`**](./Skeleton/README.md) is the structure that makes the system lawful.

**The structure is the explanation.**

> *(the lift diagram)*

---

## What Survives Is the Shape

Oblivious Compute does not carry the past forward.

The system does not remember how it arrived.  

It only admits what can exist next.

**Everything else falls away.**

---

## 📜 License

This project is released under the terms of the [**`LICENSE`**](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
