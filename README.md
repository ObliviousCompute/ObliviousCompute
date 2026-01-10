# Oblivious Compute (OC)

Oblivious Compute (OC) is a distributed computation primitive that determines correctness through **selection and erasure** rather than agreement and historical coordination.

Instead of passing messages in chains or preserving logs, OC allows multiple candidate states to exist briefly and then **deterministically collapses them to a single surviving state**. The surviving state is the truth; all others are erased and leave no trace.

OC is a primitive, not a product.

---
## Start here

This repository includes a short write-up, [Forward-Compute.pdf](Forward-Compute.pdf), explaining the forward-compute model used by Hydra and the Oblivious Heart.

If you want the law without ceremony, start here → the [`heart/`](./heart/) directory contains **The Oblivious Heart**—the Hydraplexus stripped to its bare, testable form, along with its primary smoke test and skeleton.

If you just want to see it move, go here → the [`Hydra/`](./Hydra/) demo.

Both paths are concrete instantiations of the same oblivious compute primitive.


---
## What Problem Does OC Address?

Most distributed systems assume correctness requires memory:
- message ordering
- logs and replay
- consensus and reconciliation
- long-lived historical state

These assumptions make systems heavy and complex.

Oblivious Compute removes the requirement to remember the past.  
Correctness is defined operationally as **what survives**, not how it was reached.

---

## Repository Structure

This repository separates **theory** from **instantiation**.

```text
ObliviousCompute
├── Heart
|   ├── ObliviousHeart.py
|   ├── ObliviousSkeleton.py
|   ├── ObliviousSmokeTest.py
|   └── README.md
├── Hydra/
|   ├── Video/
|   │   └── hydra-demo.mp4
|   ├── src/
|   |   ├── Fleash_and_Bone.py
|   |   ├── Hydra.py
|   |   ├── HydraPlexus.py
|   |   └── Lore.pdf
|   └── README.md      
├── Whitepaper/
|   └── Oblivious-Compute-Primitive.pdf
├── Forward-Compute.pdf
├── LICENSE
├── NOTICE
└── README.md

