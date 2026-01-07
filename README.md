# Oblivious Compute (OC)

Oblivious Compute (OC) is a distributed computation primitive that determines correctness through **selection and erasure** rather than agreement and historical coordination.

Instead of passing messages in chains or preserving logs, OC allows multiple candidate states to exist briefly and then **deterministically collapses them to a single surviving state**. The surviving state is the truth; all others are erased and leave no trace.

OC is a primitive, not a product.

---
## Start here

If you want to see the idea running live, start with the Hydra demo:

→ [`Hydra/`](./Hydra)

It includes:
- a short screen capture of the system running
- instructions to run five nodes locally
- a concrete instantiation of the oblivious compute primitive

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
├── README.md
├── LICENSE
├── NOTICE
├── Whitepaper/
│   └── Oblivious Compute Primitive.pdf
└── Hydra/
    ├── README.md
    ├── Video/
    │   └── hydra-demo.mp4
    ├── Whitepaper/
    │   └── Hydra Proofs.pdf
    └── src/
        ├── README.md
        ├── hydra-demo.py
        └── hydra-skeleton.py

