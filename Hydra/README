# Hydra

Hydra is a **concrete instantiation** of the **Oblivious Compute (OC)** primitive.

It exists to demonstrate that distributed convergence can be achieved through **selection and erasure**, without consensus protocols, historical logs, or coordinated agreement.

Hydra is not a finished system.  
It is a proof of mechanism.

---

## What Hydra Demonstrates

Hydra illustrates the core properties of Oblivious Compute in a running system:

- Nodes operate independently
- Multiple candidate states may exist transiently
- A deterministic selection rule causes one state to survive
- Discarded states are erased completely
- Convergence occurs without:
  - leader election
  - message ordering
  - consensus rounds
  - history replay

The surviving state is the truth.

---

## Repository Layout

```text
hydra/
├── README.md
├── whitepaper/
│   └── Hydra_Oblivious_Compute_Instantiation.pdf
├── demo/
│   ├── hydra_node.py
│   └── README.md
└── src/
