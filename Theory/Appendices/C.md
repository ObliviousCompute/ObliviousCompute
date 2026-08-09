# Appendix C — Axioms

## Purpose

This appendix establishes the fundamental axioms from which Oblivious Compute is derived. Each axiom is presented in both natural language and mathematical form.

## Notation

| Symbol | Meaning |
|---------|---------|
| 𝑠 | State |
| 𝑐 | Cell |
| 𝑔 | Genesis |
| 𝑇 | Transition |
| 𝐼 | Invariant |
| 𝐸 | Event Horizon |
| 𝑆 | Synchronization |

---

## Axiom 1 — Genesis

Every admissible progression begins from a genesis.

### Mathematical Form

∃! 𝑔

---

## Axiom 2 — Transition

Progression occurs exclusively through transitions.

### Mathematical Form

𝑇 : 𝑠ᵢ → 𝑠ᵢ₊₁

Every admissible progression is composed of transitions.

---

## Axiom 3 — Invariance

Every admissible transition preserves the system invariants.

### Mathematical Form

𝐼(𝑠ᵢ) = 𝐼(𝑠ᵢ₊₁)

---

## Axiom 4 — Event Horizon

Every cell possesses exactly one current event horizon.

### Mathematical Form

∀𝑐, ∃! 𝐸(𝑐)

---

## Axiom 5 — Synchronization

A cell re-establishes its present state through synchronization with another cell.

Synchronization occurs exclusively between two cells, with each cell occupied by an observer.

### Mathematical Form

𝑆(𝑐ᵢ, 𝑐ⱼ)

---

## Resolution

***The axioms define the admissible field.*** Their purpose is not to describe every invalid state, but to define lawful state with sufficient **resolution** that invalid states cannot pass through it.

Additional invariants may increase this **resolution** without changing the underlying computational primitive. The problem therefore becomes one of discovering the ***smallest set of independently verifiable invariants*** capable of distinguishing admissible state from inadmissible state.

**Resolution also applies to observation.** An observer may be a transistor, thread, process, virtual machine, computer, cluster, or any other independently resolvable computational boundary. ***The primitive does not prescribe scale; one may simply zoom in or out.***

***In this sense, the field does not need to know how a state was produced. It only needs to determine whether the state belongs.***

---

**Go back to** [**`Theory`**](../README.md)

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
