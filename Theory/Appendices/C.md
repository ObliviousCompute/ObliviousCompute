# Appendix C — Axioms

## Purpose

This appendix establishes the fundamental axioms from which Oblivious Compute is derived. Each axiom is presented in both natural language and mathematical form.

## Notation

| Symbol | Meaning |
|--------|---------|
| $s$ | State |
| $c$ | Cell |
| $\Omega$ | State space |
| $\mathcal{C}$ | Set of cells |
| $g$ | Genesis |
| $T$ | Transition |
| $A$ | Admissibility |
| $I$ | Invariant |
| $E$ | Event Horizon |
| $\Sigma$ | Symmetry |
| $𝓕$ | Field |
| $S$ | Synchronization |
| $P_i$ | Non-empty set of peer cells |

---

## Axiom 1 — Genesis

Every admissible progression begins from a genesis.

### Mathematical Form

$\exists! g\in\Omega \qquad s_0=g$

---

## Axiom 2 — Transition

Progression occurs exclusively through transitions.

### Mathematical Form

$s_i\xrightarrow{T}s_{i+1}$

Every admissible progression is composed of transitions.

---

## Axiom 3 — Invariance

Every admissible transition preserves the system invariants.

### Mathematical Form

$A(s_i,s_{i+1})=1 \Longrightarrow I(s_i)=I(s_{i+1})$

---

## Axiom 4 — Event Horizon

Every cell possesses exactly one current event horizon.

### Mathematical Form

$\forall c_i\in\mathcal{C},\ \exists! E(c_i)$

---

## Axiom 5 — Symmetry

The field exists only where independently held states preserve the required relational invariants.

### Mathematical Form

$𝓕 \equiv \Sigma(s_1,s_2,\ldots,s_n)$

---

## Axiom 6 — Synchronization

A cell re-establishes its present state through synchronization with one or more other cells.

Synchronization occurs between a target cell and a non-empty set of other cells, with each cell occupied by an observer.

### Mathematical Form

$S(c_i,P_i)\qquad \varnothing\neq P_i\subseteq\mathcal{C}\setminus\{c_i\}$

---

## Resolution

***The axioms define the admissible field.*** Their purpose is not to describe every invalid state, but to define lawful state with sufficient **resolution** that invalid states cannot pass through it.

Additional invariants may increase this **resolution** without changing the underlying computational primitive. The problem therefore becomes one of discovering the ***smallest set of independently verifiable invariants*** capable of distinguishing admissible state from inadmissible state.

***In this sense, the field does not need to know how a state was produced. It only needs to determine whether the state belongs.***

**Resolution also applies to observation.** An observer may be a transistor, thread, process, virtual machine, computer, cluster, or any other independently resolvable computational boundary. ***The primitive does not prescribe scale; one may simply zoom in or out.***

---

**Continue to [**`Appendix D`**](./D.md)...**

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
