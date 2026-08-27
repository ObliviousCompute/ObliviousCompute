# Appendix B — Axioms

## Purpose

This appendix establishes the fundamental axioms from which Oblivious Compute is derived. Each axiom is presented in both natural language and mathematical form.

## Notation

| Symbol | Meaning |
|--------|---------|
| $s$ | State |
| $c$ | Cell |
| $\Omega$ | State space |
| $\mathcal{C}$ | Set of cells |
| $G$ | Genesis |
| $T$ | Transition |
| $𝓐$ | Admissibility |
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

$\Large s_0 = G \in \Omega$

---

## Axiom 2 — Transition

Progression occurs exclusively through transitions.

### Mathematical Form

$\Large s_i \xrightarrow{T} s_{i+1}$

Every admissible progression is composed of transitions.

---

## Axiom 3 — Invariance

Every admissible transition preserves the system invariants.

### Mathematical Form

$\Large s_i \xrightarrow{T} s_{i+1} \Longrightarrow I(s_i)=I(s_{i+1})$

---

## Axiom 4 — Event Horizon

Every cell possesses exactly one current event horizon.

### Mathematical Form

$\Large E(c)=\{x\in\Omega\mid 𝓐(s_c,x)=1\}$

---

## Axiom 5 — Symmetry

The field exists only where independently held states preserve the required relational invariants.

### Mathematical Form

$\Large 𝓕 \equiv \Sigma(s_1,s_2,\ldots,s_n)$

---

## Axiom 6 — Synchronization

A cell re-establishes its present state through synchronization with one or more other cells.

Synchronization occurs between a target cell and a non-empty set of other cells, with each cell occupied by an observer.

### Mathematical Form

$\Large S(c_i,P_i)\qquad \varnothing\neq P_i\subseteq\mathcal{C}\setminus\{c_i\}$

---

## Resolution

***The axioms define the conditions under which states may participate in the field.*** Their purpose is not to describe every invalid state, but to define admissibility with sufficient **resolution** that inadmissible states cannot progress through the system.

Additional invariants may increase this **resolution** without changing the underlying computational primitive. The problem therefore becomes one of discovering the ***smallest set of independently verifiable invariants*** capable of distinguishing admissible state from inadmissible state.

***An observer does not need to know how a presented state was produced. It only needs to determine whether that state belongs from its present position.***

**Resolution also applies to observation.** An observer may be a transistor, thread, process, virtual machine, computer, cluster, or any other independently resolvable computational boundary. ***The primitive does not prescribe scale; one may simply zoom in or out.***

---

**Continue to [**`Implementations`**](./C.md) in Appendix C...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
