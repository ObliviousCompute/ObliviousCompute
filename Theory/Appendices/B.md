# Appendix B — Axioms

## Purpose

This appendix presents the current axiomatic formulation of Oblivious Compute. Each axiom is stated in both natural language and mathematical form.

## Resolution

***The axioms define the conditions under which states may participate in the field.*** Their purpose is not to describe every invalid state, but to define admissibility with sufficient **resolution** that inadmissible states cannot progress through the system.

Additional invariants may increase this **resolution** without changing the underlying computational primitive. The problem therefore becomes one of discovering the ***smallest set of independently verifiable invariants*** capable of distinguishing admissible state from inadmissible state.

***An observer does not need to know how a presented state was produced. It only needs to determine whether that state belongs from its present position.***

**Resolution also applies to observation.** An observer may be a transistor, thread, process, virtual machine, computer, cluster, or any other independently resolvable computational boundary. ***The primitive does not prescribe scale; one may simply zoom in or out.***

## Notation

| Symbol | Meaning |
|--------|---------|
| $s$ | State
| 𝓐 | Admissibility function |
| Ω | State space |
| $I$ | Invariant
| Σ | Relational symmetry |
| 𝓕 | Computational field |

---

## Axiom 3 — Invariance

Every admissible transition preserves the system invariants.

### Mathematical Form

$\Large \mathcal{A}(s_i,s_{i+1}) = 1 ;\Longrightarrow; I(s_i) = I(s_{i+1}) $

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

**Continue to [**`Implementations`**](./C.md) in Appendix C...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
