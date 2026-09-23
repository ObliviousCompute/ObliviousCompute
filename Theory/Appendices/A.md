# Appendix A — Axioms

## Purpose

This appendix presents the smallest axiomatic formulation of Oblivious Compute. The primitive requires independently maintained state, positional admissibility, and a relational computational field.

Additional machine-specific invariants may refine admissibility without changing the underlying primitive.

## Notation

| Symbol | Meaning |
|--------|---------|
| $s$ | Present observer state |
| $x$ | Projected state |
| 𝓐 | Admissibility function |
| Ω | State space |
| Σ | Relational symmetry |
| 𝓕 | Computational field |
| Δ | Diagonal |

---

## Axiom 1 — Position

Every observer independently maintains a present state in Ω. That present position is the point from which any encountered state is evaluated.

### Mathematical Form

$\Large s_i\in\Omega \qquad (s_1,s_2,\ldots,s_n)\in\Omega^n$

---

## Axiom 2 — Admissibility

A projected state carries no authority merely because it was projected. Any observer that encounters x evaluates it from its own present position using the same binary admissibility function 𝓐.

### Mathematical Form

$\Large 𝓐:\Omega\times\Omega\rightarrow\lbrace 0,1\rbrace \qquad 𝓐(s_i,x)\in\lbrace 0,1\rbrace$

The same projected state may therefore belong from one observer position and not from another.

$\Large s_i\neq s_j \qquad 𝓐(s_i,x)=1 \qquad 𝓐(s_j,x)=0$

---

## Axiom 3 — Relation

Across independently maintained observer states, relational symmetry Σ constitutes the computational field 𝓕.

### Mathematical Form

$\Large 𝓕\equiv\Sigma(s_1,s_2,\ldots,s_n)$

***No observer contains the field. The field contains no state of its own. It exists only through the relation among independently maintained states.***

---

## Perfect Symmetry

When independently maintained observer states **collide on the diagonal**, their positions coincide in perfect relational symmetry.

### Mathematical Form

$\Large s_1=s_2=\cdots=s_n=s \qquad (s_1,s_2,\ldots,s_n)\in\Delta_n(\Omega)$

At perfect symmetry, the independently maintained coordinates occupy one relational position.

$\Large \Delta_n(\Omega)\cong\Omega$

***Perfect symmetry is one condition of the field. The field exists both on and off the diagonal.***

---

🧭 **Continue to [**`Definitions`**](./B.md)...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
