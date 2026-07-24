## Definitions

**State**
: The complete present representation of a system.

**Admissibility**
: The property determining whether a candidate state may be accepted.

**Invariant**
: A constraint that must remain true for a state transition to be admissible.

**Transition**
: A proposed movement from one admissible state to another.

**Geometry**
: The constrained structure induced by the system's invariants.

---

## Formal Correspondence

**Admissibility**

s′ ∈ A(s)

Valid transitions belong to the admissible set.

---

**Window Constraint**

A(s) = {s, next(s)}

Only the current state and its immediate successor are admissible.

---

**Rejection**

s′ ∉ A(s)

The transition produces no effect.

---

**Cycle Closure**

nextᵏ(s) = s

Finite admissibility spaces may form closed cycles.

---

**Verification**

Verify(s, s′) = true ⇔ s′ ∈ A(s)

Validity is determined entirely by the current state and its admissible structure.

---

## Axioms

### 1. State Primacy

A system is defined entirely by its current state.

---

### 2. Admissibility Equivalence

A state is valid if and only if it satisfies the system's invariants.

---

### 3. History Independence

The validity of a state is independent of the sequence of transitions that produced it.

---

### 4. Admissible Progression

Forward progression occurs only through admissible states.

States that are not admissible produce no effect.

---

### 5. Erasure Semantics

Inadmissible states are not stored, processed, or reconciled.

They are treated as non-existent.

---

### 6. Geometric Realization

The set of admissible states forms a constrained geometry.

All valid computation occurs within that geometry.

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
