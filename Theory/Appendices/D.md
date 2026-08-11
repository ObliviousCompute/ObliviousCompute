# Appendix D — Correspondence

## Purpose

This appendix places Oblivious Compute beside established computational models using a common symbolic vocabulary.

***These are correspondences, not equivalences.*** Each section identifies a shared structure and the point at which the models diverge.

## Notation

| Symbol | Meaning |
|---------|---------|
| 𝑠 | State |
| 𝑐 | Cell |
| 𝑥 | Proposed input or continuation |
| ℎ | History |
| 𝑇 | Transition |
| 𝐼 | Invariant |
| 𝐸 | Event Horizon |
| 𝐴 | Admissibility |
| 𝑄 | Agreement / commit |
| ⊔ | Join / merge |
| ⊑ | Information order |
| 𝐹 | Computation |

---

## Markovian

A Markovian model makes the present state sufficient for determining subsequent progression.

### Mathematical Form

Pr(𝑠ᵢ₊₁ | 𝑠ᵢ, ℎᵢ) = Pr(𝑠ᵢ₊₁ | 𝑠ᵢ)

The history need not remain visible once the information relevant to future progression is represented by the present state.

***The present summarizes what the past established.***

---

## Replicated Log

In replicated state-machine systems such as Multi-Paxos or Raft, agreement establishes an authoritative sequence of inputs.

### Abstract Form

𝑄(𝑥ᵢ) → ℎᵢ₊₁ = ℎᵢ ⧺ 𝑥ᵢ

𝑠ᵢ₊₁ = Apply(ℎᵢ₊₁)

The present state is obtained from an agreed progression.

***Agreement establishes the history from which the present is derived.***

---

## Partially Ordered Log

A partially ordered log relaxes the requirement that every event occupy one total sequence.

### Abstract Form

ℎ = (𝑋, ≺)

𝑠 = Apply(ℎ)

where 𝑋 is the set of recorded events and ≺ preserves only the ordering relationships required between them.

Independent events may therefore progress concurrently while dependent events remain ordered.

***Sequence is relaxed, but history remains the shared substrate.***

---

## State-Based CRDT

A state-based CRDT permits replicas to progress independently and later converge through a join operation.

### Mathematical Form

𝑠ᵢ′ = 𝑇(𝑠ᵢ, 𝑥ᵢ)

𝑠ᵢ ← 𝑠ᵢ ⊔ 𝑠ⱼ

The join is associative, commutative, and idempotent:

a ⊔ b = b ⊔ a

(a ⊔ b) ⊔ c = a ⊔ (b ⊔ c)

a ⊔ a = a

***Divergent replicas are structured so their states can be reconciled.***

---

## Invariant Confluence

Invariant confluence asks whether independently valid executions can be merged while preserving an application invariant.

### Abstract Form

𝑠₀ →* 𝑠₁

𝑠₀ →* 𝑠₂

𝐼(𝑠₁) ∧ 𝐼(𝑠₂) ⇒ 𝐼(𝑠₁ ⊔ 𝑠₂)

for states independently reachable from a common ancestor.

***Coordination can be avoided when independently valid executions remain valid after reconciliation.***

---

## CALM

CALM relates coordination-free computation to monotonicity.

### Abstract Form

𝑠₁ ⊑ 𝑠₂ ⇒ 𝐹(𝑠₁) ⊑ 𝐹(𝑠₂)

Additional information does not invalidate conclusions already reached.

***Coordination freedom emerges because information grows monotonically.***

---

## Oblivious Compute

Oblivious Compute does not derive the authority of the present from an agreed, merged, or reconstructed history.

For correspondence, let 𝐴 represent the local admissibility test performed at the current event horizon.

### Mathematical Form

𝐴ᵢ(𝑥) = [𝑥 ∈ 𝐸(𝑐ᵢ)] ∧ [𝐼(𝑠ᵢ) = 𝐼(𝑥)]

𝐴ᵢ(𝑥) ⇒ 𝑇 : 𝑠ᵢ → 𝑥

History does not appear as an input to admissibility. Synchronization re-establishes present state between observers without requiring historical reconstruction.

***The present does not summarize an authoritative past. The present is evaluated directly for admissible progression.***

---

## Consensus

Oblivious Compute does not require observers to participate in determining a common progression. **Each observer already anticipates the admissible continuations of its own state.**

When a transition is presented, no agreement must first be reached. The observer resolves only whether that transition was already admissible.

***Consensus is participatory. Oblivious Compute is anticipatory.***

---

**Go back to [**`Theory`**](../README.md)...**

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
