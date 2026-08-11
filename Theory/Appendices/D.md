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

## Oblivious Compute

Oblivious Compute does not derive the authority of the present from an agreed, merged, or reconstructed history.

For correspondence, let 𝐴 represent the local admissibility test performed at the current event horizon.

### Mathematical Form

$A_i(x)\iff [x\in E(c_i)]\land[I(s_i)=I(x)]$

$A_i(x)\Rightarrow T:s_i\to x$

History does not appear as an input to admissibility. Synchronization re-establishes present state between observers without requiring historical reconstruction.

***The present does not summarize an authoritative past. The present is evaluated directly for admissible progression.***

---

## Consensus

Oblivious Compute does not require observers to participate in determining a common progression. **Each observer already anticipates the admissible continuations of its own state.**

When a transition is presented, no agreement must first be reached. The observer resolves only whether that transition was already admissible.

***Consensus is participatory. Oblivious Compute is anticipatory.***

---

## Markovian

A Markovian model makes the present state sufficient for determining subsequent progression.

### Mathematical Form

$\Pr(s_{i+1}\mid s_i,h_i)=\Pr(s_{i+1}\mid s_i)$

The history need not remain visible once the information relevant to future progression is represented by the present state.

***The present summarizes what the past established.***

---

## Replicated Log

In replicated state-machine systems such as Multi-Paxos or Raft, agreement establishes an authoritative sequence of inputs.

### Abstract Form

$Q(x_i)\rightarrow h_{i+1}=h_i \mathbin{\|} x_i$

$𝑠_{i+1} = \mathrm{Apply}(ℎ_{i+1})$

The present state is obtained from an agreed progression.

***Agreement establishes the history from which the present is derived.***

---

## Partially Ordered Log

A partially ordered log relaxes the requirement that every event occupy one total sequence.

### Abstract Form

$h=(X,\prec)$

$𝑠 = \mathrm{Apply}(ℎ)$

where 𝑋 is the set of recorded events and ≺ preserves only the ordering relationships required between them.

Independent events may therefore progress concurrently while dependent events remain ordered.

***Sequence is relaxed, but history remains the shared substrate.***

---

## State-Based CRDT

A state-based CRDT permits replicas to progress independently and later converge through a join operation.

### Mathematical Form

$s_i'=T(s_i,x_i)$

$s_i\sqsubseteq s_i'$

The join is associative, commutative, and idempotent:

$s_i\leftarrow s_i\sqcup s_j$

$a\sqcup b=b\sqcup a$

$(a\sqcup b)\sqcup c=a\sqcup(b\sqcup c)$

$a\sqcup a=a$

***Divergent replicas are structured so their states can be reconciled.***

---

## Invariant Confluence

Invariant confluence asks whether independently valid executions can be merged while preserving an application invariant.

### Abstract Form

$s_0\to^{*}s_1$

$s_0\to^{*}s_2$

$I(s_1)\land I(s_2)\Rightarrow I(s_1\sqcup s_2)$

for states independently reachable from a common ancestor.

***Coordination can be avoided when independently valid executions remain valid after reconciliation.***

---

## CALM

CALM relates coordination-free computation to monotonicity.

### Abstract Form

$s_1\sqsubseteq s_2\Rightarrow F(s_1)\sqsubseteq F(s_2)$

Additional information does not invalidate conclusions already reached.

***Coordination freedom emerges because information grows monotonically.***

---

***Consensus is participatory. Oblivious Compute is anticipatory.***

---

**Go back to [**`Theory`**](../README.md)...**

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
