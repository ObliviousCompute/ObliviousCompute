# Appendix D — Correspondence

## Purpose

This appendix places Oblivious Compute beside established computational models using a common symbolic vocabulary.

***These are correspondences, not equivalences.*** Each section identifies a shared structure and the point at which the models diverge.

## Notation

| Symbol | Meaning |
|--------|---------|
| $o$ | Observer |
| $s$ | State |
| $c$ | Cell |
| $x$ | Proposed input or continuation |
| $h$ | History |
| $T$ | Transition |
| $I$ | Invariant |
| $E$ | Event Horizon |
| $A$ | Admissibility |
| $\Sigma$ | Symmetry |
| $\mathcal{F}$ | Field / computational object |
| $Q$ | Agreement / commit |
| $\sqcup$ | Join / merge |
| $\sqsubseteq$ | Information order |

---

## Oblivious Compute

***I’m here. Something appears. Does it belong?***

An observer occupies a present position and a projected state appears within the same state space. The observer performs a binary positional determination:

$\Large s\in\Omega \qquad x\in\Omega \qquad A(s,x)\in\lbrace 0,1\rbrace$

The result is simple. The relationship is not. The same projected state may have different computational significance at different observer positions:

$\Large s_i\neq s_j \qquad A(s_i,x)=1 \qquad A(s_j,x)=0$

Distributed computation begins when the same primitive is instantiated across independently maintained observer positions:

$\Large s_1,s_2,\ldots,s_n\in\Omega \qquad A(s_i,x)\in\lbrace 0,1\rbrace$

An admitted state change may establish a new local position, while independently maintained observers need not occupy equivalent positions:

$\Large s_i\rightarrow s_i' \qquad s_i'\neq s_j'$

Different executions may therefore traverse different distributed configurations:

$\Large (s_1',s_2',\ldots,s_n')\neq(s_1'',s_2'',\ldots,s_n'')$

The intermediate configurations need not themselves constitute the distributed computational object.


Let $\Sigma$ denote the relational symmetry among independently maintained observer positions. The resulting field is:

$\Large 𝓕\equiv\Sigma(s_1,s_2,\ldots,s_n)$

The states remain local to their observers.


***𝓕 contains no state of its own. The field is the computational locus.***

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
