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

The same projected state may have different computational significance at different observer positions:

$\Large s_i\neq s_j \qquad A(s_i,x)=1 \qquad A(s_j,x)=0$

Distributed computation begins when the same primitive is instantiated across independently maintained observer positions:

$\Large s_1,s_2,\ldots,s_n\in\Omega \qquad A(s_i,x)\in\lbrace 0,1\rbrace$

An admitted state change may establish a new local position. Different observers and different executions need not traverse equivalent configurations:

$\Large s_i\rightarrow s_i' \qquad s_i'\neq s_j' \qquad (s_1',s_2',\ldots,s_n')\neq(s_1'',s_2'',\ldots,s_n'')$


Let $\Sigma$ denote the relational symmetry among independently maintained observer positions. The resulting field is:

$\Large 𝓕\equiv\Sigma(s_1,s_2,\ldots,s_n)$


***𝓕 contains no state of its own. The field is the computational locus.***

---

## Nearest Neighbors

Oblivious Compute can be approached through three nearby ideas. Invert Dijkstra’s use of local state relations, borrow the broadcast move from Broadcast Consensus Protocols, and apply the resulting mechanism at the field scale explored by Field Calculus. What remains is a smaller kernel: observers contain state, admissibility is positional, and the relational field becomes the computational locus.

### Dijkstra

In *Self-Stabilizing Systems in Spite of Distributed Control*, Dijkstra showed how finite-state machines can act from local relationships between their own state and the states of their neighbors while driving a distributed system toward a legitimate global condition.

Dijkstra’s constructions use asymmetry to control the evolution of the system. Oblivious Compute keeps the local relational mechanism but inverts its role: symmetry is no longer merely something to break, restore, or manage on the way to legitimacy. The relational symmetry itself becomes the distributed computational object.

### Broadcast Consensus Protocols

Broadcast Consensus Protocols replace pairwise interaction with broadcast transitions, allowing a state change to be exposed across a population.

Oblivious Compute borrows the broadcast move but removes the computationally designated recipient. A state is projected into the communication environment; any observer that encounters it independently determines its significance from its own position.

### Field Calculus

Field Calculus moves distributed computation upward from individual devices toward computations over fields, treating collective behavior as an object of computation in its own right.

Oblivious Compute applies the inverted local mechanism at that scale, then removes state from the field itself. Observers contain state. The field exists in the relationship among those states.

***Dijkstra supplies the local relation. Broadcast removes the designated recipient. Field Calculus supplies the scale. Oblivious Compute moves the computational locus into the relation itself.***

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
