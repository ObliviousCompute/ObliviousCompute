# Appendix D — Correspondence

## Purpose

This appendix establishes Oblivious Compute in its own symbolic terms before placing it beside established computational models. The comparison begins from the primitive itself: an observer occupies a present state, a projected continuation appears, and admissibility is determined from that position.

***These are correspondences, not equivalences.*** Each section identifies a shared structure and the point at which the models diverge.

## Notation

The equations below describe the kernel used throughout this appendix. $A$ is the local admissibility test performed at the current event horizon: the present does not derive its authority from an agreed, merged, or reconstructed history.

| Symbol | Meaning |
|--------|---------|
| $s$ | Present observer state / position |
| $s'$ | Admitted successor state / position |
| $x$ | Projected state, input, or continuation |
| $\Omega$ | State space |
| $A(s,x)$ | Local admissibility test at the current event horizon |
| $i,j$ | Observer indices |
| $n$ | Number of independently maintained observer positions |
| $\Sigma$ | Relational symmetry among observer positions |
| $\mathcal{F}$ | Field / distributed computational locus |

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

### Dijkstra-1974

In *Self-Stabilizing Systems in Spite of Distributed Control*, Dijkstra showed how finite-state machines can act from local relationships between their own state and the states of their neighbors while driving a distributed system toward a legitimate global condition.

Dijkstra’s constructions use asymmetry to control the evolution of the system. Oblivious Compute keeps the local relational mechanism but inverts its role: symmetry is no longer merely something to break, restore, or manage on the way to legitimacy. The relational symmetry itself becomes the distributed computational object.

### Broadcast Consensus Protocols-2019

Broadcast Consensus Protocols replace pairwise interaction with broadcast transitions, allowing a state change to be exposed across a population.

Oblivious Compute borrows the broadcast move but removes the computationally designated recipient. A state is projected into the communication environment; any observer that encounters it independently determines its significance from its own position.

### Field Calculus-2019

Field Calculus moves distributed computation upward from individual devices toward computations over fields, treating collective behavior as an object of computation in its own right.

Oblivious Compute applies the inverted local mechanism at that scale, then removes state from the field itself. Observers contain state. The field exists in the relationship among those states.

***Dijkstra supplies the local relation. Broadcast removes the designated recipient. Field Calculus supplies the scale. Oblivious Compute moves the computational locus into the relation itself.***

---

**Go back to [**`Theory`**](../README.md)...**

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
