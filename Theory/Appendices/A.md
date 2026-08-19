# Appendix A — Correspondence

## Purpose

This appendix establishes [**`Oblivious Compute`**](https://github.com/ObliviousCompute) in its own symbolic terms before placing it beside established computational models. The comparison begins from the primitive itself: an observer occupies a present state, a projected continuation appears, and admissibility is determined from that position.

***These are correspondences, not equivalences.*** Each section identifies a shared structure and the point at which the models diverge.

## Notation

The equations below describe the kernel used throughout this appendix. 𝓐 is the local admissibility test performed at the current event horizon: the present does not derive its authority from an agreed, merged, or reconstructed history.

| Symbol | Meaning |
|--------|---------|
| 𝓐 | Admissibility function |
| $\Omega$ | State space |
| $\Sigma$ | Relational symmetry |
| 𝓕 | Computational field |

---

## Oblivious Compute

***I’m here. Something appears. Does it belong?***

Oblivious Compute distributes a single admissibility function 𝓐 across a set of independently state-maintaining observers within a state space $\Omega$. No observer requires a computationally designated neighbor; each evaluates presented state from its own position. Together, these local determinations form a matrix of relations across the observer set.

$\Large 𝓐:\Omega\times\Omega\rightarrow\lbrace 0,1\rbrace \qquad 𝓐(s,x)\in\lbrace 0,1\rbrace$

The same presented state may therefore be admissible from one observer position and inadmissible from another.

$\Large s_i\neq s_j \qquad\Longrightarrow\qquad 𝓐(s_i,x)=1,\quad 𝓐(s_j,x)=0$

Across the observer set, let $\Sigma$ denote relational symmetry within the resulting matrix of relations. That symmetry constitutes the computational field $𝓕$.

$\Large 𝓕\equiv\Sigma(s_1,s_2,\ldots,s_n)$

***No observer contains the field. The field contains no state. It exists only through the symmetry between independently maintained states.***

---

## Nearest Neighbors

Take Dijkstra’s local state relations and strip away temporal direction with Borrill. Remove pairwise interaction with broadcast consensus and move computation to field scale with field-based coordination. What remains is a small oblivious kernel.

Observers hold state, projection requires no computationally designated recipient, admissibility is positional, and the relation among observer states becomes the computational locus.

### Borrill-2026

In *Message Passing Without Temporal Direction: Constraint Semantics and the FITO Category Mistake*, Paul Borrill removes temporal direction from the semantic foundation of message passing and reformulates interaction through compatibility constraints among local states.

Borrill comes remarkably close to the relational structure of Oblivious Compute, but retains a global state space formed from local-state valuations and treats compatibility as a constraint over that space. Oblivious Compute removes the additional global state object: observers hold state, while the relation among those states is itself the distributed computational object.

### Field-Based Coordination-2025

In *FBFL: A Field-Based Coordination Approach for Data Heterogeneity in Federated Learning*, Domini, Aguzzi, Esterle, and Viroli describe computational fields as distributed data structures that associate agents with computational values. Agents possess state, interact through defined neighborhoods, and collectively compute over those fields.

Oblivious Compute operates at the same collective scale but separates state from field. Observers hold state; those states do not constitute the field. Symmetry among independently maintained observer states produces the field, and the field contains no state of its own.

### Broadcast Consensus Protocols-2019

*Expressive Power of Broadcast Consensus Protocols* replaces exclusively pairwise population interactions with reliable broadcast transitions, allowing an agent to expose a state change across the population.

Oblivious Compute borrows the broadcast move but removes the computationally designated recipient. A state is projected into the communication environment; any observer that encounters it independently determines its significance from its own position.

### Dijkstra-1974

In *Self-Stabilizing Systems in Spite of Distributed Control*, Dijkstra showed how finite-state machines can act from local relationships between their own state and neighboring states while driving a distributed system toward a legitimate global condition.

Oblivious Compute retains Dijkstra’s local relational principle while removing the need for computational adjacency.

---

**Continue to [**`Appendix B`**](./B.md) go to [**`Theory`**](../../Theory/README.md) or check out [**`Skeleton`**](../../Skeleton/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
