# Appendix A — Correspondence

## Purpose

This appendix establishes [**`Oblivious Compute`**](https://github.com/ObliviousCompute) in its own symbolic terms before placing it beside established computational models. The comparison begins from the primitive itself: an observer occupies a present state, a projected continuation appears, and admissibility is determined from that position.

Interpret the notation at the resolution being examined. **Perfect symmetry is a realizable settled condition of the field, not its definition.**

> ***These are correspondences, not equivalences.*** Each section identifies a shared structure and the point at which the models diverge.

## Notation

The following equations describe the kernel used throughout this appendix. 𝓐 is the local admissibility test performed at the current event horizon: the present does not derive its authority from an agreed, merged, or reconstructed history.

| Symbol | Meaning |
|--------|---------|
| 𝓐 | Admissibility function |
| Ω | State space |
| Σ | Relational symmetry |
| 𝓕 | Computational field |
| Δ | Diagonal |

---

## Oblivious Compute

$\Large \cdots\Omega\rightarrow\Omega\times\Omega\rightarrow\Omega\leftarrow\Omega\times\Omega\leftarrow\Omega\cdots$

Oblivious Compute distributes a single admissibility function 𝓐 across a set of independently state-maintaining observers within a state space $\Omega$. A computationally designated neighbor is unnecessary. State is projected into an oblivious medium, where any observer that encounters it evaluates the projection from its own position. The medium determines visibility; the observer determines admissibility. Together, these local determinations form a matrix of relations across the observer set.

$\Large 𝓐:\Omega\times\Omega\rightarrow\lbrace 0,1\rbrace \qquad 𝓐(s,x)\in\lbrace 0,1\rbrace$

Therefore, the same presented state *may* be admissible from one observer position and inadmissible from another.

$\Large s_i\neq s_j \Longrightarrow\ 𝓐(s_i,x)=1\quad 𝓐(s_j,x)=0$

Across **$n$** observers, independently maintained states form a configuration in the Cartesian product $\Omega^n$. Let $\Sigma$ denote relational symmetry among those states. That symmetry constitutes the computational field 𝓕.

$\Large (s_1,s_2,\ldots,s_n)\in\Omega^n \qquad 𝓕\equiv\Sigma(s_1,s_2,\ldots,s_n)$

**No observer contains the field. The field contains no state of its own. It exists through symmetry among independently maintained states.** 

As the observer configuration resolves toward perfect relational symmetry, the states coincide and the aggregate configuration lies on the diagonal of the product space.

$\Large s_1=s_2=\cdots=s_n=s \qquad (s_1,s_2,\ldots,s_n)\in\Delta_n(\Omega)\subseteq\Omega^n$

At perfect symmetry, the $n$ observer coordinates no longer vary independently. The diagonal is canonically isomorphic to the original state space.

$\Large \Delta_n(\Omega)\cong\Omega$

**At rest, the distributed configuration reduces to the state space.**

$\Large \Omega$

---

## Nearest Neighbors

***Dijkstra supplies the local relation. Pereira exposes the synchronization diagonal. Alpay reduces synchrony to fewer independent coordinates. Borrill removes temporal direction. Field-based coordination supplies the collective scale. Oblivious Compute moves the computation into the field itself.***

### Dijkstra-1974

In *Self-Stabilizing Systems in Spite of Distributed Control*, Dijkstra showed how finite-state machines can act from local relationships between their own state and neighboring states while driving a distributed system toward a legitimate global condition.

Oblivious Compute retains the local relational principle but removes the requirement that computational adjacency be assigned by the network. Each observer determines admissibility from its own position.

### Pereira-2013

In *Towards a General Theory for Coupling Functions Allowing Persistent Synchronisation*, Pereira, Eldering, Rasmussen, and Veneziani study networks of coupled dynamical systems in which full synchronization satisfies $x_1=x_2=\cdots=x_n=s$. These synchronized configurations form an invariant diagonal manifold within the product state space.

The geometry closely matches the resting configuration of Oblivious Compute. The distinction lies in what the diagonal means: Pereira et al. study synchronization produced by coupled dynamics, while Oblivious Compute reaches the diagonal through relational admissibility among independently maintained observer states.

### Alpay-2025

In *A Topological and Operator Algebraic Framework for Asynchronous Lattice Dynamical Systems*, Alpay models asynchronously evolving subsystems within a stratified state space organized by degrees of synchrony. As synchrony increases, fewer coordinates vary independently, and the fully synchronous stratum is often isomorphic to the state space of a single subsystem.

Oblivious Compute arrives at a similar geometry through a smaller computational mechanism. Observer states occupy $\Omega^n$, while perfect relational symmetry places the aggregate configuration on $\Delta_n(\Omega)\cong\Omega$.

### Field-Based Coordination-2025

In *FBFL: A Field-Based Coordination Approach for Data Heterogeneity in Federated Learning*, Domini, Aguzzi, Esterle, and Viroli use computational fields to coordinate collective behavior across distributed agents. Their fields associate agents with computational values and evolve through structured neighborhood interactions.

Oblivious Compute operates at the same collective scale but separates state from field. Observers maintain state. The field contains no state of its own and exists through relational symmetry among those states.

### Borrill-2026

In *Message Passing Without Temporal Direction: Constraint Semantics and the FITO Category Mistake*, Paul Borrill argues that temporal direction is not fundamental to message-passing semantics and reformulates interaction through compatibility constraints among local states.

Borrill comes remarkably close to the relational semantics of Oblivious Compute. His construction still represents executions as valuations in a global product state space and imposes compatibility constraints over those valuations. Oblivious Compute keeps state in the observers and places the computational locus in the relation among them.

---

**Continue to [**`Appendix B`**](./B.md) go to [**`Theory`**](../../Theory/README.md) or check out [**`Skeleton`**](../../Skeleton/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
