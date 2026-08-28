# Appendix A — Correspondence

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

> ***🧠 Big Brain Brad says:*** *“I’m a plant doing plant things. You can describe me all you want.* ***You’re still not doing plant things.”***

---

**Continue to [**`Axioms`**](./B.md) in Appendix B...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
