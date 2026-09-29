# Examination

## The CRDT Challenge

At first glance, **Oblivious Compute** resembles a CRDT. Independent observers maintain local state, encounter information from elsewhere, and may converge. The resemblance is real, but convergence alone is not the computational object proposed here.

A [CRDT](https://dsf.berkeley.edu/cs286/papers/crdt-tr2011.pdf) specifies a replicated data type whose state or operations satisfy conditions for convergence. Oblivious Compute begins instead with a state space $\Omega$, a positional admissibility relation $\mathcal A(s,x)$, and a shared medium. Observers independently decide which encountered projections belong from their present states. The proposed computation is the evolving relational symmetry among those independently maintained states, not a value maintained by any one observer.

[**HaltingMachine**](../../Spark/Halt/README.md) makes this distinction testable. Ordinary observers progress through Rock, Paper, Scissors while an inverter traverses the same state space in reverse. An observer can **HALT its independent stimulus** yet continue to admit incoming projections, mutate, and reproject. Halting one observer's stimulus is not necessarily halting its participation in the collective evolution.

> **The CRDT Challenge:** Reconstruct HaltingMachine as a CRDT while preserving independent stimulus, HALT, inversion, positional admission, and mutation-triggered reprojection. Identify which behavior belongs to the replicated data type and which requires additional transition rules, event handling, or coordination.

This is a comparison to perform, not an impossibility claim. A sufficiently expressive CRDT-based application may reproduce the behavior. The question is **where the computation occurs in that reconstruction**, and what machinery must be introduced to preserve the same behavior rather than merely reach a similar final value.

## Points of Contact

An oblivious machine begins with independent observers, an admissibility rule, and a **shared medium**. An observer projects without designating a computational recipient. Any observer that encounters the projection evaluates it from its own position. An admitted change can trigger another projection. No observer must maintain a computational peer list or an authoritative representation of the collective field.

The medium provides **common opportunity for observation, not common authority**. It carries projections, but does not decide their meaning. Its physical implementation still has to deliver the required opportunities for observation. The computational abstraction does not eliminate networking, guarantee delivery through partitions, or make broadcast unique to Oblivious Compute. Its proposed distinction is **the combination of recipient-oblivious projection, independent positional admission, and the evolving relation among observer states as the computational object**.

The five comparisons below examine different parts of that construction: **local decisions and global conditions, complex behavior from simple rules, collective field semantics, broadcast interaction, and synchronization geometry**. They are points of contact, not ingredients claimed as inventions or an ordered measure of proximity.

## Peers

### Dijkstra · Local Control

In [*Self-Stabilizing Systems in Spite of Distributed Control* (1974)](https://www.cs.utexas.edu/~EWD/transcriptions/EWD04xx/EWD426.html), Dijkstra demonstrates that local rules can bring a distributed system into a globally legitimate condition even when its complete state is not held in a shared store. His example explicitly assumes processes communicating with neighbors.

**Point of contact:** independently controlled local actions can produce a property of the whole system. **Question to examine:** Dijkstra specifies a global legitimacy condition that the algorithm is designed to reach. Oblivious Compute instead identifies the evolving relation among independently maintained states as the computation, including configurations away from perfect symmetry. Its computational interface also does not assign neighbors as recipients of individual projections.

### Wolfram · Cellular Automata

In [*Statistical Mechanics of Cellular Automata* (1983)](https://doi.org/10.1103/RevModPhys.55.601), Stephen Wolfram investigates how simple local transition rules generate collective patterns and complex behavior. His elementary cellular automata evolve through discrete steps using fixed nearest-neighbor relationships.

**Point of contact:** simple local rules can generate behavior that is visible only at the level of a larger configuration. **Question to examine:** the classical cellular-automaton construction uses a prescribed neighborhood and update scheme. Oblivious Compute exposes projections through a shared medium, then lets each observer determine admissibility from its own state. More general cellular-automaton models may narrow this distinction, so the relevant comparison is the actual transition and observation semantics, not merely the presence or absence of a grid.

### Pereira et al. · Synchronization Geometry

In [*Towards a Theory for Diffusive Coupling Functions Allowing Persistent Synchronization* (2014)](https://doi.org/10.1088/0951-7715/27/3/501), Pereira, Eldering, Rasmussen, and Veneziani study coupled dynamical systems and conditions supporting stable synchronization. Fully synchronized configurations lie on a diagonal in the product state space.

**Point of contact:** independently represented subsystems can coincide on a synchronization diagonal. In the notation of Oblivious Compute, $n$ observer states occupy $\Omega^n$, and perfect symmetry intersects $\Delta_n(\Omega)\cong\Omega$. **Question to examine:** Pereira et al. investigate the stability of synchrony under coupling. In Oblivious Compute, the diagonal is only one possible configuration of a field that exists before, during, and after alignment. Neither the shared geometry nor the admissibility rule alone proves that arbitrary implementations will converge.

### Field Calculus · Collective Computation

[Viroli, Beal, Damiani, Audrito, Casadei, and Pianini (2019)](https://doi.org/10.1016/j.jlamp.2019.100486) develop the foundations of field calculus and aggregate computing: collective behavior expressed through computational fields, with corresponding local execution semantics. A computational field is a mathematical description, not necessarily a separately stored global object.

**Point of contact:** computation is understood at the scale of a collective rather than only as a collection of isolated outputs. **Question to examine:** field calculus expresses and executes collective computations through its language and device interactions. Oblivious Compute proposes positional admission and reprojection as the primitive, with the field defined as relational symmetry among observer states. Can one faithfully express the other, and if so, which semantics and mechanisms must be supplied?

### Broadcast Consensus · Shared Projection

In [*Expressive Power of Broadcast Consensus Protocols* (2019)](https://doi.org/10.4230/LIPIcs.CONCUR.2019.31), Blondin, Esparza, and Jaax study anonymous finite-state agents extended with reliable global broadcasts. This is a direct comparison for any claim involving a population that communicates without individually addressing ordinary peers.

**Point of contact:** global broadcast permits distributed interaction without maintaining pairwise computational recipient lists. **Question to examine:** broadcast consensus protocols define agent transitions through broadcast actions and study the predicates a population can compute. Oblivious Compute treats the medium as an opportunity to encounter projections, leaving admission to each observer's current position and identifying the evolving relation as its computational object. The comparison must account for different delivery assumptions, transition semantics, and computational goals. The absence of a peer list is **not**, by itself, a distinction from broadcast consensus.

> **Description is not construction.** A formalism may describe a collective configuration without specifying the same mechanism that produces its evolution. The comparison worth making is between the actual machines, not just the shapes of their resulting states.

---

🧭 **[**`EXIT`**](https://github.com/ObliviousCompute)...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it, just respect the terms outlined there.
