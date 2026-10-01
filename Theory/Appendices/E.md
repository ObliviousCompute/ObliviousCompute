# Examination

## Computational Advantage

[**Oblivious Compute**](https://github.com/ObliviousCompute) explores whether a computational advantage can be gained by **trading inexpensive communication for simpler local computation**. The hypothesis is that **a single recurring rule**, distributed across independently maintained states, can produce collective behavior without requiring any observer to coordinate the machine as a whole.

The proposed leverage comes from **the arrangement of those states and their interactions**. Additional communication through the shared medium may reduce the computational work required elsewhere. Whether this trade produces a net advantage remains to be measured.

## Mechanical Advantage

> Think about how pulleys work. By arranging pulleys together, you can lift the same weight with less force at the cost of pulling more rope. An oblivious machine explores a similar possibility. **The medium provides the rope, and the observers supply the arrangement.** By spending more of one resource, we may gain leverage over another.

## The Inversion Challenge

The challenge here is not simply to reproduce the observable behavior of [**`HaltingMachine`**](../../Spark/Halt/README.md). Of course, another sufficiently expressive formalism may be able to reproduce the same sequence of states or outputs while retaining a different computational structure.

Instead, **the goal is to construct an oblivious computation within another formalism**. HaltingMachine provides the test case. Reproduce its inversion, halting, positional admission, mutation, and reprojection using the native mathematical objects and operations of the system under examination. Do not wrap an Oblivious Compute implementation inside an existing formalism or merely reproduce its outputs externally.

If need be, strip away that formalism's separate representations until the irreducible construction remains. Then, from there, show the resulting mathematics. If that reduced construction instantiates the same evolving relational object defined below, the formalism has reached an oblivious computation. If additional machinery remains necessary, identify precisely what that machinery contributes.

***The question is, what must be removed from another formalism before its separately represented components cease to be separate and the evolving relation among states becomes the computation itself?***

> **The Computational Object** is the evolving relation represented by $\Sigma_{M_s}$, manifested through independently maintained observer states. All observer states and admissible projections are expressed within the common state space $\Omega$. No observer contains the complete object, and the medium need not define or maintain a separate representation of the field. A projection is a manifestation of the evolving object from an observer's state. When an admitted projection mutates an observer's state, the resulting state may itself be reprojected, allowing the same evolving object to continue across observers.
>
> **No second computational object is introduced between these manifestations.** What appears as a state, projection, message, tuple, replica, or field in another formalism is, in this construction, a manifestation of the same evolving relational object. The medium and topology provide conditions for those manifestations to encounter one another, but they do not constitute separate computational objects.

***Our wager is that, once the machinery is stripped away, something very, very similar in shape remains.***

## Peers

### State-based CRDTs

In [*A Comprehensive Study of Convergent and Commutative Replicated Data Types*](https://dsf.berkeley.edu/cs286/papers/crdt-tr2011.pdf), Shapiro, Preguiça, Baquero, and Zawirski formalize replicated objects whose independently modified replicas converge under state-based or operation-based conditions. State-based CRDTs provide a close comparison because their formal object is replicated state.

**Point of contact:** Independent replicas maintain state and exchange state without foreground synchronization.

**Examination:** Construct an oblivious computation within the state-based CRDT formalism using HaltingMachine as the test case. Express the construction using native CRDT mathematics. Then identify what must be removed before replicated state, merge, and any other CRDT-specific machinery cease to be separate computational objects and the evolving relation itself becomes the computation.

### Tuples On The Air

In [*Tuples On The Air: A Middleware for Context-Aware Computing in Dynamic Networks*](https://iris.unimore.it/handle/11380/18833), Mamei, Zambonelli, and Leonardi use spatially distributed tuples to represent contextual information and support uncoupled interactions between distributed components.

**Point of contact:** Information can be projected into a distributed environment and encountered by independently operating components.

**Examination:** Construct an oblivious computation within the tuple formalism using HaltingMachine as the test case. Express the construction using native tuple mathematics. Then identify what must be removed before the tuple, propagation rule, local tuple space, neighborhood structure, or middleware cease to be separate computational objects and the evolving relation itself becomes the computation.

### Field Calculus

In [*From Distributed Coordination to Field Calculus and Aggregate Computing*](https://doi.org/10.1016/j.jlamp.2019.100486), Viroli et al. develop the Field Calculus lineage as a formal model for specifying and composing collective behavior. The later [*The eXchange Calculus*](https://doi.org/10.1016/j.jss.2024.111976) develops this lineage further by combining computation, communication, and state over time within a single exchange construct.

**Point of contact:** Computation is understood at the scale of a collective rather than only as a collection of isolated outputs.

**Examination:** Construct an oblivious computation within Field Calculus using HaltingMachine as the test case, including the exchange mechanism described by the eXchange Calculus where appropriate. Then identify what must be removed before the separately represented field, neighborhood, communication, state, or execution semantics cease to be separate computational objects and the evolving relation itself becomes the computation.

### Broadcast Consensus

In [*Expressive Power of Broadcast Consensus Protocols* (2019)](https://doi.org/10.4230/LIPIcs.CONCUR.2019.31), Blondin, Esparza, and Jaax study anonymous finite-state agents extended with reliable global broadcasts.

**Point of contact:** Agents can participate in collective computation through a shared broadcast mechanism without requiring individually addressed ordinary recipients.

**Examination:** Construct an oblivious computation within the Broadcast Consensus formalism using HaltingMachine as the test case. Express the construction using native broadcast-consensus mathematics. Then identify what must be removed before broadcast actions, agent transitions, population structure, or other separately represented machinery cease to be separate computational objects and the evolving relation itself becomes the computation.

---

🧭 **[**`EXIT`**](https://github.com/ObliviousCompute)...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it, just respect the terms outlined there.
