# Examination

## Computational Advantage

[**Oblivious Compute**] explores whether a computational advantage can be gained by **trading inexpensive communication for simpler local computation**. The hypothesis is that a single recurring rule, distributed across independently maintained states, can produce collective behavior without requiring any observer to coordinate the machine as a whole.

The proposed leverage comes from the arrangement of those states and their interactions. Additional communication through the shared medium may reduce the computational work required elsewhere. Whether this trade produces a net advantage remains to be measured.

## Mechanical Advantage

> Think about how pulleys work. By arranging pulleys together, you can lift the same weight with less force at the cost of pulling more rope. An oblivious machine explores a similar possibility. **The medium provides the rope, and the observers supply the arrangement.** By spending more of one resource, we may gain leverage over another.

## The Inversion Challenge

The challenge is not simply to reproduce the observable behavior of HaltingMachine. A sufficiently expressive formalism may be able to simulate the same behavior while retaining a different computational structure.

Instead, construct an **oblivious computation within the formalism itself**. HaltingMachine provides the test case. Reproduce its inversion, halting, positional admission, mutation, and reprojection using the native mathematical objects and operations of the system under examination. Do not wrap an Oblivious Compute implementation inside the competing formalism or merely reproduce its outputs externally.

Then reduce the construction. Identify what must be removed from the original formalism before its separately represented states, messages, tuples, replicas, field structures, topology, or medium cease to be separate computational objects. Show the resulting mathematics. If the reduced construction instantiates the same evolving relational object, then the formalism has reached an oblivious computation. If additional machinery remains necessary, identify precisely what that machinery contributes.

**Computational Object**

> The evolving relation represented by $\Sigma_{M_s}$, manifested through independently maintained observer states, is the object in question. All observer states and admissible projections are expressed within the common state space $\Omega$. No observer contains the complete object, and the medium need not define or maintain a separate representation of the field. A projection is a manifestation of the object from an observer's state. When an admitted projection mutates an observer's state, the resulting state may itself be reprojected, allowing the same evolving object to continue across observers.
>
> The question is therefore not whether another formalism can reproduce the behavior of an oblivious computation. The question is what must be removed from that formalism before the state, projection, message, tuple, replica, field, topology, and medium cease to be separately represented computational objects and the evolving relation among states becomes the computation itself.

## Peers

### State-based CRDTs

In [*A Comprehensive Study of Convergent and Commutative Replicated Data Types*], Shapiro, Preguiça, Baquero, and Zawirski formalize asynchronous replicated objects and provide the state-based conditions under which independently modified replicas converge.

**Point of contact:** Independent observers maintain state and can exchange state without foreground synchronization.

**Inversion Challenge:** Construct an oblivious computation within the state-based CRDT formalism using HaltingMachine as the test case. Express the construction using native CRDT mathematics. Then identify what must be removed before replicated state, merge, and any other CRDT-specific machinery cease to be separate computational objects and the evolving relation among states becomes the computation.

### Tuples On The Air

In [*Tuples On The Air: A Middleware for Context-Aware Computing in Dynamic Networks*], Mamei, Zambonelli, and Leonardi use spatially distributed tuples to represent contextual information and support uncoupled interactions between distributed components.

**Point of contact:** Information can be projected into a distributed environment and encountered by independently operating components.

**Inversion Challenge:** Construct an oblivious computation within the tuple formalism using HaltingMachine as the test case. Express the construction using native tuple mathematics. Then identify what must be removed before the tuple, propagation rule, local tuple space, neighborhood structure, or middleware cease to be separate computational objects and the evolving relation among states becomes the computation.

### Field Calculus

In [*From Distributed Coordination to Field Calculus and Aggregate Computing*], the Field Calculus lineage develops a formal model for collective computation through information propagating across device collectives. The later [*The eXchange Calculus*] develops this lineage further by combining computation, communication, and state over time within a single exchange construct.

**Point of contact:** Computation is explicitly considered at the level of a collective rather than only at the level of isolated devices.

**Inversion Challenge:** Construct an oblivious computation within Field Calculus using HaltingMachine as the test case, including the exchange mechanism described by the eXchange Calculus where appropriate. Then identify what must be removed before the separately represented field, neighborhood, communication, state, or execution semantics cease to be separate computational objects and the evolving relation among states becomes the computation.

### Broadcast Consensus Protocols

In [*Expressive Power of Broadcast Consensus Protocols*], Blondin, Esparza, and Jaax study anonymous finite-state agents extended with reliable global broadcasts.

**Point of contact:** Agents can participate in collective computation through a shared broadcast mechanism without requiring individually addressed ordinary recipients.

**Inversion Challenge:** Construct an oblivious computation within the Broadcast Consensus formalism using HaltingMachine as the test case. Express the construction using native broadcast-consensus mathematics. Then identify what must be removed before broadcast actions, agent transitions, population structure, or other separately represented machinery cease to be separate computational objects and the evolving relation among states becomes the computation.

* * *

🧭 [**EXIT**] ...
