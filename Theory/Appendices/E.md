# Examination

## Computational Advantage

[**Oblivious Compute**](https://github.com/ObliviousCompute) explores whether a computational advantage can be gained by **trading inexpensive communication for simpler local computation**. A single recurring rule, distributed across independently maintained states, may produce collective behavior without requiring any observer to coordinate the machine as a whole.

The proposed leverage comes from **the arrangement of those states and their interactions**. Additional communication through the shared medium may reduce the computational work required elsewhere. Whether this trade produces a net advantage remains to be measured.

## Mechanical Advantage

> Think about how pulleys work. By arranging pulleys together, you can lift the same weight with less force at the cost of pulling more rope. An oblivious machine explores a similar possibility. **The medium provides the rope, and the observers supply the arrangement.** By spending more of one resource, we may gain leverage over another.

## The Inversion Challenge

The central examination is not whether another formalism can reproduce the observable behavior of HaltingMachine. A sufficiently expressive system may be able to reproduce the same sequence of states or outputs through a different construction.

The question is whether that reconstruction instantiates **the same computational object**. HaltingMachine provides a compact test because independently maintained observers progress through a shared state space while an inverter traverses that space in the opposite direction. An observer may halt its independent stimulus while continuing to participate in collective evolution through admitted projections and reprojection.

The challenge is therefore to reconstruct the machine within another formalism without quietly replacing its computational object with a different one. A reconstruction that produces equivalent outputs is informative, but it is not sufficient by itself. The examination asks what the formalism actually has to represent, transmit, merge, construct, or coordinate in order to produce those outputs.

**Computational Object**

> The computational object proposed here is the evolving relation represented by $\Sigma_{M_s}$, manifested through independently maintained observer states. No observer contains the complete object, and the medium need not define or maintain a separate representation of it. A projection is a view of that evolving object from an observer's state. When an admitted projection mutates an observer's state, the resulting state may itself be reprojected, allowing the same evolving object to continue across observers.
>
> The examination therefore distinguishes **reproducing the behavior of a computation** from **instantiating the computational object that performs the computation**.

## Peers

### State-based CRDTs

In [*A Comprehensive Study of Convergent and Commutative Replicated Data Types*](https://dsf.berkeley.edu/cs286/papers/crdt-tr2011.pdf), Shapiro, Preguiça, Baquero, and Zawirski formalize replicated objects whose independently modified replicas converge under state-based or operation-based conditions. State-based CRDTs provide the closest direct comparison because their formal object is replicated state.

**Point of contact:** Independent replicas maintain state and exchange state without foreground synchronization.

**Examination:** Can a state-based CRDT reconstruct HaltingMachine while preserving the same evolving relational computational object, rather than encoding its behavior into a replicated data structure? Account explicitly for every additional state, merge rule, ordering requirement, or other mathematical machinery required by the reconstruction.

### Tuples On The Air

In [*Tuples On The Air: A Middleware for Context-Aware Computing in Dynamic Networks*](https://iris.unimore.it/handle/11380/18833), Mamei, Zambonelli, and Leonardi use spatially distributed tuples to represent contextual information and support uncoupled interactions. Tuples propagate according to application-specific patterns and can form distributed computational fields.

**Point of contact:** A projected object can move through a distributed environment and be encountered by independently operating components.

**Examination:** Can Tuples On The Air reconstruct HaltingMachine while treating the projected state itself as the continuing computational object, rather than as a tuple that is subsequently interpreted by a separate mechanism? Account explicitly for every propagation, storage, reaction, or coordination mechanism required.

### Field Calculus

In [*From Distributed Coordination to Field Calculus and Aggregate Computing*](https://doi.org/10.1016/j.jlamp.2019.100486), Viroli et al. develop the field-calculus lineage as a formal model for specifying and composing collective behavior. The later [*The eXchange Calculus*](https://doi.org/10.1016/j.jss.2024.111976) extends this lineage with a single exchange construct combining computation, communication, and state over time.

**Point of contact:** Computation is represented at the scale of a collective rather than only as isolated device computation.

**Examination:** Can Field Calculus, including the exchange mechanism of the eXchange Calculus, instantiate the same computational object $\Sigma_{M_s}$ without introducing a separately constructed field representation whose semantics supply the collective computation? Account explicitly for every neighborhood, communication, state, alignment, or execution mechanism required.

### Broadcast Consensus Protocols

In [*Expressive Power of Broadcast Consensus Protocols*](https://doi.org/10.4230/LIPIcs.CONCUR.2019.31), Blondin, Esparza, and Jaax study anonymous finite-state agents extended with reliable global broadcasts and characterize the computational power of the resulting population.

**Point of contact:** Agents can participate in collective computation through a shared communication mechanism without requiring individually addressed recipients.

**Examination:** Can Broadcast Consensus Protocols reconstruct the same computational object under an undefined medium, where the medium supplies only an opportunity for observation and does not itself define the collective computation? Account explicitly for every broadcast, transition, delivery, population, or coordination assumption required.
