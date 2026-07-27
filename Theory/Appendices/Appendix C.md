# Appendix C — Implementations

The following implementations demonstrate progressively more capable realizations of the Oblivious Compute axioms. Each implementation preserves the same foundational model while introducing additional invariants appropriate to its operating environment.

The reference implementations are maintained independently from this paper and serve as executable specifications of the theory.

---

## C.1 Skeleton

### Purpose

Skeleton is the minimal reference implementation of Oblivious Compute.

Its purpose is to demonstrate the computational primitive using the fewest possible assumptions. It contains no networking, cryptography, persistence, consensus, or optimization. Every transition is evaluated exclusively against the present admissible state.

Skeleton serves as the canonical implementation of the five axioms.

### Characteristics

- Single execution context
- No historical reconstruction
- No networking
- No persistence
- Present-state evaluation only
- Minimal implementation complexity

### Core Invariants

- Every transition is evaluated for admissibility.
- Every admissible transition preserves implementation invariants.
- Exactly one current event horizon exists.
- Invalid transitions produce desynchronization.

### Reference Implementation

See the Skeleton reference implementation.

---

## C.2 Hydra

### Purpose

Hydra extends Skeleton into a distributed implementation capable of synchronizing multiple participating cells.

Hydra preserves the same computational primitive while introducing networking, synchronization, persistence, and recovery. These additions do not alter the Oblivious Compute axioms; they introduce implementation-specific invariants required for distributed operation.

### Characteristics

- Distributed execution
- Cell synchronization
- Network transport
- Persistent state
- Recovery from desynchronization
- Multiple participating cells

### Additional Invariants

- Cell identity
- Synchronization integrity
- State propagation
- Persistent recovery
- Network message validity

### Relationship to Skeleton

Hydra is a strict extension of Skeleton. Every Skeleton invariant remains valid while additional invariants support distributed execution.

### Reference Implementation

See the Hydra reference implementation.

---

## C.3 Byzantium

### Purpose

Byzantium extends Hydra for operation in adversarial environments.

While Hydra assumes cooperative participants, Byzantium introduces authentication, verification, and Byzantine-resistant behavior through additional implementation invariants.

The underlying computational primitive remains unchanged.

### Characteristics

- Byzantine-resistant operation
- Authentication
- Digital signatures
- Trust verification
- Adversarial recovery
- Secure synchronization

### Additional Invariants

- Signature validity
- Identity verification
- Authentication requirements
- Byzantine fault handling
- Secure synchronization

### Relationship to Hydra

Byzantium is a strict extension of Hydra. Every Hydra invariant remains valid while additional security invariants permit operation in hostile environments.

### Reference Implementation

See the Byzantium reference implementation.

---

## C.4 Invariant Library

The Oblivious Compute axioms remain fixed across all implementations.

Implementations differ by the invariants they choose to enforce.

### Structural

- Genesis immutability
- Parent-before-child
- Event horizon uniqueness
- Sequence continuity

### Distributed

- Cell identity
- Synchronization validity
- Network integrity
- Message authenticity

### Security

- Signature verification
- Authorization
- Byzantine fault handling
- Identity validation

### Domain-Specific

- G-Counter monotonicity
- OR-Set membership rules
- Financial conservation
- Resource ownership
- Access control
- Application-specific constraints

The invariant library is intentionally open-ended. Additional invariants may be introduced without modifying the underlying Oblivious Compute axioms, provided every admissible transition preserves the selected invariants.
