# Appendix A — Implementations

## Purpose

The Oblivious Compute axioms define the computational model. The following reference implementations demonstrate progressively more capable realizations of that model.

Skeleton demonstrates the computational primitive.

Hydra demonstrates the primitive operating as a distributed system.

Byzantium demonstrates the primitive operating in adversarial environments.

These implementations are published as executable reference implementations and are intended to be independently inspected, executed, modified, and evaluated.

---

## Skeleton

[**`Skeleton`**](../../Skeleton/README.md) is the minimal reference implementation of Oblivious Compute. It demonstrates the computational primitive using the fewest possible assumptions. Networking, persistence, synchronization, and security are intentionally omitted so that the axioms may be examined in their simplest executable form.

---

## Hydra

[**`Hydra`**](../../Hydra/README.md) extends Skeleton into a distributed implementation capable of synchronizing multiple participating cells. It demonstrates that the Oblivious Compute primitive can operate across a distributed system while preserving the same axiomatic foundation. Readers interested in distributed operation, synchronization, and convergence should evaluate the Hydra implementation.

---

## Byzantium

[**`Byzantium`**](../../Byzantium/README.md) extends Hydra into adversarial environments through authentication, verification, and Byzantine-resistant behavior. The computational primitive remains unchanged while additional implementation requirements permit secure operation in hostile environments. Readers interested in the security properties of the framework should evaluate the Byzantium implementation.

---

## Evaluation

This paper presents the theoretical foundation of Oblivious Compute. Questions regarding implementation behavior should be evaluated against the published reference implementations.

The reference implementations are intended to facilitate independent inspection, execution, testing, and analysis. Claims regarding distributed behavior, convergence, synchronization, or adversarial operation should be evaluated by examining and executing the corresponding implementation rather than inferred solely from the abstract axioms.

---

**Continue to** [**`Appendix B`**](./B.md)

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
