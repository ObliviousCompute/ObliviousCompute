# Attack Surface

**What can go wrong — and why it doesn’t.**

---

Mowsie does not secure itself through **consensus**, **history**, or **ordering**.

It secures itself through **admissibility**.

Every node independently evaluates whether a proposed state transition is lawful. If it is not, it is rejected immediately.

There is no coordination required.

**It simply fails to exist.**

---

## Transactions

Every change to the system is a transaction.

A valid transaction preserves the total amount of salt, is properly signed, and only modifies the sender’s state. It must also fit within the structure of the system.

In technical terms, transactions form a **monoid** — they combine cleanly in a single **atomic transition** and always produce a valid next state.

If any of these conditions fail, the transaction is rejected immediately.

There is no pending state, no ordering, and no delay. A transaction is either admitted or discarded.

**Invalid transactions do not propagate.**

---

## Equivocation

Equivocation does not create forks.

**It collapses.**

If an attacker attempts to send two transactions at once, both are evaluated independently.

If both are valid and funded, both are counted.

The attacker is charged twice.

It’s not a double spend — it’s just spending twice.

If a transaction is not funded, it is rejected immediately.

There is no fork, no ambiguity, and no delay.

---

## The Shape of the System

If a stash holds no salt, it disappears the next time state is shared.

The system retains only economically active participants.

Every element of state must be well-formed. Keys must belong to the system, signatures must verify, and balances must preserve the total supply.

Any state that violates these rules is incompatible and rejected.

The system does not repair invalid state.

**It refuses it.**

---

## Adversarial Behavior

Whether an attacker is Sybil or Byzantine, the constraint is the same.

They can only submit state.

The most likely way to attack the system is through visibility — attempting to isolate a user, delay updates, or control what state they see.

During such an attack, an attacker may spam malformed or incomplete state.

However, state is not accepted blindly.

Each participant expects to see their own value reflected in the system.

If it is missing, the state is rejected and refreshed.

Nodes do not form trust relationships with one another. They broadcast toward shared visibility surfaces — **Lanterns**.

This makes eclipse-style attacks significantly harder to achieve, and multiple Lanterns can bypass any single point of interference.

As soon as visibility is restored, the correct state reasserts itself.

The attacker is not followed — they are bypassed.

**Invalid state does not persist.**

---

## Final Thoughts

Mowsie removes entire classes of attack by design.

There is no ordering to manipulate, no history to replay, and no consensus to break.

What remains is a system where invalid state is rejected locally, valid state propagates naturally, and attackers are constrained to lawful behavior.

**Truth is not negotiated.**

**It is admitted.**
