# Attack Surface

*What can go wrong — and why it doesn’t.*

Mowsie does not secure itself through **consensus**, **history**, or **ordering**.

It secures itself through **admissibility**.

Every node independently evaluates whether a proposed state transition is lawful. If it is not, it is rejected immediately.

There is no coordination required.

**It simply fails to exist.**

---

## Transactions

Every change to the system is a transaction.

A valid transaction preserves the total amount of salt, is properly signed, and only modifies the sender’s state. It must also fit within the structure of the system.

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

**Only valid state survives.**

---

## The Shape of the System

Only valid, funded state exists.

If a stash holds no salt, it disappears the next time state is shared. The system retains only economically active participants.

Every element of state must be well-formed. Keys must belong to the system, signatures must verify, and balances must preserve the total supply.

Any state that violates these rules is incompatible and rejected.

The system does not repair invalid state.

**It refuses it.**

---

## Adversarial Behavior

Sybil and Byzantine behavior reduce to the same constraint: an attacker can only submit state.

They may create many identities or attempt malicious transitions, but every action must still follow the system’s rules.

In practice, this reduces most attacks to repeatedly submitting malformed or incompatible state.

It is seen, evaluated, and discarded.

**Invalid state does not persist.**

---

## Visibility Attacks

The remaining attack surface is not correctness, but visibility.

An attacker may attempt to isolate a user, delay updates, or control what state they see. This includes eclipse and delay attacks.

The goal is to convince a user to accept incomplete or incorrect state.

However, state is not accepted blindly. Each participant expects to see their own value reflected in the system.

If it is missing, the state is rejected and refreshed.

Incorrect or delayed state does not propagate.

**It is replaced.**

---

## Final Thoughts

Mowsie removes entire classes of attack by design.

There is no ordering to manipulate, no history to replay, and no consensus to break.

What remains is a system where invalid state is rejected locally, valid state propagates naturally, and attackers are constrained to lawful behavior.

**Truth is not negotiated.**

**It is admitted.**
