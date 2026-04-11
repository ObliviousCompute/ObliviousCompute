# Attack Surface

Mowsie does not secure itself through **consensus**, **history**, or **ordering**.

It secures itself through **admissibility**.

Every wallet is a **validator**. Every node independently evaluates whether a proposed state transition is lawful. If it is not, it is rejected *immediately*. There is no coordination required to reject invalid state.

**It simply fails to exist.**

---

## The Shape of the System

State is a **prunable Merkle set** of funded leaves.

A leaf represents an account: a public key, a lock commitment, and a balance of **salt**. Leaves without balance are removed whenever state is shared. The system only retains *economically active participants*.

At small scale, the system is extremely light.

A network of **1,000 users** produces a state on the order of ~150–200 KB.  
A typical transaction is ~240 bytes on its own, and about ~300 bytes when it includes a **60-character text field**.  
State updates propagate in *milliseconds*.

There is **no historical replay**, no chain growth, and no storage accumulation.

---

## Transactions

Every transaction is a **monoid**.

It must preserve the **total salt invariant**, carry a valid signature, and mutate only the sender’s lock set. If any of these fail, the transaction is rejected.

There is **no mempool**, no ordering, and no pending state.

A transaction is either **admitted immediately** or discarded.

---

## Invariant Enforcement

Each **cache** defines a fixed total supply of **salt** at genesis.

This invariant is **public and constant**.

Any state that violates it is *incompatible* and rejected by every honest node.

Inflation does not spread.

**It creates a separate, incompatible system.**

---

## Equivocation

Equivocation does not create forks.

**It collapses.**

If an attacker produces two transactions from the same parent lock:

- same parent  
- different child  
- both valid signatures  

The network observes both.

Each node deterministically evaluates the competing children and collapses the lock set to a **dominant outcome**. Both transactions are recorded, but only one resulting state survives.

There is **no ambiguity**, no fork choice rule, and no delayed resolution.

Equivocation becomes *visible behavior*, not hidden manipulation.

An attacker attempting equivocation is not breaking the system.

They are performing a **constrained, observable double-spend attempt** that resolves immediately under deterministic rules.

The network does not debate it.

**It collapses it.**

---

## Sybil Resistance

Identities without balance do not persist.

Every time state is shared, empty leaves are **pruned**.

To exist in the system, an identity must hold **salt**.

To attack the system, a Sybil must *fund itself*.

This creates a **direct cost to participation**. Attackers cannot create arbitrary identities without committing value.

---

## Byzantine Behavior

All participants, honest or malicious, are constrained to **lawful state transitions**.

An attacker can only submit transactions that are properly signed, preserve the invariant, and fit within the structure of the system.

They cannot inject arbitrary data or mutate state outside of these rules.

This drastically reduces the *expressive power of an attack*.

---

## Eclipse Resistance

Nodes do not form peer-to-peer trust graphs.

They do not rely on neighbor selection or routing tables.

All nodes orient toward shared visibility surfaces — **Lanterns**.

This removes the typical attack vector of isolating a node by controlling its peers.

To eclipse a user, an attacker must control their **network environment directly**, not just the protocol.

---

## Lanterns

Lantern nodes relay packets.

They do not validate state, enforce consensus, or maintain authority.

Their role is limited to **visibility**.

Because packets are small (~300 bytes) and uniform, Lanterns can cheaply filter malformed or irrelevant traffic.

A Lantern can be run on low-power hardware *(Pi Zero 2W)* with minimal cost, supporting thousands of users.

---

## Packet Constraints

All packets conform to a **strict structure**.

They are small, uniform, and bounded.

An attacker cannot arbitrarily expand payload size or introduce complex data. Any attack must exist within the same constrained packet shape as a normal transaction.

This limits the **surface area for exploitation**.

---

## Delay Attacks

The strongest remaining attack is *delaying visibility*.

An attacker may attempt to disrupt network access, isolate a user from Lanterns, or delay propagation of valid state.

However, this does not corrupt the system.

**It only delays observation.**

Once connectivity is restored, the correct state reasserts itself automatically.

Delay does not propagate.

**It dissipates.**

---

## Key Security

Each account is secured by a **public/private key pair**.

State transitions require valid signatures.

Forging a transaction reduces to forging a valid signature, which is *computationally infeasible* under standard assumptions.

---

## Summary

Mowsie reduces its attack surface by removing entire classes of problems.

There is **no historical replay**, no ordering manipulation, no consensus failure, and no fee market exploitation.

What remains is a system where invalid state is rejected locally, valid state propagates naturally, and attackers are constrained to lawful behavior.

**Truth is not negotiated.**

**It is admitted.**
