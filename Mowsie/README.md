# 🐁 Mowsie 🐁

Mowsie is the first real-world application of Oblivious Compute.

Start with [`Genesis`](./Genesis.md) to see how salt comes into existence.  
After that, check out the [`Attack Surface`](./AttackSurface.md) analysis to see how it holds up.

---

## 🧀 The Idea

Could Mowsie replace punch cards, gift cards, and loyalty systems with shared state?

What if accounts, fees, and backends just… disappeared?

What remains is a name, a password, and value that moves instantly.

---

## 🧬 What Mowsie Is

Mowsie is a shared state that moves value.

It’s so small that even with around a hundred people, a single state is about the size of a small emoji.

Think of it like a tiny surface where value simply shifts from one place to another.

That’s the surface nodes share.

A transaction is a small set of instructions that rearranges the surface.

**Sending value is simply changing the state's surface.**

---

## ✨ How It Feels

Imagine a little shop or studio.

You walk in. You scan a QR code or select a cache you’ve already scanned.

You enter a name and password.

Your stash appears instantly — always the same for that name, password, and cache.

Within about a second, the state catches up.

If your stash holds value, it’s already waiting for you in the state.

No accounts. No setup. No flows.

You don’t create anything.

You just open your stash.

---

## ☕ A Real Example

A coffee shop creates their cache with one million salt.

Cashiers start the day with float. Customers exchange dollars for salt. Later, they spend salt for goods.

A returning customer selects the same cache, enters their name and password, and their stash resolves again.

If it holds value, it appears. If not, it remains empty until funded.

There is no backend tracking balances.

There is only shared state.

---

## 🗑️ What This Replaces

Punch cards get lost. Gift cards are forgotten. Loyalty apps depend on databases that must be trusted and maintained.

At their core, they’re all just ways of coordinating value within a group.

Each group could simply share the same surface.

---

## 🧩 The Pieces of Mowsie

> Value is called **salt**.  
>  
> **Salt** lives inside **stashes**.  
>  
> **Stashes** exist within **caches**.  
>  
> **Salt** moves and can carry **messages**.  
>  
> The system is made visible by **lanterns**.

---

### 🧂 Salt

Salt is the unit of value in Mowsie.

It is finite, transferable, and cryptographically secured.

It is a blank canvas. People decide what it means — messages, points, credits, passes.

They can call it whatever they want, but underneath, it is always salt.

---

### 🗝️ Caches & Stashes

A cache is the shared state — a small domain that contains the surface where value lives and moves.

Think of it like a tiny map.

A stash is a specific spot on that map.

That spot is always there, but it only becomes visible when it holds value.

Your name and password act like a secret location on the map — a pin only you can find.

Enter them again, and the same spot opens every time.

Value moves within a cache by shifting between these spots.

---

### 💬 Messages

Transactions can carry a short message, up to 60 characters.

Sometimes it’s something simple — a coffee, a class, a double cheeseburger. Sometimes it’s nothing at all.

When value moves, a message can move with it. It might act like a receipt, or just a note between two people. It doesn’t have to be anything more than that.

Everything inside a cache is encrypted. From the outside, it’s just motion — packets moving, nothing readable. From the inside, it resolves into something meaningful.

And sometimes, it’s not about the value at all.

Two people — or a small group — can share a cache and use it however they want.

---

### 🏮 Lanterns

Lantern nodes provide visibility.

They relay packets. They do not decide truth. They do not store history. They do not enforce ordering.

They just let the system be seen.

---

## 🌐 How This Exists

Mowsie is built from primitives already demonstrated in [`Byzantium`](../Byzantium).

Cells become accounts. Salt becomes value. Whispers becomes transactions. Dream becomes state. Crypt becomes transport.

Nothing new is happening here.

This is the simplest transaction possible — now routed over the internet.

---

## 🧠 Why This Is Different

There is no history. No ordering. No consensus. No fees.

Invalid state does not propagate. It fails immediately.

Even if an attacker succeeds, they can only delay visibility. They cannot change truth.

---

## 🛠️ What We’re Building

We are defining the leaf structure, Merkleizing the state, building the mobile client, and implementing Lantern nodes.

The goal is simple.

From download to opening your own private stash and receiving value in under one minute.

---

## 📡 Contact

If you see it and want to talk, reach out — ObliviousCompute@yahoo.com

---

<h2><img src="../Relics/BTC.png" width="25"/> Support</h2>

> **bc1qc69hm4smfvp4q2xwrn95926ljztxahe0q7fa8x**

---

Mowsie replaces loyalty systems with shared state.  
It is the wooden nickel of the 21st century.

---

## 📜 License

This project is released under the terms of the [`LICENSE`](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
