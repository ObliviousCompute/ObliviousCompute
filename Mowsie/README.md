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

## 🗑️ What This Replaces

Punch cards get lost. Gift cards are forgotten. Loyalty apps depend on databases that must be trusted and maintained.

Mowsie replaces loyalty systems with shared state — a cache where stashes appear only when they hold value.

---

## ⚖️ What Mowsie Is

Each vendor creates a cache with a fixed supply of value.

That supply does not change. It cannot be inflated or silently modified.

Once it exists, it becomes the law of that cache.

Everything that follows must obey it.

---

## 🗝️ Stashes

A stash is not stored or created.

It is deterministically derived from three things:

The cache.  
A name.  
A password.

The same inputs will always resolve to the same stash.

If a stash holds salt, it exists in shared state.  
If it does not, it exists only as a possibility until funded.

---

## 🧂 Salt

Salt is the unit of value in Mowsie.

It is finite, transferable, and cryptographically secured.

It is a blank canvas. Vendors decide what it means — points, credits, passes.

They can call it whatever they want, but underneath, it is always salt.

---

## 💬 Messages

Transactions can carry a short message, up to 60 characters.

Sometimes it’s something simple — a coffee, a class, a double cheeseburger. Sometimes it’s nothing at all.

When value moves, a message can move with it. It might act like a receipt, or just a note between two people. It doesn’t have to be anything more than that.

Everything inside a cache is encrypted. From the outside, it’s just motion — packets moving, nothing readable. From the inside, it resolves into something meaningful.

And sometimes, it’s not about the value at all.

Two people — or a small group — can share a cache and use it however they want.

---

## ☕ A Real Example

A coffee shop creates their cache with one million salt.

Cashiers start the day with float. Customers exchange dollars for salt. Later, they spend salt for goods.

A returning customer selects the same cache, enters their name and password, and their stash resolves again.

If it holds value, it appears. If not, it remains empty until funded.

There is no backend tracking balances.

There is only shared state.

---

## 🌐 How This Exists

Mowsie is built from primitives already demonstrated in [`Byzantium`](../Byzantium).

Cells become accounts. Salt becomes value. Whispers becomes transactions. Dream becomes state. Crypt becomes transport.

Nothing new is happening here.

This is the simplest transaction possible — now routed over the internet.

---

## 🏮 Lanterns

Lantern nodes provide visibility.

They relay packets. They do not decide truth. They do not store history. They do not enforce ordering.

They just let the system be seen.

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
