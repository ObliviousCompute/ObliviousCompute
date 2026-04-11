# 🐁 Mowsie

Mowsie is the first real-world application of Oblivious Compute.

Start with [`Genesis`](./Genesis.md) to see how salt comes into existence.  
After that, check out the [`Attack Surface`](./AttackSurface.md) document to see how it holds up.  

---

## 💡 The Idea

Can Mowsie replace punch cards, gift cards, and loyalty systems with shared state?

What if accounts, fees, and backends just… disappeared?

What remains is a name, a password, and value that moves instantly.

---

## ✨ How It Feels

Imagine a little shop or studio.

You walk in. You scan a QR code. You type a name.

Within seconds, you have a wallet.

The cashier sends you value. Your balance updates instantly.

No waiting. No syncing. No confirmation window.

You just receive value.

---

## 🗑️ What This Replaces

Punch cards get lost. Gift cards are forgotten. Loyalty apps depend on databases that must be trusted and maintained.

Mowsie removes all of that and replaces it with a shared, verifiable state — a cache visible to everyone inside it.

---

## 🧬 What Mowsie Is

Each vendor creates a *Cache* with a fixed supply of value.

That supply does not change. It cannot be inflated or silently modified.

Once it exists, it becomes the law of that cache.

Everything that follows must obey it.

---

## 🧂 Salt

Salt is the unit of value in Mowsie.

It is finite, transferable, and cryptographically secured.

It is a blank canvas. Vendors decide what it means — points, credits, passes. They can call it whatever they want, but underneath, it is always salt.

---

## 🛍️ A Real Example

A coffee shop creates thier cache with one million salt.

Cashiers start the day with float. Customers exchange dollars for salt. Later, they spend salt for goods.

There is no backend tracking balances.

There is only shared state.

---

## 🌐 How This Exists

Mowsie is built from primitives already demonstrated in [`Byzantium`](./Byzantium).

Cells become accounts. Salt becomes value. Whisper becomes transaction. Dream becomes state. Crypt becomes transport.

Nothing new is required. This is a reconfiguration.

---

## 🔥 Lanterns

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

From download to receiving value in under one minute.

---

## 📡 Contact

obliviouscompute@yahoo.com

---

<img src="../Relics/BTC.png" width="25"/>  ## Donate
**bc1qc69hm4smfvp4q2xwrn95926ljztxahe0q7fa8x**

---

Mowsie replaces punch cards with shared state.

It is the wooden nickel of the 21st century.

---

## License

This project is released under the terms of the [`LICENSE`](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.

