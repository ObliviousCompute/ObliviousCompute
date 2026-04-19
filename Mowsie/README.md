# 🐭 Mowsie 🐭

**The first real-world application of [`Oblivious Compute`](../README.md).**

---

*After reading below*, dive into [`Genesis`](./Genesis.md) to see how *salt comes into existence*.  
Then check out the [`Attack Surface`](./AttackSurface.md) analysis to see how *it holds up*.

---

## 🧬 What This Is

*Imagine* a **shared state** that moves value.

It’s so small that even with around a hundred people, a single state is about the size of a *small emoji*.

Think of it like a *tiny surface* where value simply shifts from one place to another.

That’s the **surface** nodes share.

A **transaction** is a small set of instructions that rearranges the surface.

**Sending value is simply changing the state's surface.**

---

## 🧀 The Idea

Could this replace *punch cards*, *gift cards*, and *loyalty systems* with **shared state**?

What if *accounts*, *fees*, and *backends* just... disappeared?

What remains is a **name**, a **password**, and **value** that moves instantly.

---

## ✨ How It Feels

*Imagine* a little shop or studio.

You walk in. You scan a QR code or select a cache you’ve already scanned.

You enter a **name** and **password**.

Your **stash** appears instantly — always the same for that name, password, and cache.

Within about a second, the **state catches up**.

If your stash holds **value**, it’s already waiting for you in the state.

*No accounts. No setup. No flows.*

You don’t create anything.

You just **open your stash**.

---

## ☕ A Real Example

A coffee shop creates a **cache** with an initial supply of **salt**.

Cashiers start the day with float. Customers exchange dollars for **salt**. Later, they spend **salt** for goods.

A returning customer selects the same cache, enters their name and password, and their stash resolves again.

If it holds **value**, it appears. If not, it remains empty until funded.

There is no backend tracking balances.

There is only **shared state**.

---

## 🗑️ What This Replaces

Punch cards get *lost*. Gift cards are *forgotten*. Loyalty apps depend on **databases** that must be trusted and maintained.

At their core, they’re all just ways of *coordinating value* within a group.

When groups share the same **surface**, coordination is simple.

---

## ♻️ How This Exists

Built from primitives already demonstrated in [`Byzantium`](../Byzantium/README.md).

**Cells** become *accounts*. **Salt** becomes *value*. **Whispers** become *transactions*. **Dream** becomes *state*. **Crypt** becomes *transport*.

Nothing new is happening here.

At its core, it is the *simplest transaction possible*, now routed over the internet.

Coordination happens without global consensus or energy overhead.

---

## 🧩 The Pieces of Mowsie

*If you want to understand how it works under the hood, this is the model.*

> Value is called **salt**.  
>  
> **Salt** lives inside **stashes**.  
>  
> **Stashes** exist within **caches**.  
>  
> **Salt** moves and carries **messages**.  
>  
> The system is made visible by **lanterns**.

---

## 🧂 Salt

> **Salt** is the unit of value.
> 
> It is *finite*, *transferable*, and **cryptographically secured**.
> 
> It is a *blank canvas*. People decide what it means.
> 
> Underneath, it is always **salt**.

## 🗝️ Caches & Stashes

> A **cache** is the shared state, a small domain where value lives and moves.
> 
> Think of it like a *tiny map*.
> 
> A **stash** is a specific spot on that map.
> 
> It only becomes *visible* when it holds value.
> 
> Your **name** and **password** act like a *secret location*.
> 
> Enter them again, and the same spot opens every time.

## 💬 Messages

> Transactions can carry a short message, up to **60 characters**.
> 
> Sometimes it’s a receipt. Sometimes it’s just a note.
> 
> Everything inside a cache is **encrypted**.
> 
> Meaningful from within, meaningless from the outside.

## 🏮 Lanterns

> Lantern nodes provide **visibility**.
> 
> They relay packets. They do not decide truth.
> 
> They do not store history.
> 
> They just let the system be seen.

---

## 🧠 Why This Is Different

Each cache defines a specific *shape* for what can exist within it.

Only matching stashes exist.

This is not a blockchain.

It is closer to a **lockchain**.

Not one global system.

But many small systems.

---

## 🛠️ What We’re Building

We are defining the structure, building the client, and implementing **Lantern nodes**.

The goal is simple.

Creating a value system should be as easy as sending an email.

---

## 🥔 Hardware Targets 🍞

Designed to run on *everyday devices*.

Wallets run on **toaster-class phones**.

Lantern nodes run on **potato-class hardware**.

No specialized infrastructure required.

**Built to run anywhere.**

---

## 📡 Contact

> If you see it and want to talk, reach out — ObliviousCompute@yahoo.com

---

<h2><img src="../Relics/BTC.png" width="25"/> Support</h2>

> **bc1qc69hm4smfvp4q2xwrn95926ljztxahe0q7fa8x**

---

**Mowsie replaces loyalty systems with shared state.**  
***It is the wooden nickel of the 21st century.***

---

## 📜 License

Released under [`LICENSE`](../LICENSE).

Use it, study it, modify it—just respect the terms.
