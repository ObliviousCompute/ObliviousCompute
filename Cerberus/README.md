# 🔥 Cerberus 🔥

**Watch nine heads fight over a pile of bones.**

---

<img src="../Relics/DogBoard.png" width="600"/>

> ***Bare-Bones Oblivious State***

---

## 🦴 BonePile 🦴

## 🦴 BonePile 🦴

Cerberus is an intentionally small experiment in **Byzantine-resistant distributed state**. Nine logical **Heads** fight over a single **99-bone BonePile**, while every observer maintains the complete state independently. There is **no replicated log** and no authority deciding which story happened first.

When a Head equivocates, Cerberus does not erase the conflict. Conflicting actions become *consequential state*, and **ClawBack** carries any unpaid debt through the BonePile until an admissible state remains. ***The trick is not choosing which lie was really first. It is making the consequences part of the computation.***

**Make the dogs fight. Try to split the pile.**

---

## 🐧 Operating System Support

- ✅ Linux  
- ✅ macOS  
- ❌ Windows (sorry, but not sorry)

---

## 🍄 Install

To run Cerberus, install it with:

```bash
pipx install Cerberus-Game && Cerberus
```

You’ll need **Python 3.10 or newer** and an **80x24 UNIX-like terminal environment.**

> Don't have **pipx**? See how to install it [**`Here`**](../Relics/pipx.md).

---

## 🏚️ DogHouse 🏚️

***Cerberus includes two built-in adversarial demonstrations.***

> **😈 DevilDog** *puts five greedy dogs in a field with four loyal dogs. The greedy dogs issue conflicting signed spends while ordinary play continues. Their evidence eventually reaches Oblivion, Cerberus reconciles the field, and all nine Heads bury the same 99-bone BonePile.*

> **😇 LuckyDog** *keeps escalating the same idea. Dogs progressively promise more bones than they can cover until Cerberus razes them one by one and Lucky is left standing with all 99 bones.*

```bash
Cerberus DevilDog proofs
Cerberus LuckyDog proofs
```

Remove `proofs` to watch either demonstration run interactively.

---

## 🕸️ Networking

Cerberus runs locally over sockets as a **Sandbox Smoketest**.

> *All nodes must ***use the same Cerberus name and Head Count*** to join the same projection.*  
> ***Each node chooses its own DogTag and BonePile.***
> 
> *Tip: just spam Enter to drop straight into a board*.  

---

## 🧩 Continuity

Leave and return *microseconds or millennia later*. As long as one participant still holds the state, the projection persists.

**You do not reconnect to the past. You reconnect to what is.**

---

## 🏛️ Architecture

<img src="../Relics/DogTree.png" width="600"/>

> *How Bones shape the BonePile*

The **Oblivious Medium** lets Cerberus produce unusually rich distributed behavior from a very small machine. Three layers are enough to carry identity, state, equivocation, recovery, projection, and re-entry.

*The whole thing runs in under 1,700 lines of Python.* **Small enough to understand as one object.**

---

## 🗝️ Security Notice

> Oblivious Compute does not depend on any particular encryption scheme. Some reference implementations use simple XOR obfuscation for projection separation, which is not secure encryption and is not intended to be. Add whatever transport security you want; it does not change the primitive.

---

**Go Back to [**`Hydra`**](../Hydra/README.md) or Continue to [**`Byzantium`**](../Byzantium/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.


