---

## 🐧 Operating System Support

- ✅ Linux  
- ✅ macOS  
- ❌ Windows (sorry, but not sorry)

---

## 🍄 Install

To run ICBM, install it with:

```bash
pipx install ICBM-Game && ICBM
```

You’ll need **Python 3.10 or newer** and an **80x24 UNIX-like terminal environment.**

> Don't have **pipx**? See how to install it [**`Here`**](../Relics/pipx.md).

---

## 🕸️ Networking

ICBM runs in two modes

**Simulation** is local a **Sandbox Smoketest** over sockets.

**Live** is meant for a LAN environment with **Multiple Machines**.

> *All nodes must ***use the same Doomsday and Authorization codes*** to join the same genesis.*  

---

## 🗝️ Security Notice

Cerberus uses **Ed25519 signing** to validate actions.

However, networking currently relies on simple **XOR-based obfuscation**. This is not secure encryption—and it’s not meant to be.

The system prioritizes **state integrity over transport security**.

---

**Go Back to [**`Kernel`**](../Theory/Kernel.md) or Continue to [**`Skeleton`**](../Skeleton/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
