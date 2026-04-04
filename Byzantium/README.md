# Byzantium

A distributed terminal system where the interface *is* the program.

---

<p align="center">
  <img src="../Relics/bring-the-chips.gif" width="600"/>
</p>

---

## What This Is

- A **distributed terminal system**
- A **deterministic state machine**
- A **rules + interaction sandbox**
- A **live networked simulation**

Byzantium favors:

- legibility  
- determinism  
- minimal input  
- visible pressure  

There is no history log.  
No rollback.  
No undo.

State is continuously **refined in place**.

There is no “behind the scenes.”  
The system is exactly what you see.

---

<p align="center">
  <img src="../Relics/double-it.gif" width="600"/>
</p>

---

## Install

```bash
pipx install byzantium-game
Byzantium
```

Requires:

- Python **3.9+**
- A UNIX-like terminal

Supported:

- macOS ✅  
- Linux ✅  

Not Supported:

- Windows ❌  

---

## What You're Seeing

- Messages are part of state — not logs  
- Value ("salt") moves through interaction  
- Incentives shape behavior  
- Every action mutates the shared surface  

If a mutation is valid, it becomes reality.  
If not, it disappears.

No forks.  
No reconciliation.  
No second chances.

---

## What This Is Not

- Not a finished protocol  
- Not production-ready  
- Not hardened networking  
- Not secure transport (yet)  

> a working system under tension — not a polished product

---

## How It Works (Brief)

- Players (“souls”) join a shared state  
- A deterministic genesis builds the board  
- Actions are expressed as **glyphs**  
- Glyphs are:
  - validated  
  - applied  
  - propagated  

---

## Core Modules

<table>
  <tr>
    <td valign="top">

- `Gateway.py` — entrypoint + title/menu  
- `Vault.py` — key generation + signing  
- `Dream.py` — state mutation + propagation  
- `Crypt.py` — networking layer  
- `Sanctum.py` — deterministic genesis  
- `Field.py` — validation + invariants  
- `Forge.py` — shared structures  
- `Citadel.py` — UI intent + control  
- `Spire.py` — terminal renderer  

    </td>
    <td valign="top">
    <pre> 
      
     ┌[GateWay]┐
     ↑    ↓    ↓
     ├─[Spire]─┤
     ↑    ↑    ↓
     ├[Citadel]┤
     ↑    ↑    ↓
  {Forge} | (Vault)
          ↑    ↓
   ┌─ ←(Dream)←┘←┐
   ↓      ↑      ↑
   | ┌─ → ┴ ← ─┐ |
   ↓ ↑ {Field} ↑ ↑
   | └(Sanctum)┘ |
   ↓      ↑      ↑
   └─ →(Crypt)→ ─┘
         ↓ ↑
       *.*.*.*
  
</pre>

    </td>
  </tr>
</table>

---

## Networking

Two modes:

- **Siege** — local multi-terminal (same machine)  
- **Campaign** — LAN multiplayer  

---

## ⚠️ Security Notice

Byzantium uses **Ed25519 signing** for validating actions.

However:

- networking uses **XOR-based obfuscation**
- this is **not secure encryption**

This is intentional.

The system prioritizes **state integrity over transport security**.

---

<p align="center">
  <img src="../Relics/collecting-souls.gif" width="600"/>
</p>

---

## Design

- No history  
- No rollback  
- No central authority  

Only:

> the current shape of state

---

## License

See the `LICENSE` file for details.
