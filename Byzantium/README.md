# Byzantium
---

A terminal-based system for forward-only state convergence.

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

---

## What This Is Not

- Not a finished protocol  
- Not production-ready  
- Not hardened networking  
- Not secure transport (yet)  

Think of this as:

> a working system under tension — not a polished product.

---

## Running

### Requirements

- Python **3.9+**
- A UNIX-like terminal

### Supported

- macOS ✅  
- Linux ✅  

### Not Supported

- Windows ❌  

Sorry — but also not sorry.

This system relies on terminal behavior and ANSI handling that Windows still manages to make painful.  
If you really want it, you can fight your way through WSL.

---

### Install dependencies

```bash
pip install cryptography wcwidth
```

---

### Run

From the project directory:

```bash
python3 Gateway.py
```

Follow the in-terminal prompts to:

- choose mode (Siege or Campaign)  
- set your gateway (port)  
- define your identity  
- set genesis size  

Then the system initializes and the board emerges.

---

## How It Works (Brief)

- Players (“souls”) join a shared state  
- A deterministic genesis builds the board  
- Actions are expressed as **glyphs**  
- Glyphs are:
  - validated  
  - applied  
  - propagated  

If a mutation is valid, it becomes reality.  
If not, it disappears.

No forks.  
No reconciliation.  
No second chances.

---

## Core Modules

- `Gateway.py` — entrypoint + title/menu  
- `Vault.py` — key generation + signing  
- `Dream.py` — state mutation + propagation  
- `Crypt.py` — networking layer  
- `Sanctum.py` — deterministic genesis  
- `Field.py` — validation + invariants  
- `Forge.py` — shared structures  
- `Citadel.py` — UI intent + control  
- `Spire.py` — terminal renderer  

---

## Networking

Two modes:

- **Siege** — local multi-terminal (same machine)  
- **Campaign** — LAN multiplayer  

Ports and configuration are handled inside the menu.

---

## ⚠️ Security Notice

Byzantium uses **Ed25519 signing** for validating actions.

However:

- The networking layer currently uses **XOR-based obfuscation**
- This is **not secure encryption**
- Messages can be intercepted or inspected

This is intentional for now.

The system prioritizes **state integrity over transport security**.

If you care about that layer, replace it with:

- a proper KDF  
- authenticated encryption (AES-GCM / ChaCha20-Poly1305)  
- real peer validation  

---

## Design

- No history  
- No rollback  
- No central authority  

Only:

> the current shape of state.

---
```bash
# Install (requires pipx)
pipx install "git+https://github.com/obliviousCompute/ObliviousCompute.git#subdirectory=Byzantium"

# Run
Byzantium
```

## License

See the `LICENSE` file for details.
