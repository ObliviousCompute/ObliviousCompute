# Hydra Demo — Truth Through Erasure

This repository contains a **runnable demonstration of Hydra Proofs**, a concrete instantiation of the **Oblivious Compute (OC)** primitive.

Hydra demonstrates **distributed correctness through deterministic selection and erasure**, rather than agreement, coordination, or historical replay.

This is a **research demo**, not a product.

## Quick Visual Demo

A short screen capture showing five Hydra nodes converging via erasure.

[▶ Watch the demo video](./Demo.mp4)

---

## What This Demo Demonstrates

This demo runs multiple Hydra nodes as independent processes and shows how computers can drop out, reboot, and rejoin a network while staying in sync without coordination overhead. Nodes exchange full candidate states over UDP, allow temporary divergence, and deterministically converge by overwrite. There is no leader, no ordering requirement, no replay log, and no recovery of losing states. When a process is killed, it disappears. When it restarts, it instantly hydrates the current truth. Nothing panics. Nothing waits. The system simply keeps the present state and erases everything else.

---

## Requirements

### Operating System

- ✅ Linux  
- ✅ macOS  
- ❌ Windows (not supported)

This demo relies on low-level terminal control:

- `termios`
- `tty`
- raw stdin handling

These APIs are not available on Windows.  
WSL is **not guaranteed** to work correctly.

---

### Python Version

- **Python 3.10 or higher required**

Reason:
- Uses modern type syntax (`|`, `Dict[str, Any]`)
- No backward-compatibility shims are provided

Check your version:

```bash
python3 --version
```

---

## Download

Download all three Python files into a single directory.

Navagate to that directory in a terminal.

Each Hydra node runs in its **own terminal window**.

---

## Running the Demo (5 Nodes)

You will run **five processes**, one per head:

- A
- B
- C
- D
- E

Each node binds to a UDP port.

- Ports must be **unique per machine**
- Port numbers **may be the same across different machines**

---

### Example: Single Machine (Localhost)

Open **five terminals**, then run:

#### Terminal 1 — Head A
```bash
python3 Hydra.py --id A --port 5001 --peers \
127.0.0.1:5002 127.0.0.1:5003 127.0.0.1:5004 127.0.0.1:5005
```

#### Terminal 2 — Head B
```bash
python3 hydra.py --id B --port 5002 --peers \
127.0.0.1:5001 127.0.0.1:5003 127.0.0.1:5004 127.0.0.1:5005
```

#### Terminal 3 — Head C
```bash
python3 hydra.py --id C --port 5003 --peers \
127.0.0.1:5001 127.0.0.1:5002 127.0.0.1:5004 127.0.0.1:5005
```

#### Terminal 4 — Head D
```bash
python3 hydra.py --id D --port 5004 --peers \
127.0.0.1:5001 127.0.0.1:5002 127.0.0.1:5003 127.0.0.1:5005
```

#### Terminal 5 — Head E
```bash
python3 hydra.py --id E --port 5005 --peers \
127.0.0.1:5001 127.0.0.1:5002 127.0.0.1:5003 127.0.0.1:5004
```

---

### Running Across Multiple Machines (LAN)

You may also run nodes across **multiple machines** by replacing `127.0.0.1`
with LAN IP addresses.

Each node binds to a local UDP port.  
Port numbers may be the **same across different machines**, but must be **unique per machine**.

Example:
- `192.168.1.101:5001`
- `192.168.1.102:5001`

These are distinct sockets and work correctly.

---

## Controls

- **Left / Right Arrow** — Select target head  
- **Up / Down Arrow** — Select amount  
- **Enter** — Submit transaction (“FEED”)  
- **Ctrl+X** — Soft reboot (Sever Head)  
- **Ctrl+C** — Exit  (Cauterize)

A node will **never allow you to target itself**.

---

## Notes on Stability

- This demo uses **UDP**
- Packet loss is expected under heavy load
- Wi-Fi saturation can induce temporary divergence
- The system recovers through overwrite and erasure

This behavior is **intentional** and part of the demonstration.

---

## Important Warnings

This demo intentionally provides **no guarantees** of:

- persistence  
- auditability  
- fairness  
- security hardening  
- Byzantine fault tolerance beyond erasure  

If a state survives, it is the truth.  
If it loses, it disappears completely.

---

## License & Intent

Hydra Proofs are published as a **public technical disclosure**.

This demo exists to show that **oblivious convergence is possible**.

If it fails, it fails cleanly.  
If it works, it demonstrates a new computational primitive.
