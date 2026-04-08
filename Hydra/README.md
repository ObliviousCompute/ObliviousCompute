# 🐍 Hydra

A distributed expression of the Oblivious Compute system.

Multiple instances.  
Shared state.  
No coordination layer.

Each node emits.  
Each node observes.  
State converges.

---

<img src="../Relics/ChompChomp.gif"/>


---

## Download

Download all three Python files into a single directory.

Navagate to that directory in a terminal.

Each Hydra node runs in its **own terminal window**.

---

## Running the Demo

You will run **up to 5 processes**

- Ports must be **unique per machine**
- Port numbers **may be the same across different machines**

---

### Single Machine (Localhost)

#### Terminal 1 — Head A
```bash
python3 Hydra.py --id A --port 5001 --peers \
127.0.0.1:5002 127.0.0.1:5003 127.0.0.1:5004 127.0.0.1:5005
```

#### Terminal 2 — Head B
```bash
python3 Hydra.py --id B --port 5002 --peers \
127.0.0.1:5001 127.0.0.1:5003 127.0.0.1:5004 127.0.0.1:5005
```

#### Terminal 3 — Head C
```bash
python3 Hydra.py --id C --port 5003 --peers \
127.0.0.1:5001 127.0.0.1:5002 127.0.0.1:5004 127.0.0.1:5005
```

#### Terminal 4 — Head D
```bash
python3 Hydra.py --id D --port 5004 --peers \
127.0.0.1:5001 127.0.0.1:5002 127.0.0.1:5003 127.0.0.1:5005
```

#### Terminal 5 — Head E
```bash
python3 Hydra.py --id E --port 5005 --peers \
127.0.0.1:5001 127.0.0.1:5002 127.0.0.1:5003 127.0.0.1:5004
```

---

### Multiple Machines (LAN)

You may also run nodes across **multiple machines** by replacing `127.0.0.1`
with LAN IP addresses.

Each node binds to a local UDP port.  
Port numbers may be the **same across different machines**, but must be **unique per machine**.

Example:
- `192.168.1.101:5001`
- `192.168.1.102:5001`

These are distinct sockets and work correctly.

---

### Operating System

- ✅ Linux  
- ✅ macOS  
- ❌ Windows (not supported)

---

## License & Intent

Hydra Proofs are published as a **public technical disclosure**.

This demo exists to show that **oblivious convergence is possible**.

If it fails, it fails cleanly.  
If it works, it demonstrates a new computational primitive.
