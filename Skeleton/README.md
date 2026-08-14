# 💀 Skeleton 💀

**A minimal, fully legible expression of the Oblivious Compute primitive.**

---

## 🦴 The Bones 🦴

An observer at present state $s$ within state space $\Omega$ determines whether a presented state $x$ belongs from that position.

$\Large s\in\Omega \qquad x\in\Omega \qquad A(s,x)\in\lbrace 0,1\rbrace$

For the cyclic geometry used by **Skeleton**:

$\Large A(s,x)=1 \iff x\in\lbrace s,\mathrm{NEXT}[s]\rbrace$

The present position is idempotent, the successor is progressive, and every other position lies outside the ordinary admissible relation.

---

### ☠️  ☠️  ☠️


```python
ROCK, PAPER, SCISSORS = 1, 2, 3

NEXT = {
    ROCK: PAPER,
    PAPER: SCISSORS,
    SCISSORS: ROCK,
}

def isState(packet):
    return (
        packet.get("isState")
        or packet.get("isSnap")
        or packet.get("isIntent")
    )

def inWindow(incoming, current):
    return incoming in (current, NEXT[current])

def ingest(state, packet):

    if isState(packet):
        if packet.get("tallies") and packet["tallies"] != state["tallies"]:
            state["tallies"] = dict(packet["tallies"])
        state["desync"] = False
        return state, []

    current = state["sequence"]
    incoming = packet["sequence"]

    # =========== LINCHPIN =========== #
    if not inWindow(incoming, current):
    # ================================ #
        intents = ([] if state["desync"] else ["REJECT"])
        state["desync"] = True
        return state, intents + ["ACCEPT"]

    state["desync"] = False

    if packet["tallies"] == state["tallies"]:
        return state, []

    state["tallies"] = dict(packet["tallies"])
    state["sequence"] = incoming
    state["head"] = packet.get("id")

    intents = ["PROPAGATE"]

    if incoming == NEXT[current]:
        intents.append("ACCEPT")

    return state, intents
```

### ☠️  ☠️  ☠️

---

## ⚰️ What's in the Box ⚰️

The code above instantiates the primitive directly. `state["sequence"]` is the observer's present position $s$, while `packet["sequence"]` is the presented position $x$. The values `ROCK`, `PAPER`, and `SCISSORS` occupy the state space $\Omega$, while `NEXT` encodes the relational geometry between those positions.

The function `inWindow(incoming, current)` is the executable analogue of the admissibility relation $A(s,x)$. It determines whether the presented position is the observer's present position or its admissible successor.

When an admissible presented state differs from the observer's present state, the observer advances to that position. A state outside the ordinary admissible relation does not produce forward progression and may instead initiate restorative behavior.

***There is no field object stored in the program.*** Each observer holds state. Distributed across observers, symmetry between those independently maintained states produces the field.

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
