# 💀 Skeleton 💀

**A minimal, fully legible expression of the Oblivious Compute primitive.**

---
## ⚰️ What's in the Box ⚰️

### From Formalism to Code

The mathematical model is not intended as a line-for-line specification of the implementation. It describes the structure that the code instantiates.

In the example below, `state["sequence"]` is an observer's present position $s$, while `packet["sequence"]` is the presented position $x$. The values `ROCK`, `PAPER`, and `SCISSORS` occupy the state space $\Omega$, and `NEXT` encodes the relational geometry between those positions.

The function `inWindow(incoming, current)` is the executable analogue of the admissibility relation $A(s,x)$. A presented position belongs when it is either the observer's present position or its admissible successor:

$A(s,x)=1 \iff x\in\{s,\operatorname{NEXT}(s)\}$

When an admissible presented state differs from the observer's present state, the observer may advance to that position. A state outside the admissible relation does not produce ordinary forward progression and may instead initiate restorative behavior.

***There is no field object stored in the program.*** Each observer holds state. Distributed across observers, symmetry between those independently maintained states produces the field.

> A state transition is valid only if it moves forward within a constrained window of the current state.  
> Incompatible state is rejected or treated as a boundary violation (desync).

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

---

## 🦴 Constraint 🦴

This primitive assumes no access to history.
 
No ordering to reconstruct.  
No prior state to consult.  
No logs to replay.  

All decisions are made against the present state only.

When combined with historical systems, this reduces to validation logic.

When history is removed, it becomes the system itself.

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
