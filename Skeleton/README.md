# 💀 Skeleton

A minimal, fully legible expression of the Oblivious Compute primitive.

No architecture.  
No interface.  
No abstraction.

Just the rule that determines what is allowed to exist.

---
## 📦 IFF It Fits, It Ships

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

## Constraint

This primitive assumes no access to history.

There are no logs to replay.  
No ordering to reconstruct.  
No prior state to consult.

All decisions are made against the present state only.

When combined with historical systems, this reduces to validation logic.

When history is removed, it becomes the system itself.

---

## License

This project is released under the terms of the [LICENSE](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
