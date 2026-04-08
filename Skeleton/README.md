# 💀 Skeleton

A minimal, fully legible expression of the Oblivious Compute primitive.

No architecture.  
No interface.  
No abstraction.

Just the rule that determines what is allowed to exist.

---
## IFF It Fits, It Ships

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

    # State → rehydrate only
    if isState(packet):
        if packet.get("tallies") and packet["tallies"] != state["tallies"]:
            state["tallies"] = dict(packet["tallies"])
        state["envy"] = False
        return state, []

    current = state["sequence"]
    incoming = packet["sequence"]

    # =========== LINCHPIN =========== #
    if not inWindow(incoming, current):
    # ================================ #
        intents = ([] if state["envy"] else ["ENVY"])
        state["envy"] = True
        return state, intents + ["Sync"]

    state["envy"] = False

    if packet["tallies"] == state["tallies"]:
        return state, []

    state["tallies"] = dict(packet["tallies"])
    state["sequence"] = incoming
    state["head"] = packet.get("id")

    intents = ["PROPAGATE"]

    if incoming == NEXT[current]:
        intents.append("Sync")

    return state, intents
```

---

## What This Is

Skeleton is the simplest possible form of the system.

It does not simulate behavior.  
It does not coordinate across nodes.  

It only answers one question:

> Is this state admissible?

If yes, it becomes the current state.  
If not, it is erased.

---

> Only what is admissible remains.
