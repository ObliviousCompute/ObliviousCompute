# 💀Skeleton

A minimal, fully legible expression of the Oblivious Compute invariant.

No architecture.  
No interface.  
No abstraction.

Just the rule that determines what is allowed to exist.

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

## 🧬 Admissibility Braid

State does not progress through time.

It progresses through admissibility.

Each position permits only two possibilities:
- itself  
- or a single valid next position  

Anything else is not processed and produces no effect.

The system does not remember the past.  
It does not reconstruct history.  

It only accepts what is admissible now.

---

## How to Read It

Start here:

```python
if not inWindow(incoming, current):
```

That condition defines the boundary of admissibility.

Everything else follows from it.

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

## Relation to the Repo

- **Byzantium** → the interactive surface  
- **Hydra** → the distributed behavior  
- **Skeleton** → the primitive itself  

---

> Truth through erasure.
