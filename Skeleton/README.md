# 💀 Skeleton 💀

**A minimal, fully legible expression of the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) primitive.**

---

## 🦴 The Bones 🦴

An observer at present state $s$ within state space $\Omega$ determines whether a presented state $x$ belongs from that position.

$\Large A:\Omega\times\Omega\rightarrow\lbrace 0,1\rbrace \qquad A(s,x)\in\lbrace 0,1\rbrace$

For the cyclic geometry used by **Skeleton**:

$\Large A(s,x)=1 \iff x\in\lbrace s,\mathrm{next}\rbrace$

The present position is idempotent, **next** is progressive, and every other position lies outside the ordinary admissible relation.

---

### ☠️  ☠️  ☠️


```python
ROCK, PAPER, SCISSORS = 1, 2, 3

NEXT = {
    ROCK: PAPER,
    PAPER: SCISSORS,
    SCISSORS: ROCK,
}


def admit(state, projection):
    next = NEXT[state]

    # ========== LINCHPIN ==========
    if projection not in {state, next}:
        return state, ["SYNC"]
    # ==============================

    # Self-equivalent
    if projection == state:
        return state, []

    # Admitted
    state = projection

    return state, ["PROJECT"]
```

### ☠️  ☠️  ☠️

---

## ⚰️ What's in the Box ⚰️

The code above instantiates the primitive within a minimal working mechanism. `state["sequence"]` is the observer's present position $s$, while `packet["sequence"]` is the presented position $x$. The values `ROCK`, `PAPER`, and `SCISSORS` occupy the state space $\Omega$, while `NEXT` encodes the relational geometry between those positions.

The function `inWindow(incoming, current)` is the executable analogue of the admissibility relation $A(s,x)$. It determines whether the presented position is the observer's present position or its admissible successor.

When an admissible presented state differs from the observer's present state, the observer advances to that position. A state outside the ordinary admissible relation does not produce forward progression and may instead initiate restorative behavior.

***There is no field object stored in the program.*** Each observer holds state. Distributed across observers, symmetry between those independently maintained states produces the field.

---

**Continue to [**`Hydra`**](../Hydra/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
