# 💀 Skeleton 💀

**A minimal, fully legible expression of the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) primitive.**

---

## 🦴 The Bones 🦴

An observer at present state $s$ within state space $\Omega$ determines whether a presented state $x$ belongs from that position.

$\Large 𝓐:\Omega\times\Omega\rightarrow\lbrace 0,1\rbrace \qquad 𝓐(s,x)\in\lbrace 0,1\rbrace$

For the cyclic geometry used by **Skeleton**:

$\Large 𝓐(s,x)=1 \iff x\in\lbrace s,\mathrm{next}\rbrace$

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

  # ============ LINCHPIN ============ #
    return projection in {state, next}
  # ================================== #

def apply(state, projection):
    if not admit(state, projection):
        return state, ["SYNC"]

    if projection == state:
        return state, []

    state = projection
    return state, ["PROJECT"]
```

### ☠️  ☠️  ☠️

---

## ⚰️ What's in the Box ⚰️

The code above instantiates the primitive within a minimal cyclic geometry. `state` is the observer's present position $s$, while `projection` is the presented position $x$. The values `ROCK`, `PAPER`, and `SCISSORS` occupy the state space $\Omega$, while `NEXT` defines the successor of each position.

From the observer's present `state`, `next` is selected from that geometry. Together they form the ordinary admissible window:

$\Large 𝓐(s,x)=1 \iff x\in\lbrace s,\mathrm{next}\rbrace$

The linchpin is therefore:

`projection in {state, next}`

A projection equal to `state` is self-equivalent and idempotent. A projection equal to `next` is admitted, becomes the observer's present state, and may be projected again. A projection outside the window does not advance the observer and may instead initiate synchronization.

***There is no field object stored in the program.*** Each observer holds state. Distributed across observers, symmetry between those independently maintained states produces the field.

> ***Reprojection is semantically dense.** By the time a normally projected state appears at an observer, it has already passed another observer’s admissibility gate, become that observer’s present state, and has been reprojected again. **Its appearance is itself a deduction from the relations that produced it.***
>
> 🧠 ***Big Brain Brad says: “Wait, does this repo actually move where computation takes place?”***

---

**Go back to [**`Skeleton`**](../Skeleton/README.md) or Continue to [**`Hydra`**](../Hydra/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
