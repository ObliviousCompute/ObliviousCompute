# ✨ Fusion ✨

***Under enough pressure, possibility collapses to state.***

## Interactive Consistency

**Lamport, Shostak & Pease — The Byzantine Generals Problem, 1982**

**IC1** All loyal lieutenants obey the same order

**IC2** If the commanding general is loyal, then every loyal lieutenant obeys the order he sends

***Oral messages establish who spoke.***

***Signed messages establish what was signed.***

> ***Lamport's oral construction requires fewer than one-third traitors, and the signed construction removes that bound under its authentication assumptions.***

---

## The Reaction

Fusion places Interactive Consistency under the ***Kernel primitive***. Instead of sending private messages from one participant to another, a participant **projects state into an oblivious medium**. Every loyal observer encounters the same projection from its own independently maintained state and applies the ***same admissibility rule***.

The commander begins with an order committed into the **Genesis state**. A candidate reveal does not become computational state merely because it was projected. Each observer independently determines whether it is ***admissible from the state already held***. If it matches the commitment, **it survives**. If it does not, it contributes nothing.

The medium does not **decide, coordinate, vote, or choose a recipient**. It carries the projection. The observers perform the computation, and agreement appears through the relation among their independently maintained states. ***What survives the same rule becomes the same state.***

---

## Critical Mass

***This is executable Python, not pseudocode.*** Save the implementation below as `Fusion.py` and run it directly:

`python3 Fusion.py`

It executes the loyal and Byzantine cases shown in the construction and checks **IC1** and **IC2** as assertions. Fusion deliberately keeps the medium abstract so the admissibility mechanism remains exposed; sockets, discovery, encryption, and Genesis formation arrive in **ICBM**.

> **Evaluating the construction? Run it before continuing.** The point of Fusion is not to simulate a network. It is to isolate the smallest executable form of the argument so you can inspect exactly what is doing the computational work.

```python
from dataclasses import dataclass
from hashlib import sha256
Orders = ("ATTACK", "RETREAT")
Lieutenants = ("B", "C", "D", "E")

def Commitment(order, key):
    return sha256(f"{order}\0{key}".encode()).hexdigest()

@dataclass(frozen=True)
class Set:
    Commander: str
    Commitment: str
@dataclass(frozen=True)
class Key:
    Sender: str
    Order: str
    Secret: str

class Observer:
    def __init__(self, name, genesis):
        self.Name = name
        self.Set = genesis
        self.State = set()
    def Observe(self, payload):
        match = payload.Sender == self.Set.Commander and payload.Order in Orders
        # ================ LINCHPIN ================ #
        admissible = match and Commitment(payload.Order, payload.Secret) == self.Set.Commitment
        # ========================================== #
        if admissible:
            self.State.add(payload.Order)
    def Decide(self):
        return min(self.State) if self.State else "RETREAT"

# Oblivious medium: one projection, same observation.
def Project(observers, payload):
    for observer in observers:
        observer.Observe(payload)

def Trial(order, loyal=True, reveal=True):
    secret = "MIDNIGHT"
    genesis = Set("A", Commitment(order, secret))
    observers = [Observer(name, genesis) for name in Lieutenants]
    if not loyal:
        Project(observers, Key("A", "ATTACK", "WRONG"))
        Project(observers, Key("A", "RETREAT", "WRONG"))
        Project(observers, Key("B", order, secret))
    if reveal:
        Project(observers, Key("A", order, secret))
    decisions = [observer.Decide() for observer in observers]
    assert len(set(decisions)) == 1                         # IC1
    if loyal:
        assert all(decision == order for decision in decisions)  # IC2
    return decisions

if __name__ == "__main__":
    print("Loyal ATTACK:    ", Trial("ATTACK"))
    print("Loyal RETREAT:   ", Trial("RETREAT"))
    print("Byzantine reveal:", Trial("ATTACK", loyal=False))
    print("Byzantine silent:", Trial("ATTACK", loyal=False, reveal=False))
```

> **59 lines total. Our ***smallest conventional Lamport OM reference is 210% larger*** by physical line count.**

---

**Continue the [**`Spark`**](../Spark/README.md)**`⟶`**[**`ICBM`**](./ICBM/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
