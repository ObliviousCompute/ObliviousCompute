# ✨ Fusion ✨

***Pressure collapses possibility.***

---

## The Reaction

Fusion places Interactive Consistency under the ***Kernel primitive***. Instead of sending private messages from one participant to another, a participant **projects state into an oblivious medium**. Every loyal observer encounters the same projection from its own independently maintained state.

The commander begins with an order committed into the **Genesis state**. A candidate reveal does not become computational state merely because it was projected. Each observer independently determines whether it is ***admissible from the state already held***. If it matches the commitment, **it survives**. If it does not, it contributes nothing.

The medium does not **decide, coordinate, vote, or choose a recipient**. It carries the projection. The computation exists in the **relation among their independently maintained states**, not inside any observer. ***What survives the same rule becomes the same state.***

---

## Critical Mass

***This is executable Python, not pseudocode.*** Save the implementation below as `Fusion.py` and run it directly:

`python3 Fusion.py`

It executes the loyal and Byzantine cases shown in the construction and checks **IC1** and **IC2** as assertions. Fusion deliberately keeps the medium abstract so the admissibility mechanism remains exposed; sockets, discovery, encryption, and Genesis formation arrive in **ICBM**.

> **Fusion presents an empirical anomaly:** this construction is **59 lines**, while the smallest Lamport reference implementation we found is ***more than 3× larger*** by physical line count. A familiar coordination problem collapses once the **Oblivious Medium** and ***admissibility*** are treated as machine primitives. ***Either the missing complexity has merely been hidden elsewhere, or the computational primitive has changed.*** The rest of the repository is an attempt to distinguish those explanations.

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

> ***This specimen fixes one particular*** $\Phi_s$. ***Every loyal observer is exposed to the same projection. Lamport's oral-message model does not make that assumption. Fusion is not a reproduction of that machine. It is a specimen of admission under the Oblivious Compute machine.***

---

## Interactive Consistency

**Lamport, Shostak & Pease — The Byzantine Generals Problem, 1982**

**IC1** All loyal lieutenants obey the same order

**IC2** If the commanding general is loyal, then every loyal lieutenant obeys the order he sends

---

🧭 **Continue to [**`ICBM`**](./ICBM/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
