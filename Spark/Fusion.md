# ✨ Fusion ✨

> ***Lamport, Shostak & Pease — The Byzantine Generals Problem, 1982***
>
> *IC1. All loyal lieutenants obey the same order.*
>
> *IC2. If the commanding general is loyal, then every loyal lieutenant obeys the order he sends.*

Faros is the Byzantine Generals problem under the Kernel primitive. Instead of sending private messages from general to general, a participant projects into one oblivious medium. Every loyal observer sees the same projection, applies the same admissibility rule, and independently maintains the resulting state.

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

Faros intentionally leaves the medium abstract. It does not implement sockets, encryption, discovery, or Genesis formation. `Project()` is the spotlight: one projection, same observation. The point is to expose how much algorithm remains once that communication primitive is granted. **ICBM** comes next and builds the runnable distributed machine.

***59 lines total. About 67% smaller by physical line count than the smallest conventional Lamport OM implementation we found. What disappeared? The messengers.***
