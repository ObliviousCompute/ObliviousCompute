# The Black Box

## Inside the Oblivious Machine

A computer computes. State enters the machine, the machine executes, and something happens next. Conventionally, whatever happens inside that box is treated as the computation, while whatever emerges is treated as its result. The machine may be simple or arbitrarily complex, but the **computational question remains pointed inward.**

Oblivious Compute moves the distributed computation **outside of the box.** Give independent machines **the identical state** $x$ and allow each machine $i$ to produce a state $F_i(x)$. No individual result is authoritative; ***the computation appears in the relation*** $\Sigma$ among those independently produced states.

$\large x\rightarrow F_1(x),F_2(x),\ldots,F_n(x)\qquad \Sigma(F_1(x),F_2(x),\ldots,F_n(x))$

When independently maintained states resolve to the same position, their relational configuration lies on the diagonal. Many machines remain physically independent while the distributed state resolves to **one computational position.**

## House of Mirrors

Now the fun begins with a simple example. Give several independent machines **the identical state: Rock.** Healthy machines move through the state space in one direction. Place an inverter among them and it moves through the ***same state space in the opposite direction.*** Nothing inside the inverter needs to be inspected. Its inversion becomes visible in what it projects.

$\large \text{Rock}\rightarrow\text{Paper}\rightarrow\text{Scissors}\rightarrow\text{Rock}
\qquad
(\mathrm{I})\ \text{Rock}\rightarrow\text{Scissors}\rightarrow\text{Paper}\rightarrow\text{Rock}$

Place **HALT** and **REPEAT** on top of that cycle. Give every machine **HALT** and the healthy machines stop producing fresh continuation while the inverter continues. Invert the experiment and give every machine **REPEAT**. The healthy machines continue through the cycle while the inverter halts, yet admissible projections from the field may still move it.

***A projected state does not become authoritative merely because a machine produced it. Each observer retains its own state and admits only the continuation available from that position. The machine determines what it projects. The relation determines whether that projection becomes state.***

> ***The inverter can reflect forever. The field reflects the reflection.***

## Raise the Anti

A black box inside an oblivious machine creates **an inversion of an inversion.** The first machine may hide everything **within its bounds.** The second gives those bounds **no authority.** Whatever the box produces is projected back into relation with independently maintained states. ***Its interior can remain oblivious. Its output cannot.***

**Expanding the boundary does not restore that authority.** A projection may carry the geometry of the field, but ***it is not the field.*** Enclose a projection and the relation that gave it life remains outside. The closest any observer comes to the field is the admissible state it holds. **An inverted field may exist, but only as another live relation, not as a captured object turned inside out.**

**The machine reveals its hand. Its state is the ante. The field calls.**

---

**Go back to [**`Halt`**](./README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.

