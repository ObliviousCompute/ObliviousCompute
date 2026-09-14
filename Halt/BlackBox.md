# The Black Box

## Inside the Oblivious Machine

A computer computes. State enters the machine, the machine executes, and something happens next. Conventionally, whatever happens inside that box is treated as the computation, while whatever emerges is treated as its result. The machine may be simple or arbitrarily complex, but the **computational question remains pointed inward.**

Oblivious Compute moves the distributed computation **outside of the box.** Give independent machines **the identical state** $x$ and allow each machine $i$ to produce a state $F_i(x)$. No individual result is authoritative; ***the computation appears in the relation*** $\Sigma$ among those independently produced states.

$\large x\rightarrow F_1(x),F_2(x),\ldots,F_n(x)\qquad \Sigma(F_1(x),F_2(x),\ldots,F_n(x))$

When independently maintained states resolve to the same position, their relational configuration lies on the diagonal. Many machines remain physically independent while the distributed state resolves to **one computational position.**

$\large (s,\ldots,s)\in\Delta_n(\Omega)\qquad \Delta_n(\Omega)\cong\Omega$

## House of Mirrors

Now the fun begins. Give several independent machines **the identical state: Rock.** Healthy machines move through the state space in one direction. Place an inverter among them and it moves through the ***same state space in the opposite direction.*** Nothing inside the inverter needs to be inspected. Its inversion becomes visible in what it projects.

$\large F:\text{Rock}\rightarrow\text{Paper}\rightarrow\text{Scissors}\rightarrow\text{Rock}
\qquad
F_I:\text{Rock}\rightarrow\text{Scissors}\rightarrow\text{Paper}\rightarrow\text{Rock}$

Now place **HALT** and **REPEAT** on top of that cycle. Give every machine **HALT** and the healthy machines stop producing fresh continuation while the inverter continues. Invert the experiment and give every machine **REPEAT**. The healthy machines continue through the cycle while the inverter halts, yet admissible projections from the field may still move it.

***HALT and REPEAT belong to the machine, not the field. The field determines what continues to belong.***

> ***The inverter can reflect forever. The field reflects the reflection.***

## Raise the Anti

A black box inside an oblivious machine creates **an inversion of an inversion.** The first machine may hide everything **within its bounds.** The second gives those bounds **no authority.** Whatever the box produces is projected back into a relation with independently maintained states. ***Its interior can remain oblivious. Its output cannot.***

**Expanding the boundary around the system does not restore that authority. A boundary may only expand when the computation does.** The boundary may contain every machine, every projection, and every changing state, but it is **only a description of what has been enclosed.** The computation remains in the relation among independently maintained states. A black box can reflect itself forever. **The field reflects the reflection.**

The machines execute. Their outputs project. **The relation determines what continues to belong.**

**The machine reveals its hand. Its state is the ante. The field calls.**

***Now the game begins.***

> ***If the machine is a true inverter, add Rock → Paper → Scissors → Rock on top of HALT/REPEAT. An inverting machine produces the wrong next state or bit, so its projection is not admissible; it falls into oblivion while the healthy cycle continues.***

---

**Go back to [**`Halt`**](./README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.

