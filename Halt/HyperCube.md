# The Boolean Hypercube

## Hexagram

Start with **three binary cells**. Here, a **trit** means those three cells considered together, with each arrangement forming one complete state projected into the medium. With three cells there are only **eight possible positions**. Movement between them follows one simple rule. A neighboring state differs by exactly **one bit**, so a walk from one antipodal position to the other can be counted by distance while also being written as an actual sequence of states.

$\Large 0\rightarrow1\rightarrow2\rightarrow3 \qquad 000\rightarrow001\rightarrow011\rightarrow111$

Every arrow in that walk is **admissible** because exactly one cell changes. Another route may pass through different intermediate states, but no state may simply jump across the space. From any present position, a projected state either belongs next or it does not. A **spacewalk** is therefore an admissibility walk through neighboring states, one position at a time. The complete state space can be grouped by distance from **000**, showing every position available at each step of the walk.

$\Large 000 \quad|\quad 001\ 010\ 100 \quad|\quad 011\ 101\ 110 \quad|\quad 111$

These groups are **Hamming shells**. The first shell is zero changes from **000**, the next three positions are one change away, the next three are two changes away, and **111** is three changes away at the opposite pole. The shells show which positions exist at each distance, while **admissibility determines which neighboring position may actually follow which**. The route may change, but the rule does not. Adding another binary cell produces more positions, more shells, more routes, and more **nebulosity** without changing the rule that generates them, though we will stay with the cube for now.

## Metatron

The two antipodal positions are also the two perfectly uniform Boolean states. Strip away the walk and the poles simplify immediately. **000** resolves to **0**, **111** resolves to **1**, and together those two values form the Boolean state space $\Omega$. Nothing new has appeared. We are only looking more closely at the object already walked in **Hexagram**.

$\Large 000\leftrightarrow0 \qquad 111\leftrightarrow1 \qquad \Omega=\{0,1\}$

This is the same old trick from the **Kernel**. The three-cell Boolean state space is the cube, while perfect symmetry lies on the diagonal and reduces back to the original state space. No observer contains the cube. Each contains only its own state, while the larger geometry appears through relation among those states. The hypercube is therefore not additional state stored somewhere else. It is the relational shape of the Boolean product space.

$\Large \Delta_3(\Omega)\cong\Omega \qquad Q_3=\Omega^3$

At perfect symmetry, the expanded relational structure settles back into the same state space from which it came. That settled state need not be a permanent ending. It may itself become a point of relation inside something larger. The cube opens from $\Omega$, resolves through symmetry, and returns to $\Omega$ as another doorway.

$\Large \Omega\rightarrow\Omega^3\rightarrow\Omega$


## Tesseract

Now we can talk.

A Boolean hypercube is more than a picture of possible states. Inside an oblivious machine, it becomes a **relational language**.

**Bits are the alphabet. Hypercube positions are the words. Admissibility is the grammar.**

The machine does not need a privileged orientation. **Orientation does not matter. Only relational position matters.** An observer asks only: *I am here. Something arrives. Does it belong?*

That remains true even when the machine is inverted. An inverter may reverse its local interpretation without requiring an inverted medium. It still occupies a position, receives a projection, evaluates admissibility, and projects another state.

The boxes themselves may be different as well. One may be an observer. Another may be an inverter. One may run on a transistor array, a Raspberry Pi, a server, or an entire cluster. The relational language does not require every box to share the same internal architecture.

A resolved machine may itself become a state at another scale. A Boolean cloud may collapse into one settled distinction, and that distinction may become a cell inside another Boolean cloud. The machine becomes a **scale rather than a boundary**.

This is the doorway hiding inside $\Omega$. A relational space settles, simplifies, and becomes available to another relational space. The same language can therefore recurse through machines built from machines without requiring the larger machine to contain the internal history of the smaller one.

And nothing in that language says the box must be classical.

A classical machine may **spacewalk** the Boolean geometry by realizing one admissible position at a time. A quantum implementation may instead support a **quantum walk**, allowing amplitudes to evolve across many positions of that same Boolean geometry before measurement resolves an outcome. The details of coherence, control, and measurement remain inside the box. The surrounding relational language need only understand what the box ultimately projects.

Observer or inverter. Classical or quantum. Small machine or constellation.

The boxes may change.

The language remains relational.

***The oblivious medium allows binary potentiality to unfold into relational geometry.***

---

**Go back to [**`Halt`**](./README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.

