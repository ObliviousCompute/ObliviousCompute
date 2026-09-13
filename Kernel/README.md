# 💎 Kernel 💎

| Symbol | Meaning |
|--------|---------|
| 𝓐 | Admissibility function |
| Ω | State space |
| Σ | Relational symmetry |
| 𝓕 | Computational field |
| Δ | Diagonal |

> ***Interpretive note:*** *Relational symmetry **Σ** intentionally names the field relation without fully resolving its internal formalism at this stage. Perfect symmetry **Δ** is a realizable settled condition of the field, not its definition. This formulation presents one cross-section of the field at one fixed resolution.*

---

$\Large \cdots\Omega\rightarrow\Omega\times\Omega\rightarrow\Omega\leftarrow\Omega\times\Omega\leftarrow\Omega\cdots$

Oblivious Compute distributes a single admissibility function 𝓐 across a set of independently state-maintaining observers within a state space $\Omega$.

***State is projected into a shared medium without selecting, or requiring knowledge of, a computationally designated recipient. Any observer that encounters a projection evaluates it from its own position.*** 

Together, these local determinations form a matrix of relations across the observer set.

$\Large 𝓐:\Omega\times\Omega\rightarrow\lbrace 0,1\rbrace \qquad 𝓐(s,x)\in\lbrace 0,1\rbrace$

Therefore, the same presented state may be admissible from one observer position and inadmissible from another.

$\Large s_i\neq s_j \qquad \ 𝓐(s_i,x)=1\qquad 𝓐(s_j,x)=0$

Across **$n$** observers, independently maintained states form a configuration in the Cartesian product $\Omega^n$. Let Σ denote the relational symmetry among those states induced by 𝓐. The computational field 𝓕 is that relational structure, not any individual observer state.

$\Large (s_1,s_2,\ldots,s_n)\in\Omega^n \qquad 𝓕\equiv\Sigma(s_1,s_2,\ldots,s_n)$

**No observer contains the field.** It contains no state of its own and exists only through symmetry among independently maintained states. In this perfectly symmetric resolution, those states coincide and the aggregate configuration lies on the diagonal of the product space.

$\Large s_1=s_2=\cdots=s_n=s \qquad (s_1,s_2,\ldots,s_n)\in\Delta_n(\Omega)\subseteq\Omega^n$

At perfect symmetry, the $n$ observer coordinates no longer vary independently. The diagonal is canonically isomorphic to the original state space.

$\Large \Delta_n(\Omega)\cong\Omega$

**At rest on the diagonal, the distributed configuration reduces to the state space.**

$\Large \Omega$

## 💎 Diamond Tip 💎

At a fixed observer configuration, ***the field has a whole-space shape.*** Let $s$ denote the independently maintained observer states considered together within the Cartesian product $\Omega^n$. From that configuration, $\Sigma_s$ represents relational admissibility across the whole space, distinguishing admissible from inadmissible configurations with a value of zero or one. This is a **whole-space view of the field at one fixed resolution**, not a definition of the field itself.

$\Large s=(s_1,\ldots,s_n)\in\Omega^n \quad \Sigma_s:\Omega^n\rightarrow\{0,1\}$

Now let $\Phi_s$ denote the physical and causal medium through which that configuration can actually exist. It includes whatever determines which projection can exist with which observer at which instant: latency, processor timing, scheduling, memory access, geographic separation, packet propagation, and even the speed of light where it matters. The configuration and the medium together constitute the realized machine, denoted $M_s$.

$\Large M_s=(s,\Phi_s)$

Relational symmetry can now be taken over the realized machine itself. The observer configuration supplies the positions, $\Phi_s$ supplies the physical conditions through which those positions can relate, and the resulting relational object is the field at that realized scale.

$\Large \Sigma_{M_s}$

> ***The space between the observers constitutes the machine.***

---

**Continue the [**`Spark`**](../Spark/README.md)**`⟶`**[**`Fusion`**](../Spark/Fusion.md) or start [**`Skeleton`**](../Skeleton/README.md)...**

> ***If you must...spoil the fun, go straight to*** [**`HALT`**](../Theory/Halt.md)***...***

---

## 📜 License

See the [**`NOTICE`**](../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.

