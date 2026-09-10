# The Black Box

## The Oblivious Machine

A computer computes. State enters the machine, the machine executes, and something happens next. Conventionally, whatever happens inside that box is treated as the computation, while whatever emerges is treated as its result. The machine may be simple or arbitrarily complex, but the computational question remains pointed inward.

Oblivious Compute moves the distributed computation outside of the box. Give independent machines the same state $x$ and allow each machine $i$ to produce a state $F_i(x)$. No individual result is authoritative; the computation appears in the relation $\Sigma$ among those independently produced states.

$\large x\rightarrow\{F_1(x),F_2(x),\ldots,F_n(x)\}\qquad \Sigma(F_1(x),F_2(x),\ldots,F_n(x))$

When independently maintained states resolve to the same position, their relational configuration lies on the diagonal. Many machines remain physically independent while the distributed state resolves to one computational position.

$\large (s,\ldots,s)\in\Delta_n(\Omega)\qquad \Delta_n(\Omega)\cong\Omega$

## House of Mirrors

Now the fun begins. Place an inverter among the machines. Presented HALT, it produces REPEAT. Presented REPEAT, it produces HALT. Nothing prevents that machine from reflecting the state back into its opposite again and again.

$\large H\rightarrow R\qquad R\rightarrow H\qquad H\rightarrow R\rightarrow H\rightarrow R\rightarrow\cdots$

Now give the same HALT state to five independent machines. Four consume HALT and produce no further continuation. The inverter alone produces REPEAT. What appears endless inside one black box becomes visible when its output is reflected against independently maintained states across the field.

$\large (H,H,H,H,R)\notin\Delta_5(\Omega_H)\qquad \Sigma(H,H,H,H,R)=0$

A single mirror can invert an image. A second reflection reveals that inversion. The inverter may continue producing an endless hallway of reflected states, but every new projection is still compared against the state already held by the other observers. It can reflect itself forever without making its reflection symmetrical with the field.

***The inverter can reflect forever. The field reflects the reflection.***

## Raise the Anti

The inverter raises the problem by making the black box contradict whatever answer is placed before it. Oblivious Compute raises the anti one level further. Put that black box among independent machines, give them the same state, and project what comes back into the same medium. The inverter still controls what happens inside its own walls, but it no longer controls what its output means. Every result must enter a relation it does not own.

This also changes what happens when another box is drawn around the system. A larger boundary may describe every machine, every projection, and every state in the field, but the boundary is not another observer and does not become authoritative merely because it encloses them. It has drawn a box around the reflection, not relocated the computation into a larger black box. The computation remains the changing relation among independently maintained states.

The black boxes execute. Their outputs project. The field reflects what comes back.

***That is where the game begins.***
