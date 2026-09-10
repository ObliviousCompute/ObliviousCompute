# The Black Box

## The Oblivious Machine

A computer computes. State enters the machine, the machine executes, and something happens next. Conventionally, whatever happens inside that box is treated as the computation, while whatever emerges is treated as its result. The machine may be simple or arbitrarily complex, but the computational question remains pointed inward.

Oblivious Compute moves the distributed computation outside of the box. Give independent machines the same state $x$ and allow each machine $i$ to produce a state $F_i(x)$. No individual result is authoritative; the computation appears in the relation $\Sigma$ among those independently produced states.

$\large x\rightarrow\{F_1(x),F_2(x),\ldots,F_n(x)\}\qquad \Sigma(F_1(x),F_2(x),\ldots,F_n(x))$

When independently maintained states resolve to the same position, their relational configuration lies on the diagonal. Many machines remain physically independent while the distributed state resolves to one computational position.

$\large (s,\ldots,s)\in\Delta_n(\Omega)\qquad \Delta_n(\Omega)\cong\Omega$

## Price of Admission

Imagine several black boxes around a table. Put the exact same ticket into every box. Each ticket carries the same state. Whatever machinery exists behind the walls may remain completely unknown; what matters is that every machine began from the same presented state and whatever comes back must enter the same relational computation.

A returned ticket does not become authoritative because a machine printed it. It is simply another projection seeking admission from the state already held. For the smallest halting example, let $H$ mean **HALT** and $R$ mean **REPEAT**. A healthy machine presented HALT consumes the state and emits no further ticket. If another ticket appears, feed it back into the same oblivious medium and evaluate it again.

$\large \Omega_H=\{H,R\}\qquad H\rightarrow\varnothing\qquad \mathcal A(H,R)=0$

Projection is unrestricted. Continued participation is not.

## House of Mirrors

Now place an inverter among the machines. Presented HALT, it produces REPEAT. Presented REPEAT, it produces HALT. Nothing prevents the inverter from performing this reflection forever. Every output becomes another ticket, every ticket may be projected again, and the machine may continue executing for as long as its own internal rule demands.

$\large H\rightarrow R\qquad R\rightarrow H\qquad H\rightarrow R\rightarrow H\rightarrow R\rightarrow\cdots$

Now give the same HALT state to five independent machines. Four consume HALT and produce nothing further. The inverter produces REPEAT. The important result is not hidden inside any box: one machine has produced a continuation from a state for which the other machines have no continuation. What looked like an endless logical reflection inside one box has become visible asymmetry outside of it.

$\large (H,H,H,H,R)\notin\Delta_5(\Omega_H)\qquad \mathcal A(H,R)=0$

The inverter may continue. Its next reflection does not acquire authority merely because another ticket appears. The machines already settled at HALT do not have to follow it, reconstruct its history, or discover why it behaved differently. From the state they already hold, its continuation simply does not belong.

The mirror can reflect forever. The computation does not have to move with it.

$\large \text{process still running}\not\Rightarrow\text{distributed computation still running}$

## Outside the Box

The black box may remain black. Its internal instructions, timing, circuitry, implementation, and private execution need not be reconstructed. Give independent boxes the same state, observe what comes back, and compare those projections from the computational position already established. A box that behaves differently exposes that difference through state.

The ticket itself is not the computer, and neither is any individual box. The boxes execute locally. Their outputs become projections. The relation among those independently produced states determines what continues to belong. What was opaque inside one machine becomes computationally visible outside of it.

$\large \text{state in}\rightarrow\text{boxes execute}\rightarrow\text{states out}\rightarrow\Sigma$

The black box need not be opened. Oblivious Compute places the black box inside an oblivious medium, where its output becomes relative to independently produced state. What is unknowable inside one machine becomes observable as relation outside it.

The box receives state. The box executes. The box projects state.

***Then the computation begins.***
