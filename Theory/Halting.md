# The Black Box

## The Oblivious Machine

A computer computes. A state enters the machine, the machine executes, and a result emerges. Whatever comes out is ordinarily treated as the computation.

Oblivious Compute moves the computation outside of the box.

A machine still executes locally, but its output is only a projected state. Give multiple independent machines the same state and no individual result needs to be authoritative. What matters is the relation among the states they independently produce.

> ***The black boxes execute. Their relation computes.***

Let an observer occupy a state within a state space $\Omega$. Given a presented state $x\in\Omega$, the observer determines whether that state belongs from its present position according to an admissibility relation $\mathcal A:\Omega\times\Omega\rightarrow\{0,1\}$.

Give the same state $x$ to $n$ independently operating machines. Each produces a state $F_i(x)$, forming $(F_1(x),F_2(x),\ldots,F_n(x))\in\Omega^n$.

No individual $F_i(x)$ is the distributed computation. The computation is the symmetry $\Sigma$ among the independently produced states.

When every machine resolves to the same state $s$:

$(s,\ldots,s)\in\Delta_n(\Omega)$

and:

$\Delta_n(\Omega)\cong\Omega$

Many independently executing machines have resolved relationally to one computational state.

## Price of Admission

Imagine a row of black boxes around a table. Give every box the exact same ticket. Each ticket contains the same state. Every box performs whatever computation exists behind its walls and returns another ticket.

The boxes may remain black. Their outputs do not.

Because every machine began from the same presented state, what comes back can be evaluated against the same admissibility rule. A machine does not gain authority merely by producing an answer. Its answer is another state seeking admission to the computation.

> ***Projection is free. Admission is conditional.***

Consider a minimal state space:

$\Omega_H=\{H,R\}$

where $H$ means **HALT** and $R$ means **REPEAT**.

Let HALT be idempotent:

$\mathcal A(H,H)=1$

and let REPEAT be inadmissible from an established HALT state:

$\mathcal A(H,R)=0$

Once an observer occupies $H$, another $H$ changes nothing. It is already there. A later $R$ does not require reconstruction of the machine's history or inspection of whatever happened inside the box.

It simply does not belong from here.

## House of Mirrors

Now place an inverter among the machines.

Presented with HALT, it returns REPEAT. Presented with REPEAT, it returns HALT:

$H\rightarrow R$

$R\rightarrow H$

The inverter may perform this reflection indefinitely. Nothing requires its internal process to stop.

Now give five independent machines the same state $H$. Four return $H`. The inverter returns $R$.

The resulting configuration is:

$(H,H,H,H,R)$

It is not perfectly symmetric:

$(H,H,H,H,R)\notin\Delta_5(\Omega_H)$

The inverter has not changed the established state. It has exposed its own incompatibility with it.

For every observer already occupying $H$, the distinction remains mechanical:

$\mathcal A(H,H)=1,\qquad\mathcal A(H,R)=0$

The four idempotent results remain part of the same relation. The inverted result does not.

The inverter may continue executing. It may continue projecting REPEAT, HALT, REPEAT, HALT forever. Those projections remain subject to the state already held by the other observers.

A non-halting process therefore need not produce a non-halting distributed computation.

$\text{process still running}\not\Rightarrow\text{computation still running}$

The mirror can reflect forever.

The field does not have to follow it.

## Outside the Box

The classical black box asks what happens inside a machine.

Oblivious Compute asks what happens between machines.

Put the same state into independent black boxes. Let them execute. Let them project what comes back. No box determines the distributed result simply by speaking first, speaking last, or continuing to speak forever.

The output of a black box is only another state.

Its computational meaning appears through its relation to the states independently maintained by the other observers.

The machine may be healthy, faulty, divergent, adversarial, or indefinitely executing. None of those conditions grants it authority over the larger computation. Continued participation is determined by admissibility.

The box may therefore remain physically opaque while becoming relationally transparent. We do not need to know every instruction executed behind its walls. We need to know the state that entered, the state that emerged, and whether that state belongs from the computational position already established.

The box receives state.

The box executes.

The box projects state.

***Then the computation begins.***

> ***Put the same ticket into every black box. Let the boxes execute. Compare what comes back. The relation is the computer.***
