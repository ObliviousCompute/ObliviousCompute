# Black Box

## The Oblivious Halting Machine

A computer is ordinarily treated as the locus of its computation. A state enters the machine, the machine executes, and a result emerges. Whatever comes out is treated as the output of the computation.

Oblivious Compute moves the computational locus one step outward. A machine still executes locally, but its output is only a projected state. When multiple independent machines receive the same state and perform the same computation, no individual result is authoritative. The distributed computation exists in the relation among the states they independently produce.

> ***The black boxes execute. Their relation computes.***

Let an observer occupy a state within a state space $\Omega$. Given a presented state $x\in\Omega$, the observer determines whether that state belongs from its present position according to the admissibility relation $\mathcal A:\Omega\times\Omega\rightarrow\{0,1\}$.

Now give the same state $x$ to $n$ independently operating machines. Each produces some result $F_i(x)$, giving an aggregate configuration $(F_1(x),F_2(x),\ldots,F_n(x))\in\Omega^n$. No individual $F_i(x)$ is the distributed answer. What matters is the symmetry $\Sigma$ among the independently produced states.

When every machine independently resolves to the same state $s$, the resulting configuration is $(s,\ldots,s)\in\Delta_n(\Omega)$, and the diagonal is canonically isomorphic to the original state space:

$\Delta_n(\Omega)\cong\Omega$

Many independently executing machines therefore participate in a relational configuration that resolves to a single computational state. The state is not selected by a machine. It is revealed through the relation among machines.

## The Ticket

Imagine a row of black boxes around a table. Give every box the exact same ticket. The ticket contains state. Each box performs whatever internal computation it performs and spits another ticket back onto the table.

Nothing inside any one box is authoritative. The boxes may contain different processors, different implementations, or arbitrarily complicated machinery. The important fact is that every box was given the same computational position. What comes back can therefore be evaluated relationally.

If every conforming box receives the same state and applies the same rule, the resulting states should remain compatible with the same relation. The tickets coming out are not merely reports about a computation occurring somewhere else.

***The relation among the tickets is the computation.***

Consider a minimal halting state space $\Omega_H=\{R,H\}$, where $R$ denotes running and $H$ denotes halted. Let the halt state be idempotent, so that $\mathcal A(H,H)=1$, while a return from an established halt state to running is inadmissible, so that $\mathcal A(H,R)=0$.

Once an observer occupies $H$, receiving $H$ again changes nothing. It is already there. A later projection of $R$ does not require the observer to reconstruct a history, inspect another machine, or determine why the conflicting state was produced. From its present position, $R$ simply does not belong.

The computation asks only:

***Does this belong from here?***

## The Inverter

Now place an inverting machine at the table. When presented with `HALT`, it continues. When presented with `DO NOT HALT`, it halts. Internally, it may behave however its construction requires. It may continue executing forever.

Give five independent machines the same established state $H$. Four return $H$. The inverter returns $R$. The aggregate configuration is now $(H,H,H,H,R)$.

That configuration is visibly asymmetric:

$(H,H,H,H,R)\notin\Delta_5(\Omega_H)$

The inverter has not redefined the meaning of the established state. It has produced an incompatible projection. For observers already occupying $H$, another $H$ remains idempotent while $R$ remains inadmissible.

The inverting process may continue to execute. It may continue to project states into the medium indefinitely. None of this requires the other observers to leave the state they already occupy.

> ***A constituent process may run forever after the distributed computation has halted.***

## No Authority

The important distinction is not merely redundancy. It is authority.

After the state space and admissibility relation have been established, no individual participant determines the distributed computation. Each machine maintains state, performs local computation, and projects whatever state it produces. A projection does not become computationally authoritative merely because a machine produced it.

The machine executes locally. The projected state becomes available. The relation determines what continues to belong.

$\text{local execution}\rightarrow\text{projected state}\rightarrow\text{relational computation}$

A machine may be correct, faulty, divergent, adversarial, or indefinitely executing. Those are properties of the machine. Continued participation in the field is a property of the relation.

This separates physical execution from distributed computation.

$\text{process still running}\not\Rightarrow\text{field still computing}$

The black box may continue doing whatever happens inside the black box. Once its output no longer belongs from the state independently maintained by the other observers, continued execution does not grant continued computational authority.

## The Black Box

The classical black box places the computational question inside the machine. Something enters, something happens, and something comes out. To understand the computation, attention naturally turns toward whatever machinery exists behind the walls.

Oblivious Compute does not require the walls to be opened.

Put the same state into multiple independent black boxes. Let them execute. Let them project what comes back. The distributed computational object is then formed outside every individual box, through the relation among those independently produced states.

A black box may contain a conventional program. It may contain another distributed system. It may contain a machine deliberately constructed to contradict a result presented to it. Its internal complexity does not make its output authoritative.

The box receives state. The box executes. The box projects state.

Then the computation begins.

> ***Put the same ticket into every black box. Let the boxes execute. Compare what comes back. The relation is the computer.***
