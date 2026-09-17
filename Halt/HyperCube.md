# The Boolean Hypercube

## Hexagram

A binary distinction has two positions. One observer may hold $0$ or $1$; nothing more is required. Place several independently maintained binary distinctions beside one another and the same two-position primitive acquires geometry. For $n$ observers, the relational configuration may be written as a point in the Boolean hypercube $Q_n$. The notation describes the configuration; no observer is required to contain the whole point as an authoritative global state.

$\large b_i\in\{0,1\}\qquad Q_n=\{0,1\}^n\qquad |Q_n|=2^n$

The two maximally symmetric positions sit at opposite poles of the space. One pole contains only zeros, the other only ones. Every mixed configuration lies between them. The smallest visually useful case is three binary distinctions: eight possible configurations arranged as the vertices of an ordinary cube. The cube is not added to the computation. It appears when independent binary positions are considered together.

$\large 0^n=(0,0,\ldots,0)\qquad 1^n=(1,1,\ldots,1)\qquad Q_3=\{0,1\}^3$

A local rule can remain almost embarrassingly small. From its present position, an observer may treat the state it already holds as idempotent and admit a projected state lying one edge farther in the permitted direction. If the active pole is $\tau\in\{0,1\}$, Hamming distance gives the direction without supplying a route.

$\large D_\tau(q)=\sum_{i=1}^{n}(q_i\oplus\tau)\qquad A_\tau(q,x)=1\ \text{if}\ x=q\ \text{or}\ D_\tau(x)=D_\tau(q)-1$

Different executions may walk different edges. The rule does not choose a path through the cube; it gives the cube an orientation.

***A bit has two positions. A population of bits has somewhere to go.***

## Metatron

Once admitted state is reprojected, the geometry begins to move. An observer encounters a projection, determines whether it belongs from its present position, admits it when permitted, and may project the resulting state again. One observer may change one binary distinction; another may encounter that new position and continue from there. No participant needs the route by which the state arrived. The current position is sufficient to determine the next admissible edge.

For three binary distinctions oriented toward $000$, one execution may descend:

$\large 111\rightarrow110\rightarrow100\rightarrow000\qquad D_0:3\rightarrow2\rightarrow1\rightarrow0$

Another execution may take a different route through exactly the same geometry:

$\large 111\rightarrow011\rightarrow001\rightarrow000\qquad D_0:3\rightarrow2\rightarrow1\rightarrow0$

The intermediate states form a Boolean cloud. There may be many paths, many projections, and many transient configurations, but the orientation remains simple. Toward the zero pole, healthy motion removes a $1$. Toward the one pole, healthy motion removes a $0$. The two computations are mirror images over the same state space.

$\large 1^n\longrightarrow\cdots\longrightarrow0^n\qquad\qquad 0^n\longrightarrow\cdots\longrightarrow1^n$

Now invert the operator. Nothing about the hypercube changes. The observer still moves one edge at a time, but its local orientation reverses. At a settled pole, ordinary progression has no farther edge toward coherence. An inverted move necessarily leaves that pole and enters the first Hamming shell.

$\large D_\tau=0\quad\longrightarrow\quad D_\tau=1$

For the three-dimensional case, $000$ may be left only through $001$, $010$, or $100$. These are different vertices but the same relational displacement from the pole. Inversion has become geometry: it is no longer merely the opposite bit, but motion against the prevailing orientation of the field.

If terminal projections are treated as immediately admissible by observers operating under the same orientation, a population may migrate through the cloud and then snap to the first terminal pole it reaches. A later reversal of the active orientation releases the population and sends it through the same geometry toward the opposite pole.

***The cloud remembers no path. It only has a direction, a distance, and a place to settle.***

## Tesseract

Nothing in the construction depends on the cube remaining three-dimensional. Three binary distinctions produce eight vertices. Four produce the sixteen vertices of a tesseract. In general, $n$ independently maintained binary distinctions produce an $n$-dimensional Boolean hypercube containing $2^n$ possible configurations.

$\large Q_3=\{0,1\}^3\qquad Q_4=\{0,1\}^4\qquad Q_n=\{0,1\}^n$

The raw configuration space grows exponentially, but the simplest relational description need not. Relative to a selected pole, every vertex belongs to one of only $n+1$ Hamming shells. An enormous cloud may therefore be organized by a very small quantity: how many binary distinctions still disagree with the active symmetry.

$\large S_k=\{q\in Q_n:D_\tau(q)=k\}\qquad k\in\{0,1,\ldots,n\}$

A million binary distinctions induce $2^{1,000,000}$ possible vertices, yet their distance from a selected pole is still an integer between $0$ and $1,000,000$. The microscopic route may be combinatorially enormous while the macroscopic orientation remains unchanged.

The same idea can also be composed rather than flattened. A small group may resolve to a relationally coherent result; several such resolved groups may then participate as positions in another group, and the same local primitive may be applied again. Scale need not require a new rule. It may arise by repeating the same relation at a higher order.

$\large 3\rightarrow3^2\rightarrow3^3\rightarrow\cdots$

The hypercube therefore supplies both a finite geometry and a route to arbitrary scale. Local observers retain binary distinctions. Their relations supply the dimensions. Their admissible motion supplies the orientation. Their settled symmetry supplies the pole.

***The rule stays small. The space does not.***
