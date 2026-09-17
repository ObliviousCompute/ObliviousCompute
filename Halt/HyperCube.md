# The Boolean Hypercube

## Hexagram

Start with three binary cells. Here, a **trit** means those three cells considered together. Each cell can hold 0 or 1, so the trit has eight possible positions. Nothing else is required.

$\Large 000 \quad 001 \quad 010 \quad 011 \quad 100 \quad 101 \quad 110 \quad 111$

A move changes one cell at a time. Pick any route from one fully uniform state to the other and the trit walks one step at a time between two opposite ends.

$\Large 000 \rightarrow 001 \rightarrow 011 \rightarrow 111 \quad 0 \rightarrow 1 \rightarrow 2 \rightarrow 3$

***Three binary cells. Eight positions. Two opposite ends.***

## Metatron

Now take those same eight positions and sort them only by how many bits differ from 000. Nothing has been added to the trit. We are only changing how we look at the positions it already had.

$\Large 000 \|\ 001\ 010\ 100 \|\ 011\ 101\ 110 \|\ 111$

Those groups are Hamming shells. The first position is zero steps from 000; the next three are one step away; the next three are two steps away; 111 is three steps away. States in neighboring shells differ by one bit. Those one-bit relationships are exactly the edges of a three-dimensional Boolean cube.

No machine needs to store the cube. A machine holds only one trit state at a time. The cube appears from the fixed relations among the eight possible states.

***The machine sees three bits. The relation sees a cube.***

## Tesseract

Add one more binary cell. The rule does not change: two positions are neighbors when they differ by one bit. What changes is the number of positions hiding inside each Hamming shell.

$\Large 1|3|3|1 \quad 1|4|6|4|1$

The gradient has grown by only one step, but its middle has opened. A cube has four Hamming shells; a tesseract has five. The walk is still made of single-bit changes, yet there are more positions and more possible routes inside the same simple rule. Increasing the number of cells increases the **nebulosity** of the geometry without changing the rule that generates it.

A classical machine may occupy one position in that geometry at a time. Quantum mechanically, amplitudes may occupy the geometry.

***The rule stays small. The space unfolds.***
