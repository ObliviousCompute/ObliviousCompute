# The Boolean Hypercube

## Hexagram

Start with **three** binary cells. Here, a **trit** means those three cells considered together. Each arrangement is one complete state, and that complete three-bit state is what a machine projects into the medium. With **three** cells there are only **eight** possible positions.

$\Large 000\quad001\quad010\quad011\quad100\quad101\quad110\quad111$

A receiving machine admits the whole state, changes one cell, and projects another whole state. Pick any route between the two fully uniform positions and the trit walks one binary step at a time.

$\Large 000 \rightarrow 001 \rightarrow 011 \rightarrow 111 \qquad 0 \rightarrow 1 \rightarrow 2 \rightarrow 3$

***Three binary cells. Eight positions. Two opposite ends.***

## Metatron

Now take those same **eight** positions and sort them only by how many bits differ from **000**. Nothing has been added to the trit. No new state has been created. We are only changing how we look at the positions it already had.

$\Large 000 \|\ 001\ 010\ 100 \|\ 011\ 101\ 110 \|\ 111$

These groups are Hamming shells. **000** and **111** are antipodal poles. The first shell sits **zero** steps from **000**; the next **three** states sit **one** step away; the next **three** sit **two** steps away; and **111** sits **three** steps away. Change a single bit and the state moves between neighboring shells. Those one-bit relationships are not merely a sequence—they are exactly the edges of a three-dimensional Boolean cube.

A single machine never needs to store that cube. It holds only one trit state at a time. The cube appears only when the possible states are relationally superimposed.

***A single machine sees three bits. The superposition sees a cube.***

## Tesseract

Add one more binary cell. The rule does not change: two positions are neighbors when they differ by one bit. What changes is the number of positions hiding inside each Hamming shell.

$\Large 1|3|3|1 \qquad 1|4|6|4|1$

The gradient has grown by only one step, but its middle has opened. A cube has four Hamming shells; a tesseract has five. The walk is still made of single-bit changes, yet there are more positions and more possible routes inside the same simple rule. Increasing the number of cells increases the **nebulosity** of the geometry without changing the rule that generates it.

A classical machine may occupy one position in that geometry at a time. Quantum mechanically, amplitudes may occupy the geometry.

***The rule stays small. The space unfolds.***
