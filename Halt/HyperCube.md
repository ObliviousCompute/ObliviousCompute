# The Boolean Hypercube

## Hexagram

Start with three binary cells. Each cell can be either `0` or `1`.

Here, **trit** means a bundle of three binary cells, not a base-three digit.

That gives exactly eight possible trit states:

\[
000 \quad 001 \quad 010 \quad 011 \quad 100 \quad 101 \quad 110 \quad 111
\]

Nothing strange has happened yet. Three cells simply have eight possible arrangements.

The two outside positions are exact opposites:

\[
0 \equiv 000
\qquad\qquad
1 \equiv 111
\]

They are antipodal. One is all zero. The other is all one.

The six states between them are not extra logical values. They are just the possible positions a three-cell state can occupy between the two poles.

***Three binary cells. Eight positions. Two opposite ends.***

## Metatron

Now sort the same eight states by how many bits must change to reach `000`.

\[
000
\quad | \quad
001,\ 010,\ 100
\quad | \quad
011,\ 101,\ 110
\quad | \quad
111
\]

Those bars are Hamming shells.

The first shell is zero steps from `000`.
The next shell is one step away.
The next is two steps away.
The last is three steps away.

Read the same trit from the other direction and the order reverses:

\[
111
\quad | \quad
011,\ 101,\ 110
\quad | \quad
001,\ 010,\ 100
\quad | \quad
000
\]

A single-bit change moves the state one shell at a time.

\[
111 \rightarrow 110 \rightarrow 100 \rightarrow 000
\]

Another walk can take a different route:

\[
111 \rightarrow 011 \rightarrow 001 \rightarrow 000
\]

The route does not matter. Every legal one-bit move travels along one edge of the same cube.

That cube does not need to be stored anywhere. No cell contains it. A machine only holds one trit state at a time.

The cube appears because the eight possible trit states have a fixed one-bit relationship to one another.

***The machine sees three bits. The relation sees a cube.***

## Tesseract

Add one more binary cell.

Three cells have:

\[
2^3=8
\]

possible states.

Four cells have:

\[
2^4=16
\]

possible states.

The same rule still works: two states are neighbors when they differ by one bit.

The ordinary cube has become a four-dimensional Boolean cube: a tesseract.

Nothing new had to be invented. The state simply gained another binary position.

The same idea continues for any number of binary cells. More cells do not change the rule. They only create more possible positions and more possible walks.

A classical machine may walk those positions one at a time. A quantum walk can place amplitude across many positions of the same Boolean geometry before measurement. The geometry is the same; the way the machine occupies it is different.

***The rule stays small. The space unfolds.***
