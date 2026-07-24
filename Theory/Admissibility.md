# Linchpin

## The Shape

In a system like this, what does a computer's croak even look like?

Consider a chessboard.

Each piece carries its own invariants.  
A rook moves in straight lines.  
A bishop moves diagonally.  
A knight moves in its own pattern.  

These invariants define what is admissible.

A proposed position is accepted only if every piece satisfies its invariants.

If not, it is not part of the game.

A turn, then, is a cycle:  
a transition between admissible geometric states.

Reality is filtered by shape, not by choice.

---

## A Minimal Cycle

Consider the smallest non-trivial system: three shapes in a cycle.

Rock → Paper → Scissors → Rock.

It is tempting to think of this as *past*, *present*, and *waiting*.

This intuition is useful—but only briefly.

There is no past to reconstruct, and no future to predict. There is only a current shape, and a constrained set of admissible continuations.

Each shape admits itself and one successor. Nothing else.

This structure is a braid—a closed system of admissible transitions.

---

## The Boundary

All computation occurs at the boundary between a valid state and its admissible continuations.

This boundary is defined by a finite set of constraints, yet can be traversed indefinitely.

A system may remain poised at a valid state for any duration—microseconds or millennia—without affecting correctness.

When a transition occurs, it is evaluated only against what is currently valid and what is admissibly next.

Time does not participate in this evaluation. It is external to it.

Computation does not unfold through time, but across admissible states.

The braid defines the geometry of this boundary.

This boundary is the locus of computation.

---

## 📜 License

[**`Oblivious Compute`**](https://github.com/ObliviousCompute/ObliviousCompute/blob/main/README.md) is released under the terms of the [**`LICENSE`**](../LICENSE).

Use it, study it, modify it—just respect the terms outlined there.
