# Appendix A — Definitions

## Purpose

This appendix establishes the terminology used throughout *Oblivious Compute*. Many of these definitions differ subtly from their conventional interpretation. Readers are encouraged to begin with **History**, as the distinction established there underlies much of the theory.

---

## History

History is a narrative describing the sequence of events believed to have produced a present state.

History is not the same as the evidence that survives from the past.

Evidence may support a historical narrative without uniquely determining one.

History may therefore be complete, partial, uncertain, or entirely unknown.

*Example*

*Imagine an ancient coin is unearthed.*

*The coin bears an unfamiliar face, an undeciphered language, and symbols whose meaning has been lost to time.*

*The coin unquestionably exists.*

*It is evidence originating from the past.*

*What it cannot do, by itself, is uniquely reconstruct the sequence of events that produced its present location.*

*We do not know who minted it, who carried it, where it circulated, why it was lost, or the complete sequence of events that ultimately brought it to where it was discovered.*

*The artifact is real.*

*The historical narrative is uncertain.*

***Oblivious Compute relies upon this distinction. Present state may be verified directly, even when the complete historical sequence that produced that state cannot be uniquely reconstructed.***

---

## State

The complete description of a system at a particular instant.

State is the object evaluated by Oblivious Compute.

---

## Genesis

The initial admissible state from which all subsequent admissible states originate.

Every admissible state is ultimately derived from a genesis.

---

## Transition

A proposed transformation from one state to another.

A transition has no authority until its resulting state is determined to be admissible.

---

## Admissibility

The property that determines whether a proposed state satisfies every invariant required by the system.

Only admissible states may exist within the geometry.

---

## Invariant

A property that must remain true for every admissible state.

Violation of any invariant renders a proposed state inadmissible.

---

## Geometry

The complete set of admissible states together with every admissible transition connecting those states.

The geometry defines every possible evolution of the system independently of any particular history.

---

## Observer

Any participant capable of evaluating the admissibility of a proposed state.

Observers verify state.

Observers do not reconstruct history.

---

## Verification

The process of determining whether a proposed state is admissible.

Verification depends upon the present state and the governing invariants.

---

## Reality

The collection of all presently admissible states.

Reality is maintained through continual verification rather than historical agreement.

---

## Synchronization

The process by which observers update their local view to reflect the current admissible state.

Synchronization concerns present reality rather than historical reconstruction.

---

## Equivocation

The proposal of multiple incompatible future states from the same prior state.

Equivocation does not create multiple realities.

Only admissible successor states may persist.
