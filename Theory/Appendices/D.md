# Appendix D — Definitions

## Purpose

This appendix establishes the fundamental vocabulary used throughout the theory. Each definition introduces a single concept without assuming any particular implementation.

## Sequence

**A sequence is an ordered progression of states or events.**

### *Example*

*A classical Turing machine advances through successive configurations. The same sequential structure appears throughout classical computation: processors, cores, threads, pipelines, caches, memory systems, finite-state machines, GPUs, FPGAs, and ASICs all progress through ordered states or events.*

*Parallel and concurrent systems compose multiple such sequences at once, while distributed systems maintain sequences independently across participants. Parallelism changes the number and relationship of sequences; it does not eliminate sequence.*

## History

**History is an account of the sequence of events believed to have produced a present state.**

### *Example*

*History is distinct from the evidence that survives from the past. Evidence may support a historical account without uniquely determining one.*

*Imagine an ancient coin is unearthed. The coin unquestionably exists and is evidence originating from the past. By itself, however, the coin cannot uniquely determine the sequence of events that produced its present location.*

## Consensus

**Consensus is a process by which independent participants resolve potentially different views into a common decision.**

### Non-example

*Two fans watch the same basketball game from opposite sides of the court. They may disagree about who had the ball, who touched it, or exactly how a play unfolded. There is no referee reconciling their accounts. A point is counted only when the ball goes through the hoop.*

*The fans can retain different accounts of the game. Their perspectives have not been brought into consensus, but they can still agree that the score is correct.*

## Computation

**A computation is a rule-governed transformation or determination of state.**

### Example

*Computation does not require an electronic computer. An abacus transforms a represented numerical state according to rules. A punch-card loom uses the geometry encoded in punched cards to determine the state of its mechanism and the resulting fabric. Mechanical calculators, Turing machines, and modern processors likewise compute by applying rules to state.*

## Byzantine Generals Problem

**The Byzantine Generals Problem asks whether loyal participants can coordinate on a common course of action despite traitorous participants sending conflicting information.**

### IC1. 

*All loyal lieutenants obey the same order.*

### IC2. 

*If the commanding general is loyal, every loyal lieutenant obeys the order he sends.*

---

**Go back to [**`Theory`**](../README.md) or continue to [**`Skeleton`**](../.././Skeleton/README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
