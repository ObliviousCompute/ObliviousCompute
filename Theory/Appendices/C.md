# Appendix C — Criticism

> **For criticism**
>
> Oblivious Compute can be examined through a small set of participant roles. These terms are not intended to restrict how the system may be interpreted. They are simply handles for constructing experiments, describing privilege, and making clear what kind of machine is actually being tested. 
>
> When criticizing an execution, identify **what each participant can observe, what information it may trust, and what causal authority it possesses**. A counterexample is most useful when the machinery introduced to produce it is named explicitly.

## Observer

An **observer** is a state-bearing participant inside the machine. It maintains its own state and acts from the information available at its position in the system. An observer does not require a global representation of the execution and does not need to calculate the field in order to participate in it.

The field is not contained by any observer. It arises from the **relation among independently maintained observer states**.

### Inverted Observer

An **inverted observer** is an observer whose local behavior intentionally opposes, reverses, frustrates, or otherwise tests the relation being formed.

An inverted observer is **not informationally privileged**. It remains bound by the same local informational position as an ordinary observer. Its importance comes from what it does with the information available to it, not from access to information unavailable to the other observers.

Inverted observers are useful for testing whether behavior attributed to the field can arise even when one participant would not locally produce that behavior on its own.

### Oracle Observer

An **oracle observer** is an observer with privileged information unavailable to an ordinary observer at the same position.

The privilege may concern only a single predicate, fact, or class of questions. The oracle observer does not need global knowledge of the machine.

If another observer changes its behavior according to information supplied by an oracle observer, that observer must treat the oracle answer as authoritative with respect to that question. In this sense, **some epistemic authority has been outsourced**.

An oracle observer remains an observer because it still occupies a position inside the machine. Its oracle status comes from the additional information it possesses.

## Godhead

A **Godhead** occupies a globally privileged observational position relative to the execution.

Unlike an oracle observer, whose privilege may concern only a particular question, a Godhead may observe the machine across ordinary observer boundaries. It can therefore reconstruct or possess information that no individual observer is required to contain.

The important distinction is whether this global position is **merely observational** or becomes **causally necessary** to the execution.

### Passive Godhead

A **passive Godhead** has a globally privileged view of the execution but does not participate in causing it.

A packet sniffer, trace collector, or globally informed debugger is the simplest example. A passive Godhead may observe projections, reconstruct histories, calculate relational properties, or maintain a complete external record of the execution.

Its information is **not returned to the machine in a way that determines what happens next**.

Removing a passive Godhead therefore leaves the execution unchanged.

A passive Godhead is **instrumentation over the machine, not machinery within it**.

### Active Godhead

An **active Godhead** possesses globally privileged information and uses that privilege causally.

It may implement, emulate, schedule, route, suppress, modify, or otherwise mediate the communication medium. If a globally informed component determines which projections occur, where they are exposed, when they become available, or how the execution progresses, that component is no longer merely observing the medium.

It has become part of the mechanism producing the execution.

An active Godhead may reproduce the visible behavior of an oblivious medium, but the construction now depends upon **globally privileged causal authority** that the original observer geometry does not require.

## Constructing Experiments

Experiments may deliberately contain any combination of these roles.

A particularly direct adversarial construction may combine **Oracle Observers with an Active Godhead**. Such a system can introduce privileged local information together with globally informed control over the medium.

That is a valid experiment.

The useful question is then not only whether the resulting traces resemble an Oblivious Compute execution, but also **which informational and causal privileges were required to produce them**.

When describing an experiment, identify:

- **ordinary observers**
- **inverted observers**
- **oracle observers**
- **passive Godheads**
- **active Godheads**

The purpose of these terms is not to decide whether Oblivious Compute is novel, useful, universal, or equivalent to another model.

They exist so that those questions can be asked against a clearly identified machine.

**A counterexample should break the primitive, not quietly replace it.**

---

**Go back to [**`Theory`**](../README.md)...**

---

## 📜 License

See the [**`NOTICE`**](../../NOTICE.md) for licensing information on the [**`Oblivious Compute`**](https://github.com/ObliviousCompute) project.

Use it, study it, modify it—just respect the terms outlined there.
