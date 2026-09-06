# Doghouse

Cerberus comes with two built-in Doghouse demos.

Here’s the format:

```bash
Cerberus DevilDog proofs
```

`Cerberus` runs the installed game.

`DevilDog` chooses the demo.

`proofs` skips the interactive run and goes straight to the evidence from that execution.

Remove `proofs` if you want to watch the demo play out interactively.

If you’re running Cerberus directly from this folder instead of through pipx, use `python Cerberus.py` in place of `Cerberus`.

LuckyDog uses the exact same format. Just replace `DevilDog` with `LuckyDog`.

## DevilDog

Double Dog puts five greedy dogs in a field with four loyal dogs.

The greedy dogs create conflicting signed spends while ordinary gameplay continues. Their delayed evidence eventually hits Oblivion and Cerberus has to reconcile the resulting field.

Watch the dogs get Razed, the Bone Bucks settle, and all nine Heads Bury the same 99-bone BonePile.

## LuckyDog

Lucky Dog keeps going.

Dogs progressively get greedy and promise more bones than they can cover. Cerberus Razes them one by one until Lucky is the last dog standing with all 99 bones.

The interactive demos show it happen.

The proofs show what actually happened.
