# HaltingMachine

A tiny loopback-socket specimen of an Oblivious Compute field.

Install from this directory:

```bash
pipx install --force .
HaltingMachine
```

Choose the number of observers and the number of inverters. The launcher assigns the last *N* observers as inverters, randomly chooses which observer inherits the invoking shell, and drops every observer into Genesis. The random choice affects terminal persistence only; it does not change identity, role, or field behavior.

## Runtime

- `↑` / `↓` selects `HALT` or `REPEAT`.
- `SPACE` exits that observer.
- Every observer begins on `HALT` and `🪨`.
- A healthy observer advances `🪨 → 📜 → ✂️ → 🪨`.
- An inverted observer advances `🪨 → ✂️ → 📜 → 🪨`.
- Healthy `REPEAT` originates its forward successor once per second; healthy `HALT` originates nothing.
- Inverted `HALT` originates its reverse successor once per second; inverted `REPEAT` originates nothing.
- A projection never mutates its sender. A state changes only after receiving the one successor admitted by that observer's transition rule.
- Every accepted state is semantically reprojected once into the oblivious medium.
- Acceptance starts a one-second refractory interval. R/P/S packets arriving during that interval are discarded, so each observer can mutate at most once per second.
- Active origination is phase-reset by the same acceptance. The field therefore appears clocked at 1 Hz without a shared master clock.

There is no DDUP/history window in this specimen. The present R/P/S state itself makes duplicates inadmissible: after accepting a ticket, that same ticket is no longer the observer's successor.

The red-handle scissors wire state is the real UTF-8 `✂️` emoji. Its display row inserts one terminal cell after each glyph because some Linux terminal/font combinations draw the emoji across two cells while advancing the cursor by one.

On exit, HaltingMachine restores the terminal, clears the screen, and returns the original window to a clean shell. Ctrl-C exits cleanly; Ctrl-Z and Ctrl-\ are ignored while the machine owns the terminal.
