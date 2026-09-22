from __future__ import annotations
import sys
from Game.Guardian import Run as RunCerberus


def main():
    args = sys.argv[1:]
    if args and args[0] in ("DevilDog", "LuckyDog"):
        trial = args[0]
        if len(args) > 2 or (len(args) == 2 and args[1] != "proofs"):
            raise SystemExit(f"usage: cerberus {trial} [proofs]")
        if trial == "DevilDog":
            from DevilDog import Run as RunTrial
        else:
            from LuckyDog import Run as RunTrial
        RunTrial(proofs_only=len(args) == 2)
        return
    if args:
        raise SystemExit("usage: cerberus [DevilDog|LuckyDog [proofs]]")
    RunCerberus()


if __name__ == "__main__":
    main()
