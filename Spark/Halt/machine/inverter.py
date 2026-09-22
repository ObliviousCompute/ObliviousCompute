import sys
from .observer import REVERSE, run as observe


def run(index, total):
    observe(index, total, step=REVERSE, invert_controls=True, invert_mode=True)


def main():
    index, total = map(int, sys.argv[1:3])
    run(index, total)


if __name__ == "__main__": main()
