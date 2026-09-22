# The death of a Byzantine:
# equivocation is tolerated while the actor can still pay.
# Cross the estate boundary and the field closes around the fork:
# salt to zero, receipts remain, agency ends.

import os
import sys

base = os.path.dirname(os.path.abspath(__file__))
game = os.path.join(base, "Game")

if game not in sys.path:
    sys.path.insert(0, game)

import Gateway

def main():
    Gateway.main()

if __name__ == "__main__":
    main()
