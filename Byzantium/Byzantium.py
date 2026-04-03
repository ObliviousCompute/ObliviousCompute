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
