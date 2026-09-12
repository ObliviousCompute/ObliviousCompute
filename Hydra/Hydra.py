import sys
import MacAttack
from Game import Mutate

def main():
    return MacAttack.MacAttack() if len(sys.argv) > 1 and sys.argv[1].casefold() == "macattack" else Mutate.Mutate()
if __name__ == "__main__":
    main()
