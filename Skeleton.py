# ============================================== #
# Skeleton v1.0 — Truth Through Erasure
# No time. No replay. No logs.
# ============================================== #

# ============================================== #
# BRAID SEQUENCE (NOT TIME)
# ============================================== #
# Rock, Paper, Scissors is NOT a clock.
# It is NOT time. It is NOT ordering.
#
# It is a braid sequence.
#
# ROCK → PAPER → SCISSORS → ROCK
#
# Each position yields to the next.
#
# Past     → already accepted
# Present  → currently held
# Future   → next admissible
#
# The system does not move through time.
# It waits for the next admissible position.
# All change happens at that boundary.
# ============================================== #


ROCK, PAPER, SCISSORS = 1, 2, 3

NEXT = {
    ROCK: PAPER,
    PAPER: SCISSORS,
    SCISSORS: ROCK,
}


def isDream(packet):
    return (
        packet.get("isDream")
        or packet.get("isSnap")
        or packet.get("isSeed")
    )


def inWindow(incoming, current):
    return incoming in (current, NEXT[current])


def ingest(state, packet):

    # Dream → rehydrate only
    if isDream(packet):
        if packet.get("tallies") and packet["tallies"] != state["tallies"]:
            state["tallies"] = dict(packet["tallies"])
        state["envy"] = False
        return state, []

    current = state["sequence"]
    incoming = packet["sequence"]

    # ========== LINCHPIN ========== #
        ↓    ↓    ↓    ↓    ↓    ↓            
    if not inWindow(incoming, current):
        intents = ([] if state["envy"] else ["ENVY"])
        state["envy"] = True
        return state, intents + ["Sync"]       
        ↑    ↑    ↑    ↑    ↑    ↑     
    # ========== LINCHPIN ========== #

    state["envy"] = False

    if packet["tallies"] == state["tallies"]:
        return state, []

    state["tallies"] = dict(packet["tallies"])
    state["sequence"] = incoming
    state["head"] = packet.get("id")

    intents = ["PROPAGATE"]

    if incoming == NEXT[current]:
        intents.append("Sync")

    return state, intents
