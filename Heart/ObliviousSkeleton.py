# ==============================================
# ObliviousSkeleton v0 — Truth Through Erasure
# No time. No replay. No logs.
# ==============================================
ROCK, PAPER, SCISSORS = 1, 2, 3
NEXT = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}

def DomKey(p):  # deterministic tie-break within same partition
    return (p["weight"], p["id"], p["h"])

def Admit(c_in, c_cur):  # RPS gate: only CURRENT or NEXT is admissible
    return c_in in (c_cur, NEXT[c_cur])

def Step(S, P):
    admissible = (P["rps"] in (S["rps"], NEXT[S["rps"]]))
    survivor   = max(S, P, key=lambda X: (X["rps"], DomKey(X)))

    S_next = (survivor if admissible else S)
#                   ^^^^^^^^
#  junction: GATE → ORDERING → OVERWRITE
    return S_next
