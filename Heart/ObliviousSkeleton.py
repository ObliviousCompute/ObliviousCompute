# ==============================================
# ObliviousSkeleton v0.1 — Truth Through Erasure
# No time. No replay. No logs.
# ==============================================
ROCK, PAPER, SCISSORS = 1, 2, 3
NEXT = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}

def Dream(P):
    return P.get("is_seed") or P.get("is_snapshot") or P.get("is_dream")

def Admit(r_in, r_cur):
    return r_in in (r_cur, NEXT[r_cur])

def Step(S, P):
    if Dream(P):                          # hydrate only; never competes
        if P.get("tallies") and P["tallies"] != S["tallies"]:
            S["tallies"] = dict(P["tallies"])
        S["envy"] = False
        return S, []

    r_cur, r_in = S["rps"], P["rps"]
    if not Admit(r_in, r_cur):            # out-of-window → envy + sync request
    # ↑↑↑↑↑ LinchPin ↑↑↑↑↑
       
        intents = ([] if S["envy"] else ["ENVY"])
        S["envy"] = True
        return S, intents + ["REQUEST_SYNC"]

    S["envy"] = False
    if P["tallies"] == S["tallies"]:      # tallies-only no-op (ignore rps-only)
        return S, []

    S["tallies"], S["rps"], S["head"] = dict(P["tallies"]), r_in, P.get("id")
    intents = ["PROPAGATE"]
    if r_in == NEXT[r_cur]:
        intents.append("REQUEST_SYNC")
    return S, intents
