# =================================
# Hydra Proofs — Minimal Oblivious Convergence Skeleton
# =================================
# This file illustrates the core mechanism only.
# It is not a runnable implementation.


def hydra_step(local_state, incoming_states):
    candidates = [local_state] + incoming_states

    survivor = select(candidates)      # deterministic
    erase(candidates, survivor)        # total erasure

    return survivor


def node_loop():
    state = initial_state()

    while True:
        incoming = receive()
        state = hydra_step(state, incoming)
