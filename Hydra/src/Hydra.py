# =================================================
# Hydra Demo v1.0 — Truth Through Erasure
# No time. No replay. No logs.
# =================================================
import argparse, json, socket, threading, hashlib, sys, termios, tty, os
from typing import Dict, Tuple, Any, List, Union, Optional

# =================================================
# Flesh & Bone
# =================================================
PRINT_LOCK = threading.Lock()

_INPUT_BUF = ""

# Envy flag: turns [Head:X] green when crown-gate misalign
ENVY = False

TAIL: Optional[Dict[str, Any]] = None

TAIL_STATE: Dict[str, Any] = {
    "tallies": {"A": 10, "B": 10, "C": 10, "D": 10, "E": 10},
    "crown": 1,
    "h": None,
    "head": None,
}
# Storm guards
HUNGER_REPLY_LAST: Dict[Tuple[str, int], str] = {}  # per-request throttling
SEEN_H: Dict[str, None] = {}                        # recent state-hashes only
SEEN_MAX = 4096                                     # bounded to avoid history

# =================================================
# Hydra Eyes
# =================================================
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"

# Color (only used when ENVY=True)
GREEN = "\x1b[32m"
RESET = "\x1b[0m"

# Crown Partition (Rock, Paper, Scissors)
CROWN_GEMS = {1: "JADE", 2: "ONYX", 3: "OPAL"}

heads = ["A", "B", "C", "D", "E"]
selected_head_idx = 0
selected_amount = 1


# =================================================
# Eyes: low-level
# =================================================
def _cursor_left(n: int) -> str:
    return f"\x1b[{n}D" if n > 0 else ""

def crown_tag(c: int) -> str:
    return CROWN_GEMS.get(int(c or 1), "C?")

def render_status(head_id: str, input_buf: str | None = None) -> None:
    global _INPUT_BUF

    if input_buf is None:
        input_buf = _INPUT_BUF
    else:
        _INPUT_BUF = input_buf

    crown = int(TAIL_STATE.get("crown", 1))
    tallies = TAIL_STATE.get("tallies", {}) or {}

    welcome_lines = [
        "",
        "  Hydra Head:{head} Ready...It's feeding time!!!",
        "",
        "  Go for it, let another set of jaws chomp on your tailies",
        "  and give of yourself freely. Nothing here is lost.",
        "  If another takes too much, draw it back through the ichor. ",
        "  Hydra feels no pain. It has no memory...",
        "",
        "  Use ← and → to select a head, then ↑ and ↓ for an amount.",
        "  Enter(FEED!)  Ctrl+X(Sever)  Ctrl+C(Cauterize)  H(Hunger)",
        "",
        "  Heads green with envy must be severed and rehydrated.",
        "",
        "",
    ]
    welcome_block = "\n".join(line.format(head=head_id) for line in welcome_lines)

    tail_line = " ".join(f"{k}{tallies.get(k, 'x')}" for k in ["A", "B", "C", "D", "E"])
    tail_part = f"[Tails:{tail_line}]>"

    head_label = heads[selected_head_idx]
    input_segment = f">[{head_label}:{selected_amount}]<"

    head_tag = f"{GREEN}Head:{head_id}{RESET}" if ENVY else f"Head:{head_id}"

    hud_line = f"  <[Crown:{crown_tag(crown)}][{head_tag}]{input_segment}{input_buf}{tail_part}"
    render_block = welcome_block + "\n" + hud_line
    tail_len = len(tail_part)

    with PRINT_LOCK:
        sys.stdout.write("\x1b[H")
        sys.stdout.write("\x1b[0J")
        sys.stdout.write(render_block)
        sys.stdout.write(_cursor_left(tail_len))
        sys.stdout.flush()

# =================================================
# Peripheral Nerves
# =================================================
def soft_reboot() -> None:
    # "Sever the head"
    global ENVY
    ENVY = False
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()
    print("\n\n  Regrowing Head...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

def read_cmd_and_restore_hud(head_id: str) -> Union[str, Tuple[str, str, int]]:
    global selected_head_idx, selected_amount, _INPUT_BUF

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    _INPUT_BUF = ""
    render_status(head_id, _INPUT_BUF)

    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "":
                raise EOFError
            # Enter = FEED!
            if ch in ("\n", "\r"):
                to = heads[selected_head_idx]
                if to == head_id:
                    selected_head_idx = (selected_head_idx + 1) % len(heads)
                    to = heads[selected_head_idx]
                    render_status(head_id)
                return ("ARROW_FEED", to, selected_amount)

            # Ctrl+C = cauterize
            if ch == "\x03":
                raise KeyboardInterrupt
            # Ctrl+X = sever
            if ch == "\x18" and _INPUT_BUF == "":
                soft_reboot()
            # H / h = manual hunger (force resync without sever)
            if ch in ("h", "H") and _INPUT_BUF == "":
                return "HUNGER"
            # Arrow keys
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == "[":
                    if ch3 == "C":      # RIGHT
                        selected_head_idx = (selected_head_idx + 1) % len(heads)
                    if heads[selected_head_idx] == head_id:
                        selected_head_idx = (selected_head_idx + 1) % len(heads)
                    elif ch3 == "D":    # LEFT
                        selected_head_idx = (selected_head_idx - 1) % len(heads)
                    if heads[selected_head_idx] == head_id:
                        selected_head_idx = (selected_head_idx - 1) % len(heads)
                    elif ch3 == "A":    # UP
                        selected_amount += 1
                    elif ch3 == "B":    # DOWN
                        selected_amount -= 1
                    render_status(head_id)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# =================================================
# Mutate (Proof)
# =================================================
def parse_peer(p: str) -> Tuple[str, int]:
    # "host:port" → tuple
    host, port = p.split(":")
    return host, int(port)

def next_crown(c: int) -> int:
    # 3-crown coil: JADE→ONYX→OPAL
    return 1 if int(c) >= 3 else int(c) + 1

def state_hash(tallies: Dict[str, int], crown: int) -> str:
    obj = {
        "tallies": dict(tallies),
        "crown": int(crown),
    }
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:12]

def collapse_key(tail: Dict[str, Any]) -> Tuple[int, str, str]:
    return (int(tail.get("weight", 0)), str(tail.get("id", "")), str(tail.get("h", "")))


def emit_tail(
    head_id: str,
    tallies_snapshot: Dict[str, int],
    crown_snapshot: int,
    tx: Tuple[str, str, int]
) -> Dict[str, Any]:
    frm, to, amt = tx
    tallies = dict(tallies_snapshot)
    tallies[frm] = tallies.get(frm, 0) - amt
    tallies[to] = tallies.get(to, 0) + amt

    crown = next_crown(int(crown_snapshot))
    h = state_hash(tallies, crown)

    return {
        "id": head_id,
        "tallies": tallies,
        "crown": crown,
        "weight": int(h, 16),
        "h": h,
    }

def snapshot_tail(head_id: str) -> Dict[str, Any]:
    # Mirror: current truth snapshot
    tail = {
        "id": TAIL_STATE.get("head") or head_id,
        "tallies": dict(TAIL_STATE.get("tallies", {})),
        "crown": int(TAIL_STATE.get("crown", 1)),
        "weight": 0,
    }
    tail["h"] = state_hash(tail["tallies"], tail["crown"])
    return tail

# =================================================
# Hunger (Sync)
# =================================================
def hunger(
    sock: socket.socket,
    peers: List[Tuple[str, int]],
    my_crown: int,
    head_id: str,
    need_tail: bool=False
) -> None:
    msg = {"type": "HUNGER", "id": head_id, "crown": int(my_crown), "need_tail": bool(need_tail)}
    payload = json.dumps(msg).encode("utf-8")
    for host, port in peers:
        try:
            sock.sendto(payload, (host, port))
        except Exception:
            pass

# =================================================
# Constriction — The Three-Crown Coil
# =================================================
# Twist → Hold → Coil → Tighten → Ascend → Exalt → Settle
def constrict(
    tail_in: Dict[str, Any],
    sock: socket.socket,
    peers: List[Tuple[str, int]],
    lock: threading.Lock,
    head_id: str,
    src_addr: Union[Tuple[str, int], None] = None
) -> None:
    global TAIL, ENVY

    incoming_tallies = tail_in.get("tallies", {})
    incoming_crown = int(tail_in.get("crown", 1))

    deferred_envy_render = False
    deferred_hunger_crown = None

    # --- Twist: normalize contenders
    computed_h = state_hash(incoming_tallies, incoming_crown)
    incoming_h = tail_in.get("h")
    if incoming_h != computed_h:
        tail_in["h"] = computed_h
        tail_in["weight"] = int(computed_h, 16)
        incoming_h = computed_h


    # Ensure weight is always derived from h
    try:
        tail_in["weight"] = int(str(tail_in.get("h", computed_h)), 16)
    except Exception:
        tail_in["weight"] = int(computed_h, 16)

    with lock:
        # --- Hold: detect stalemate
        if incoming_h and incoming_h == TAIL_STATE.get("h"):
            return

        current_crown = int(TAIL_STATE.get("crown", 1))
        expected = next_crown(current_crown)

        # --- Coil: align the coil
        if TAIL_STATE.get("h") is None:
            current_crown = incoming_crown
            expected = next_crown(current_crown)
        else:
            # --- Coil: crown gate
            if incoming_crown not in (current_crown, expected):
                ENVY = True
                # Defer terminal I/O until after releasing the lock
                deferred_envy_render = True
                deferred_hunger_crown = current_crown
                return

        # --- Tighten: apply dominance pressure
        should_adopt = False
        if incoming_crown == expected:
            should_adopt = True
        elif incoming_crown == current_crown:
            my_tail = TAIL or snapshot_tail(head_id)
            should_adopt = collapse_key(tail_in) > collapse_key(my_tail)

        if not should_adopt:
            return

        # --- Ascend: seize the crown
        advanced = (incoming_crown == expected)

        # --- Exalt: recognize supremacy
        TAIL_STATE["tallies"] = dict(tail_in["tallies"])
        TAIL_STATE["crown"] = incoming_crown
        TAIL_STATE["h"] = incoming_h
        TAIL_STATE["head"] = tail_in.get("id")
        TAIL = tail_in
        TAIL["h"] = TAIL_STATE["h"]
        
        # Clear envy if we adopted from peer
    if deferred_envy_render:
        render_status(head_id)
        hunger(sock, peers, int(deferred_hunger_crown or TAIL_STATE.get("crown", 1)), head_id, need_tail=True)
        return

        if ENVY and tail_in.get("id") != head_id:
            ENVY = False

    # --- Settle: propagate crown
    render_status(head_id)

    payload = json.dumps(tail_in).encode("utf-8")
    for host, port in peers:
        # --- minor storm control
        if src_addr is not None and (host, port) == (src_addr[0], src_addr[1]):
            continue
        try:
            sock.sendto(payload, (host, port))
        except Exception:
            pass

    # --- Reflex: hunger only on advance
    if advanced:
        hunger(sock, peers, incoming_crown, head_id)

# =================================================
# Nerves (UDP Receiver Thread / Transport Wiring)
# =================================================
class Receiver(threading.Thread):
    def __init__(self, sock: socket.socket, lock: threading.Lock, head_id: str, peers: List[Tuple[str, int]]):
        super().__init__(daemon=True)
        self.sock = sock
        self.lock = lock
        self.head_id = head_id
        self.peers = peers

    def run(self) -> None:
        while True:
            try:
                data, addr = self.sock.recvfrom(65535)
                msg = json.loads(data.decode("utf-8"))

                # --- Nerve fiber: hunger handling
                if msg.get("type") == "HUNGER":
                    requester_id = str(msg.get("id", "?"))
                    requester_crown = int(msg.get("crown", 1))
                    need_tail = bool(msg.get("need_tail", False))

                    with self.lock:
                        my_crown = int(TAIL_STATE.get("crown", 1))
                        tail_out = TAIL or snapshot_tail(self.head_id)
                        my_h = str(tail_out.get("h", ""))

                    requester_is_behind = (next_crown(requester_crown) == my_crown)

                    if need_tail or requester_is_behind:
                        key = (requester_id, requester_crown)
                        if (not need_tail) and HUNGER_REPLY_LAST.get(key) == my_h:
                            continue
                        HUNGER_REPLY_LAST[key] = my_h
                        self.sock.sendto(json.dumps(tail_out).encode("utf-8"), addr)
                    continue

                # --- Nerve fiber: reject malformed proofs
                if not all(k in msg for k in ("id", "tallies", "crown", "weight", "h")):
                    continue

                # --- Nerve fiber: normalize hash
                try:
                    computed_h = state_hash(msg.get("tallies", {}), int(msg.get("crown", 1)))
                except Exception:
                    continue
                msg["h"] = computed_h


                # Normalize weight to the hash-derived
                msg["weight"] = int(computed_h, 16)
                # --- Storm Membrane
                with self.lock:
                    if computed_h in SEEN_H:
                        continue
                    SEEN_H[computed_h] = None
                    if len(SEEN_H) > SEEN_MAX:
                        SEEN_H.pop(next(iter(SEEN_H)))

                # --- Route into neuronal layer
                constrict(msg, self.sock, self.peers, self.lock, self.head_id, src_addr=addr)

            except Exception:
                continue

# =================================================
# Brain
# =================================================
def brain() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Head ID, e.g. A or B")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--peers", nargs="*", default=[])
    args = ap.parse_args()

    head_id = args.id.upper()

    global selected_head_idx
    try:
        me = heads.index(head_id)
        selected_head_idx = (me + 1) % len(heads)
    except ValueError:
        selected_head_idx = 0
    peers: List[Tuple[str, int]] = [parse_peer(p) for p in args.peers]

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)  # ~1 MiB
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)  # ~1 MiB
    sock.bind(("0.0.0.0", args.port))

    lock = threading.Lock()

    # Fire nerves
    Receiver(sock, lock, head_id, peers).start()

    # Open eyes
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

    render_status(head_id)

    # Initial hunger: Awake
    hunger(sock, peers, int(TAIL_STATE.get("crown", 1)), head_id, need_tail=True)

    try:
        while True:
            cmd = read_cmd_and_restore_hud(head_id)
            if cmd == "HUNGER":
                with lock:
                    my_crown = int(TAIL_STATE.get("crown", 1))
                hunger(sock, peers, my_crown, head_id, need_tail=True)
                continue
            if isinstance(cmd, tuple) and cmd[0] == "ARROW_FEED":
                _, to, amt = cmd

                # Local mutation → proof emission → neuronal constriction
                with lock:
                    base_tallies = dict(TAIL_STATE["tallies"])
                    base_crown = int(TAIL_STATE.get("crown", 1))

                tail_local = emit_tail(head_id, base_tallies, base_crown, (head_id, to, amt))
                constrict(tail_local, sock, peers, lock, head_id)

    except (KeyboardInterrupt, EOFError):
        print("\n\n\n  No Time. No Replay. No Logs.\n\n  Sniff.Snort..RAWR...bye\n\n")
    finally:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()

if __name__ == "__main__":
    brain()
