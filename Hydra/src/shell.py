# ============================================
# Shell (Body) — Truth Through Erasure
# No time. No replay. No logs.
# ============================================
from __future__ import annotations
import json, os, socket, sys, termios, threading, tty
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# ============================================
# Plexus (Heart)
# ============================================
from plexus import Intent, gem_name  # type: ignore
from typing import Protocol

class Heart(Protocol):
    head_id: str
    tail: Optional[Dict[str, Any]]
    state: Any
    def snapshot(self) -> Dict[str, Any]: ...
    def emotions(self) -> Dict[str, Any]: ...
    def ingest(self, tail_in: Dict[str, Any]) -> List[Intent]: ...
    def propose(self, to_head: str, amount: int) -> Dict[str, Any]: ...
    def dream_state(self) -> Dict[str, Any]: ...

# ============================================
# Skeleton
# ============================================
HEADS = ["A", "B", "C", "D", "E"]

# Motor Cortex
_selected_head_idx = 0
_selected_amount = 1

PRINT_LOCK = threading.Lock()

# Storm Membrane 
SEEN_H: Dict[str, None] = {}
SEEN_MAX = 4096

# Cursor / Green W/ Envy
HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"
GREEN = "\x1b[32m"
RESET = "\x1b[0m"

def parse_peer(p: str) -> Tuple[str, int]:
    host, port = p.split(":")
    return host, int(port)

def _cursor_left(n: int) -> str:
    return f"\x1b[{n}D" if n > 0 else ""

# ============================================
# Eyes for an Eye
# ============================================
def _render_status(plex: Heart, head_id: str, input_buf: str) -> None:
    snap = plex.snapshot()
    emotions = plex.emotions()

    crown = int(snap.get("crown", 1) or 1)
    tallies = snap.get("tallies", {}) or {}
    envy = bool(emotions.get("envy", False))

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
        "  Enter(FEED!)  Ctrl+X(Sever)  Ctrl+C(Cauterize)",
        "",
        "  Heads green with envy must be severed and rehydrated.",
        "",
        "",
    ]
    welcome_block = "\n".join(line.format(head=head_id) for line in welcome_lines)

    tail_line = " ".join(f"{k}{tallies.get(k, 'x')}" for k in ["A", "B", "C", "D", "E"])
    tail_part = f"[Tails:{tail_line}]>"

    head_label = HEADS[_selected_head_idx]
    input_segment = f">[{head_label}:{_selected_amount}]<"

    head_tag = f"{GREEN}Head:{head_id}{RESET}" if envy else f"Head:{head_id}"
    hud_line = f"  <[Crown:{gem_name(crown)}][{head_tag}]{input_segment}{input_buf}{tail_part}"
    render_block = welcome_block + "\n" + hud_line
    tail_len = len(tail_part)

    with PRINT_LOCK:
        sys.stdout.write("\x1b[H")
        sys.stdout.write("\x1b[0J")
        sys.stdout.write(render_block)
        sys.stdout.write(_cursor_left(tail_len))
        sys.stdout.flush()

# ============================================
# Motor Neurons 
# ============================================
def _soft_reboot() -> None:
    # Sever → Regrow Head. 
    with PRINT_LOCK:
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
    print("\n\n  Regrowing Head...")
    os.execv(sys.executable, [sys.executable] + sys.argv)

Command = Union[str, Tuple[str, str, int]]

def _read_cmd(head_id: str, plex: Heart) -> Command:
    # Motor Neurons → Raw Input → Command Impulses.
    global _selected_head_idx, _selected_amount

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    input_buf = ""
    _render_status(plex, head_id, input_buf)

    try:
        tty.setcbreak(fd)
        while True:
            ch = sys.stdin.read(1)
            if ch == "":
                raise EOFError

            # Enter = FEED!!
            if ch in ("\n", "\r"):
                to = HEADS[_selected_head_idx]
                if to == head_id:
                    _selected_head_idx = (_selected_head_idx + 1) % len(HEADS)
                    to = HEADS[_selected_head_idx]
                    _render_status(plex, head_id, input_buf)
                return ("FEED", to, _selected_amount)

            # Ctrl+C = Cauterize
            if ch == "\x03":
                raise KeyboardInterrupt

            # Ctrl+X = Sever 
            if ch == "\x18" and input_buf == "":
                _soft_reboot()

            # H / h = Hunger 
            if ch in ("h", "H") and input_buf == "":
                return "HUNGER"

            # Right, Left, Up, Down
            if ch == "\x1b":
                ch2 = sys.stdin.read(1)
                ch3 = sys.stdin.read(1)
                if ch2 == "[":
                    if ch3 == "C":  # →
                        _selected_head_idx = (_selected_head_idx + 1) % len(HEADS)
                    elif ch3 == "D":  # ←
                        _selected_head_idx = (_selected_head_idx - 1) % len(HEADS)
                    elif ch3 == "A":  # ↑
                        _selected_amount += 1
                    elif ch3 == "B":  # ↓
                        _selected_amount -= 1

                    if HEADS[_selected_head_idx] == head_id:
                        if ch3 in ("C", "A"):
                            _selected_head_idx = (_selected_head_idx + 1) % len(HEADS)
                        else:
                            _selected_head_idx = (_selected_head_idx - 1) % len(HEADS)

                    _render_status(plex, head_id, input_buf)

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

# ============================================
# Ichor + Neuronal
# ============================================
@dataclass
class Body:
    head_id: str
    sock: socket.socket
    peers: List[Tuple[str, int]]
    plex: Heart
    lock: threading.Lock

    def send_tail(self, tail: Dict[str, Any], src_addr: Optional[Tuple[str, int]] = None) -> None:
        payload = json.dumps(tail, separators=(",", ":")).encode("utf-8")
        for host, port in self.peers:
            if src_addr is not None and (host, port) == (src_addr[0], src_addr[1]):
                continue
            try:
                self.sock.sendto(payload, (host, port))
            except Exception:
                pass

    def send_hunger(self, crown: int, need_tail: bool) -> None:
        msg = {"type": "HUNGER", "id": self.head_id, "crown": int(crown), "need_tail": bool(need_tail)}
        payload = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        for host, port in self.peers:
            try:
                self.sock.sendto(payload, (host, port))
            except Exception:
                pass

    def execute_intents(self, intents: List[Intent], src_addr: Optional[Tuple[str, int]] = None) -> None:
        # Neuronal efferents → Plexus intents → Shell actions.#
        for it in intents:
            if it.type == "PROPAGATE":
                tail = dict(it.payload.get("tail", {}))
                if tail:
                    self.send_tail(tail, src_addr=src_addr)
            elif it.type == "REQUEST_SYNC":
                crown = int(it.payload.get("crown", 1) or 1)
                need_tail = bool(it.payload.get("need_tail", False))

                # Avoid "dream flicker": only request hunger when explicitly needed.
                # - If the heart is envious, hunger is the rehydration path.
                # - Otherwise, ignore routine sync pings that can pull stale dreams.
                if need_tail or bool(getattr(self.plex, "envy", False)):
                    self.send_hunger(crown, need_tail=True)
            elif it.type == "ENVY":
                # → emotion is held in the heart ←
                pass

        # Refresh Eyes →
        _render_status(self.plex, self.head_id, "")

class Receiver(threading.Thread):
    # Peripheral nerves → Stimulus → Storm Membrane → Heart → Intents → Actions

    def __init__(self, body: Body):
        super().__init__(daemon=True)
        self.body = body

    def _handle_hunger(self, msg: Dict[str, Any], addr: Tuple[str, int]) -> None:
        with self.body.lock:
            tail_out = dict(self.body.plex.dream_state())
        try:
            self.body.sock.sendto(json.dumps(tail_out, separators=(",", ":")).encode("utf-8"), addr)
        except Exception:
            pass

    def run(self) -> None:
        while True:
            try:
                data, addr = self.body.sock.recvfrom(65535)
                msg = json.loads(data.decode("utf-8"))

                if isinstance(msg, dict) and msg.get("type") == "HUNGER":
                    self._handle_hunger(msg, addr)
                    continue

                if not isinstance(msg, dict):
                    continue

                # sanity
                if "tallies" not in msg or "crown" not in msg:
                    continue

# ============================================
# Storm membrane (Skin)
# ============================================
                try:
                    seen_key = json.dumps(
                        {
                            "id": msg.get("id", ""),
                            "crown": int(msg.get("crown", 1) or 1),
                            "tallies": dict(msg.get("tallies", {}) or {}),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                except Exception:
                    continue

                with self.body.lock:
                    # If I'm envious, allow repeated dream hydration (no dedupe).
                    if msg.get("is_dream") and self.body.plex.envy:
                        pass
                    else:
                        if seen_key in SEEN_H:
                            continue
                    SEEN_H[seen_key] = None
                    if len(SEEN_H) > SEEN_MAX:
                        SEEN_H.pop(next(iter(SEEN_H)))

# ============================================
# Basal Ganglia (Nuclei)
# ============================================
                with self.body.lock:
                    intents = self.body.plex.ingest(dict(msg))

                self.body.execute_intents(intents, src_addr=addr)

            except Exception:
                continue

# ============================================
# Ribcage
# ============================================
def run_body(
    *,
    heart: Heart,
    head_id: str,
    port: int,
    peers: List[Tuple[str, int]],
) -> None:
    # Run the Shell with → Plexus heart

    head_id = str(head_id).upper()

    global _selected_head_idx
    try:
        me = HEADS.index(head_id)
        _selected_head_idx = (me + 1) % len(HEADS)
    except ValueError:
        _selected_head_idx = 0

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    sock.bind(("0.0.0.0", int(port)))

    lock = threading.Lock()
    body = Body(head_id=head_id, sock=sock, peers=list(peers), plex=heart, lock=lock)

    # Fire nerves →
    Receiver(body).start()

    # Open eyes →
    with PRINT_LOCK:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()

    _render_status(heart, head_id, "")

    # Awake → initial hunger →
    with lock:
        my_crown = int(getattr(heart.state, "crown", 1) or 1)
    body.send_hunger(my_crown, need_tail=True)

    try:
        while True:
            cmd = _read_cmd(head_id, heart)

            if cmd == "HUNGER":
                with lock:
                    my_crown = int(getattr(heart.state, "crown", 1) or 1)
                body.send_hunger(my_crown, need_tail=True)
                continue

            if isinstance(cmd, tuple) and cmd[0] == "FEED":
                _, to, amt = cmd

                with lock:
                    tail_local = heart.propose(to, amt)
                    intents = heart.ingest(dict(tail_local))

                body.execute_intents(intents)

    except (KeyboardInterrupt, EOFError):
        print("\n\n\n  No Time. No Replay. No Logs.\n\n  Sniff.Snort..RAWR...bye\n\n")
    finally:
        with PRINT_LOCK:
            sys.stdout.write(SHOW_CURSOR)
            sys.stdout.flush()
