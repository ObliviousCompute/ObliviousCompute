import os, select, signal, socket, sys, termios, time, tty
from wcwidth import wcswidth

ROCK, PAPER, SCISSORS = "🪨", "📜", "✂️"
FORWARD = {ROCK: PAPER, PAPER: SCISSORS, SCISSORS: ROCK}
REVERSE = {v: k for k, v in FORWARD.items()}
ART = {ROCK: ROCK * 4, PAPER: PAPER * 4, SCISSORS: "✂️ " * 4}
GREEK = ("Γ", "Δ", "Ε", "Θ", "Λ", "Π", "Σ", "Φ", "Ψ", "Ω")
BASE = 43120


def paint():
    sys.stdout.write("\033[40m\033[97m\033[2J\033[H\033[?25l")


def clear():
    sys.stdout.write("\033[?25h\033[0m\033[2J\033[H"); sys.stdout.flush()


def center(text, width=None):
    width = wcswidth(text) if width is None else width
    return " " * max(0, (os.get_terminal_size().columns - width) // 2) + text


def draw(identity, mode, state, inverted=False, genesis=None):
    paint(); print("\n\n" + center(f"{identity} HaltingMachine {identity}"), "\n")
    if genesis is not None:
        print(center("GENESIS"), "\n", center(genesis), flush=True); return
    face = "🙃" if inverted else "🙂"
    halt = f"{face}HALT{face}" if mode == 0 else "HALT"
    repeat = f"{face}REPEAT{face}" if mode else "REPEAT"
    print(center(halt), center(repeat), "", center(ART[state], 8), "", center("SPACE TO EXIT"), sep="\n", flush=True)


def send(sock, total, payload, skip):
    data = payload.encode("utf-8")
    for i in range(total):
        if i != skip:
            try: sock.sendto(data, ("127.0.0.1", BASE + i))
            except OSError: pass


def run(index, total, step=FORWARD, invert_controls=False, invert_mode=False):
    identity = GREEK[index]
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setblocking(False); sock.bind(("127.0.0.1", BASE + index))
    seen, state, mode = {index}, ROCK, 0
    old = termios.tcgetattr(sys.stdin); tty.setcbreak(sys.stdin.fileno())
    ignored = (signal.SIGTSTP, signal.SIGQUIT)
    handlers = {s: signal.getsignal(s) for s in ignored}
    for s in ignored: signal.signal(s, signal.SIG_IGN)
    hello_at = ready_at = emit_at = 0.0; shown_seen = -1
    try:
        while True:
            now = time.monotonic()
            if now >= hello_at:
                send(sock, total, f"H{index}", index); hello_at = now + .25
            while True:
                try: msg = sock.recvfrom(64)[0].decode("utf-8")
                except BlockingIOError: break
                if msg.startswith("H"):
                    seen.add(int(msg[1:]))
                elif now >= ready_at:
                    # ================= LINCHPIN ================= #
                    admissible = msg == step[state]
                    # ============================================ #
                    if admissible:
                        state = msg; ready_at = emit_at = now + 1.0
                        send(sock, total, msg, index)
                        draw(identity, mode, state, invert_controls)
            if len(seen) < total:
                if len(seen) != shown_seen:
                    draw(identity, mode, state, invert_controls, f"{len(seen)} / {total}"); shown_seen = len(seen)
                time.sleep(.05); continue
            if ready_at == 0.0:
                draw(identity, mode, state, invert_controls); ready_at = emit_at = now + 1.0
            active = (1 - mode) if invert_mode else mode
            if active and now >= ready_at and now >= emit_at:
                send(sock, total, step[state], index); emit_at = now + 1.0
            ready, _, _ = select.select([sys.stdin], [], [], .05)
            if ready:
                key = os.read(sys.stdin.fileno(), 3)
                if key == b" ": break
                if key in (b"\x1b[A", b"\x1b[B"):
                    down = key == b"\x1b[B"
                    mode = 1 if (not down if invert_controls else down) else 0
                    emit_at = time.monotonic() + 1.0
                    draw(identity, mode, state, invert_controls)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for s, h in handlers.items(): signal.signal(s, h)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        clear(); sock.close()


def main():
    index, total = map(int, sys.argv[1:3])
    run(index, total)


if __name__ == "__main__": main()
