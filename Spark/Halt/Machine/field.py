import os, platform, random, shlex, shutil, signal, subprocess, sys, termios, tty
from wcwidth import wcswidth


def paint():
    sys.stdout.write("\033[40m\033[97m\033[2J\033[H\033[?25l")


def clear():
    sys.stdout.write("\033[?25h\033[0m\033[2J\033[H"); sys.stdout.flush()


def center(text):
    return " " * max(0, (os.get_terminal_size().columns - wcswidth(text)) // 2) + text


def choose(label, lo, hi, value):
    old = termios.tcgetattr(sys.stdin); tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            paint(); print("\n\n" + center("HaltingMachine"), "\n", center(label), "", center(str(value)), sep="\n", flush=True)
            key = os.read(sys.stdin.fileno(), 3)
            if key in (b"\r", b"\n"): return value
            if key == b"\x1b[A": value = min(hi, value + 1)
            elif key == b"\x1b[B": value = max(lo, value - 1)
    finally: termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)


def terminal_command(cmd):
    names = ("kitty", "konsole", "gnome-terminal", "xfce4-terminal", "alacritty")
    terminal = next((shutil.which(x) for x in names if shutil.which(x)), None)
    terminal = terminal or shutil.which(os.environ.get("TERMINAL", ""))
    terminal = terminal or shutil.which("x-terminal-emulator") or shutil.which("xterm")
    if not terminal: raise SystemExit("No terminal emulator found")
    name = os.path.basename(os.path.realpath(terminal))
    if name == "gnome-terminal": return [terminal, "--", *cmd]
    if name == "xfce4-terminal": return [terminal, "--command", shlex.join(cmd)]
    if name == "alacritty": return [terminal, "-T", "HaltingMachine", "-e", *cmd]
    if name == "kitty": return [terminal, "--title", "HaltingMachine", *cmd]
    return [terminal, "-e", *cmd]


def spawn(module, index, total):
    cmd = [sys.executable, "-m", module, str(index), str(total)]
    env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
    if platform.system() == "Darwin":
        line = shlex.join(cmd).replace('"', '\\"')
        subprocess.Popen(["osascript", "-e", f'tell application "Terminal" to do script "{line}"'])
    else: subprocess.Popen(terminal_command(cmd), env=env)


def main():
    old_handlers = {s: signal.getsignal(s) for s in (signal.SIGTSTP, signal.SIGQUIT)}
    for s in old_handlers: signal.signal(s, signal.SIG_IGN)
    try:
        total = choose("OBSERVERS", 1, 10, 5)
        inverted = choose("INVERTERS", 0, total, min(1, total))
        first = total - inverted
        index = random.randrange(total)
        for i in range(total):
            if i != index:
                spawn("Machine.inverter" if i >= first else "Machine.observer", i, total)
        if index >= first:
            from .inverter import run
        else:
            from .observer import run
        run(index, total)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for s, h in old_handlers.items(): signal.signal(s, h)
        clear()


if __name__ == "__main__": main()
