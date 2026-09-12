from __future__ import annotations
import socket
import sys
import termios
import threading
import time
import tty
from select import select
from typing import Any, Dict, List, Optional, Tuple

from Game.Body import Body, ExitSignal, FirstTarget, PrintLock, ReadCommand, Receiver
from Game.Mutate import BaseHeads, BuildDen, BuildSwamp, ExitScreen, MutateShell, SetupBack
from Game.Plexus import Plexus
from Game.Pulse import (
    Ash,
    BubbleLine,
    Clear,
    Flicker3,
    Flicker4,
    Green,
    HideCursor,
    Index,
    Now,
    Phase,
    ReadKey,
    Reset,
    ShowCursor,
    Teal,
    TerminalSize,
    TitleLine,
    VerticalOffset,
    VisibleLength,
)

Targets = (999, 99999)
DefaultTarget = 999


class RaceBody(Body):
    race: int = DefaultTarget
    winner: Optional[str] = None

    def AcceptMessage(self, message: Dict[str, Any]) -> bool:
        try:
            return int(message.get("race", -1)) == int(self.race)
        except Exception:
            return False

    def SendMessage(self, message: Dict[str, Any], dstaddr: Optional[Tuple[str, int]] = None, skipaddr: Optional[Tuple[str, int]] = None) -> None:
        tagged = dict(message)
        tagged["race"] = int(self.race)
        super().SendMessage(tagged, dstaddr=dstaddr, skipaddr=skipaddr)

    def VictoryLap(self) -> None:
        with self.lock:
            snapshot = dict(self.heart.Snapshot())
            tail = dict(self.heart.tail or {})
            same = (
                tail
                and dict(tail.get("tallies", {}) or {}) == dict(snapshot.get("tallies", {}) or {})
                and int(tail.get("crown", 0) or 0) == int(snapshot.get("crown", 0) or 0)
            )
            package = tail if same else dict(self.heart.DreamState())
        self.SendTail(package)

    def RenderStatus(self) -> None:
        with self.lock:
            if not self.running:
                return
            winner = Winner(dict(self.heart.Snapshot()), self.race)
            if winner:
                self.winner = winner
                self.running = False
        if winner:
            self.VictoryLap()
            return
        super().RenderStatus()


def RunRace(*, heart: Plexus, head: str, port: int, peers: List[Tuple[str, int]], heads: List[str], race: int) -> Optional[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    if any(host == "255.255.255.255" for host, _ in peers):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("0.0.0.0", int(port)))
    sock.settimeout(0.1)
    body = RaceBody(head=str(head).upper(), heads=list(heads), sock=sock, peers=list(peers), heart=heart, lock=threading.Lock(), targethead=FirstTarget(head, heads))
    body.race = int(race)
    receiver = Receiver(body)
    receiver.start()
    with PrintLock:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(HideCursor)
        sys.stdout.flush()
    body.RenderStatus()
    with body.lock:
        body.SendHunger(body.Crown(), needtail=True)
    try:
        while body.running:
            command = ReadCommand(body)
            if not body.running:
                break
            if command == "HUNGER":
                with body.lock:
                    body.SendHunger(body.Crown(), needtail=True)
                continue
            if isinstance(command, tuple) and command[0] == "FEED":
                _, tohead, amount = command
                with body.lock:
                    intents = heart.Ingest(dict(heart.Propose(tohead, amount)))
                body.ExecuteIntents(intents)
    except (KeyboardInterrupt, EOFError) as exc:
        raise ExitSignal() from exc
    finally:
        body.running = False
        try:
            sock.close()
        except Exception:
            pass
        receiver.join(timeout=0.25)
        with PrintLock:
            sys.stdout.write(ShowCursor)
            sys.stdout.flush()
    return body.winner


def Card(
    lines: list[str],
    offsets: Optional[dict[int, int]] = None,
    *,
    bias: float = 0.35,
) -> None:
    # Keep MacAttack cards as one tight block.  Unlike the generic renderer,
    # do not right-pad each row to 80 columns: on an 80-column terminal that
    # can trigger an automatic wrap before the explicit newline and visually
    # insert a blank row between every line.
    linelist = list(lines)
    offsets = {} if offsets is None else dict(offsets)
    _, terminalheight = TerminalSize()
    height = max(24, terminalheight)
    topgap = VerticalOffset(len(linelist), height, bias)
    sys.stdout.write("\x1b[H")
    if topgap:
        sys.stdout.write("\r\n" * topgap)
    for index, line in enumerate(linelist):
        gap = max(0, ((80 - VisibleLength(line)) // 2) + int(offsets.get(index, 0)))
        sys.stdout.write((" " * gap) + line + "\x1b[K")
        if index + 1 < len(linelist):
            # VictoryScreen uses tty raw mode so Ctrl-C/Escape can be ignored.
            # In raw mode LF does not imply carriage return; use CRLF explicitly
            # or each centered line starts where the previous one ended.
            sys.stdout.write("\r\n")
    sys.stdout.flush()


def TargetLine(target: int) -> str:
    number = str(int(target))
    # MacAttack deliberately offers only odd-width tail targets: 999 / 99999.
    # Both glyphs therefore share the exact center axis with the five-letter TAILS label.
    return f"{Green}.{Reset}{Teal}{number}{Reset}{Green}.{Reset}"


def IntroScreen(target: int = DefaultTarget) -> int:
    target = int(target) if int(target) in Targets else DefaultTarget
    filedescriptor = sys.stdin.fileno()
    original = termios.tcgetattr(filedescriptor)
    tty.setcbreak(filedescriptor)
    start = Now()
    lastpulse: Optional[int] = None
    lasttarget: Optional[int] = None
    try:
        sys.stdout.write(HideCursor)
        Clear()
        while True:
            phase = Phase(start)
            pulse = Index(phase, 9)
            if pulse != lastpulse or target != lasttarget:
                Card([
                    TitleLine("Hydra"),
                    TitleLine("MacAttack"),
                    BubbleLine(phase),
                    f"{Ash}TAILS{Reset}",
                    TargetLine(target),
                ])
                lastpulse = pulse
                lasttarget = target
            try:
                ready, _, _ = select([sys.stdin], [], [], 1 / 60)
            except KeyboardInterrupt:
                raise ExitSignal
            if not ready:
                continue
            key = ReadKey()
            if key == "\x03":
                raise ExitSignal
            if key in ("A", "B"):
                direction = 1 if key == "A" else -1
                target = Targets[(Targets.index(target) + direction) % len(Targets)]
                continue
            if key in ("\n", "\r", " ", "C"):
                return target
            # Left and every other key stay on the MacAttack card.
    finally:
        termios.tcsetattr(filedescriptor, termios.TCSADRAIN, original)
        Clear()
        sys.stdout.write(ShowCursor)
        sys.stdout.flush()


def ShimmerWord(text: str) -> str:
    # The underwater pair: Teal<->Blue and Blue<->Teal. Deliberately avoid
    # Flicker1/Flicker2 here because those cycles include Venom green; Venom
    # belongs exclusively to the growing ellipsis on this card.
    pieces = []
    for index, character in enumerate(str(text)):
        shimmer = Flicker3 if index % 2 == 0 else Flicker4
        pieces.append(f"{shimmer()}{character}{Reset}")
    return "".join(pieces)


def SetScreen() -> None:
    # One full 0->1->2->3->2->1->0 ellipsis breath for READY, SET and GO!!!.
    # The whole line sits at the same vertical center used by Sniff.Snort..RAWR...bye.
    dotframes = (0, 1, 2, 3, 2, 1, 0)
    frameduration = 1.0 / 7.0
    sys.stdout.write(HideCursor)
    Clear()
    try:
        for word in ("READY", "SET", "GO!!!"):
            for dotcount in dotframes:
                deadline = Now() + frameduration
                while Now() < deadline:
                    dots = f"{Green}{'.' * dotcount}{Reset}"
                    Card([f"{dots}{ShimmerWord(word)}{dots}"], bias=0.5)
                    time.sleep(1 / 60)
    finally:
        Clear()
        sys.stdout.write(ShowCursor)
        sys.stdout.flush()


def Winner(snapshot: Dict[str, Any], target: int) -> Optional[str]:
    tallies = dict(snapshot.get("tallies", {}) or {})
    qualified = [
        (int(value), str(head).upper())
        for head, value in tallies.items()
        if int(value) >= int(target)
    ]
    if not qualified:
        return None
    qualified.sort(key=lambda item: (-item[0], item[1]))
    return qualified[0][1]


def VictoryScreen(winner: str, localhead: str) -> None:
    filedescriptor = sys.stdin.fileno()
    original = termios.tcgetattr(filedescriptor)
    tty.setraw(filedescriptor)
    start = Now()
    lastpulse = None
    result = "YOU WIN" if str(winner).upper() == str(localhead).upper() else f"HEAD {str(winner).upper()} WINS"
    try:
        sys.stdout.write(HideCursor)
        Clear()
        while True:
            phase = Phase(start)
            pulse = Index(phase, 9)
            if pulse != lastpulse:
                Card([
                    TitleLine("Hydra"),
                    TitleLine("MacAttack"),
                    BubbleLine(phase),
                    f"{Ash}ORDER THROUGH CHAOS{Reset}",
                    TitleLine(result),
                ])
                lastpulse = pulse
            ready, _, _ = select([sys.stdin], [], [], 1 / 60)
            if not ready:
                continue
            # Victory is deliberately modal: only Space releases into Hydra's
            # normal Sniff.Snort..RAWR...bye screen. Everything else is ignored.
            if sys.stdin.read(1) == " ":
                return
    finally:
        termios.tcsetattr(filedescriptor, termios.TCSADRAIN, original)
        Clear()
        sys.stdout.write(ShowCursor)
        sys.stdout.flush()


def MacAttack() -> None:
    target = DefaultTarget
    setupstate: Optional[Dict[str, str]] = None
    try:
        while True:
            target = IntroScreen(target)
            try:
                state = MutateShell(race=target, initial=setupstate, back_to_race=True)
                break
            except SetupBack as back:
                setupstate = dict(back.state)
                continue
        heads = BaseHeads[:int(state["mutation"])]
        head = state["head"] if state["head"] in heads else heads[0]
        depth = int(state["depth"])
        port, peers = BuildDen(heads, depth, head) if state["environment"] == "Den" else BuildSwamp(heads, depth, head)
        SetScreen()
        try:
            winner = RunRace(
                heart=Plexus(head=head, heads=heads),
                head=head,
                port=port,
                peers=peers,
                heads=heads,
                race=target,
            )
            if winner:
                VictoryScreen(winner, head)
                ExitScreen()
        except OSError:
            ExitScreen()
    except (ExitSignal, KeyboardInterrupt):
        ExitScreen()


if __name__ == "__main__":
    MacAttack()
