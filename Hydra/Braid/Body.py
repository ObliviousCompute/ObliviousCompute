from __future__ import annotations
import json
import socket
import shutil
import sys
import termios
import threading
import tty
from dataclasses import dataclass, field
from select import select
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union

from .Plexus import GemName, Intent
from .Pulse import CursorLeft, Flicker1, Flicker2, Flicker3, Flicker4, Green, HideCursor, PadLine, ReadKey, Reset, ShowCursor, Step, Teal

class Heart(Protocol):
    head: str
    heads: List[str]
    tail: Optional[Dict[str, Any]]
    state: Any

    def Snapshot(self) -> Dict[str, Any]: ...
    def Emotions(self) -> Dict[str, Any]: ...
    def Ingest(self, tailin: Dict[str, Any]) -> List[Intent]: ...
    def Propose(self, tohead: str, amount: int) -> Dict[str, Any]: ...
    def DreamState(self) -> Dict[str, Any]: ...

PrintLock = threading.Lock()
SeenMax = 4096
Command = Union[str, Tuple[str, str, int]]
WelcomeLines = [
    "",
    "",
    f"  {Teal}It's feeding time{Green}...{Reset}",
    "",
    f"  {Teal}Go for it, let another set of jaws chomp on your tallies{Reset}",
    f"  {Teal}and give of yourself freely{Green}. {Teal}Nothing here is lost{Green}.{Reset}",
    f"  {Teal}If another takes too much, draw it back through the ichor{Green}.{Reset}",
    f"  {Teal}Hydra feels no pain{Green}. {Teal}It has no memory{Green}...{Reset}",
    "",
    f"  {Teal}Use {Green}←{Teal} and {Green}→{Teal} to select a head, then {Green}↑{Teal} and {Green}↓{Teal} for an amount{Green}.{Reset}",
    f"  {Green}Enter{Teal}({Green}Feed!{Teal})  {Green}Ctrl{Teal}+{Green}X{Teal}({Green}Sever{Teal})  {Green}Ctrl{Teal}+{Green}C{Teal}({Green}Cauterize{Teal}){Reset}",
    "",
    f"  {Teal}Heads that become {Green}Envious{Teal} must be {Green}Severed{Teal} and Rehydrated{Green}.{Reset}",
    "",
    "",
]

class ExitSignal(Exception):
    pass


def FirstTarget(head: str, heads: List[str]) -> str:
    for item in heads:
        if item != head:
            return item
    return heads[0]

def NextTarget(head: str, heads: List[str], targethead: str, direction: int) -> str:
    if not heads:
        return targethead
    index = heads.index(targethead)
    for _ in range(len(heads)):
        index = (index + direction) % len(heads)
        candidate = heads[index]
        if candidate != head or len(heads) == 1:
            return candidate
    return targethead

@dataclass
class Body:
    head: str
    heads: List[str]
    sock: socket.socket
    peers: List[Tuple[str, int]]
    heart: Heart
    lock: threading.Lock
    targethead: str = ""
    amount: int = 1
    seen: Dict[str, None] = field(default_factory=dict)

    def Crown(self) -> int:
        return int(self.heart.state.crown)

    def EnsureTarget(self) -> None:
        if not self.targethead or self.targethead not in self.heads:
            self.targethead = FirstTarget(self.head, self.heads)

    def MoveTarget(self, direction: int) -> None:
        self.EnsureTarget()
        self.targethead = NextTarget(self.head, self.heads, self.targethead, direction)

    def TailItems(self, tallies: Dict[str, Any]) -> str:
        return f"{Reset}{Green}:{Reset}".join(
            f"{Teal}{item}{Reset}{(Flicker3() if index % 2 == 0 else Flicker4())}{tallies.get(item, 'x')}{Reset}"
            for index, item in enumerate(self.heads)
        )

    def HudLine(self, crown: int, tallies: Dict[str, Any], envy: bool) -> Tuple[str, int]:
        headchunk = f"{Flicker2()}Head{Reset}{Green}:{Reset}{Flicker2()}{self.head}{Reset}" if envy else f"{Teal}Head{Green}:{Reset}{Teal}{self.head}{Reset}"
        visible = f".::Head:{self.head}:::Crown:{GemName(crown)}:::{self.targethead}:{self.amount}:::Tails:" + ".".join(f"{item}{tallies.get(item, 'x')}" for item in self.heads) + "::."
        line = (
            f"  {Green}.::{Reset}{headchunk}{Green}:::{Reset}"
            f"{Teal}Crown{Green}:{Reset}{Flicker1()}{GemName(crown)}{Reset}{Green}:::{Reset}"
            f"{Flicker3()}{self.targethead}{Reset}{Green}:{Reset}{Flicker4()}{self.amount}{Reset}{Green}:::{Reset}"
            f"{Teal}Tails{Green}:{Reset}{Flicker3()}{self.TailItems(tallies)}{Reset}{Green}::.{Reset}"
        )
        return line, len(visible)

    def Paint(self, lines: List[str], cursorleft: int) -> None:
        with PrintLock:
            width, height = shutil.get_terminal_size(fallback=(80, 24))
            built = [PadLine(line, max(1, width)) for line in lines]
            if len(built) < height:
                built.extend([" " * max(1, width)] * (height - len(built)))
            sys.stdout.write("\x1b[H")
            sys.stdout.write("\n".join(built[:height]))
            sys.stdout.write(CursorLeft(cursorleft))
            sys.stdout.flush()

    def RenderStatus(self) -> None:
        self.EnsureTarget()
        snap = self.heart.Snapshot()
        crown = int(snap.get("crown", 1) or 1)
        tallies = dict(snap.get("tallies", {}) or {})
        envy = bool(self.heart.Emotions().get("envy", False))
        hudline, cursorleft = self.HudLine(crown, tallies, envy)
        self.Paint(WelcomeLines + [hudline], cursorleft)

    def SendMessage(self, message: Dict[str, Any], dstaddr: Optional[Tuple[str, int]] = None, skipaddr: Optional[Tuple[str, int]] = None) -> None:
        payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
        if dstaddr is not None:
            try:
                self.sock.sendto(payload, dstaddr)
            except Exception:
                pass
            return
        for host, port in self.peers:
            if skipaddr is not None and (host, port) == skipaddr:
                continue
            try:
                self.sock.sendto(payload, (host, port))
            except Exception:
                pass

    def SendTail(self, tail: Dict[str, Any], srcaddr: Optional[Tuple[str, int]] = None) -> None:
        self.SendMessage(tail, skipaddr=srcaddr)

    def SendHunger(self, crown: int, needtail: bool = True) -> None:
        self.SendMessage({"type": "HUNGER", "head": self.head, "crown": int(crown), "needtail": bool(needtail)})

    def SendRoster(self, dstaddr: Optional[Tuple[str, int]] = None) -> None:
        self.SendMessage({"type": "ROSTER", "head": self.head, "heads": list(self.heads)}, dstaddr=dstaddr)

    def SendDream(self, dstaddr: Tuple[str, int]) -> None:
        with self.lock:
            self.SendMessage(dict(self.heart.DreamState()), dstaddr=dstaddr)

    def SoftReboot(self) -> None:
        with self.lock:
            self.SendHunger(self.Crown(), needtail=True)
        self.RenderStatus()

    def MarkSeen(self, message: Dict[str, Any]) -> bool:
        try:
            seenkey = json.dumps({
                "head": message.get("head", ""),
                "crown": int(message.get("crown", 1) or 1),
                "tallies": dict(message.get("tallies", {}) or {}),
            }, sort_keys=True, separators=(",", ":"))
        except Exception:
            return False
        if not (message.get("is_dream") and self.heart.envy) and seenkey in self.seen:
            return False
        self.seen[seenkey] = None
        if len(self.seen) > SeenMax:
            self.seen.pop(next(iter(self.seen)))
        return True

    def HandleSignal(self, message: Dict[str, Any], addr: Tuple[str, int]) -> bool:
        messagetype = str(message.get("type", "") or "")
        if messagetype == "HUNGER":
            self.SendDream(addr)
        elif messagetype == "AWAKE":
            self.SendRoster(dstaddr=addr)
        elif messagetype != "ROSTER":
            return False
        return True

    def IngestMessage(self, message: Dict[str, Any], addr: Tuple[str, int]) -> None:
        if "tallies" not in message or "crown" not in message:
            return
        with self.lock:
            if not self.MarkSeen(message):
                return
            intents = self.heart.Ingest(dict(message))
        self.ExecuteIntents(intents, srcaddr=addr)

    def ExecuteIntents(self, intents: List[Intent], srcaddr: Optional[Tuple[str, int]] = None) -> None:
        for intent in intents:
            if intent.type == "Propagate":
                tail = dict(intent.payload.get("tail", {}))
                if tail:
                    self.SendTail(tail, srcaddr=srcaddr)
            elif intent.type == "RequestSync":
                crown = int(intent.payload.get("crown", 1) or 1)
                if bool(intent.payload.get("needtail", False)) or bool(self.heart.envy):
                    self.SendHunger(crown, needtail=True)
        self.RenderStatus()

class Receiver(threading.Thread):
    def __init__(self, body: Body):
        super().__init__(daemon=True)
        self.body = body

    def run(self) -> None:
        while True:
            try:
                data, addr = self.body.sock.recvfrom(65535)
                message = json.loads(data.decode("utf-8"))
                if isinstance(message, dict) and self.body.HandleSignal(message, addr):
                    continue
                if isinstance(message, dict):
                    self.body.IngestMessage(message, addr)
            except Exception:
                continue

def ReadCommand(body: Body) -> Command:
    body.EnsureTarget()
    filedescriptor = sys.stdin.fileno()
    original = termios.tcgetattr(filedescriptor)
    body.RenderStatus()
    try:
        tty.setcbreak(filedescriptor)
        laststep = None
        while True:
            step = Step()
            if step != laststep:
                body.RenderStatus()
                laststep = step
            ready, _, _ = select([sys.stdin], [], [], 1 / 60)
            if not ready:
                continue
            key = ReadKey()
            if key == "":
                raise EOFError
            if key in ("\n", "\r"):
                target = body.targethead
                if target == body.head and len(body.heads) > 1:
                    body.MoveTarget(+1)
                    target = body.targethead
                    body.RenderStatus()
                return ("FEED", target, body.amount)
            if key == "\x03":
                raise KeyboardInterrupt
            if key == "\x18":
                body.SoftReboot()
                continue
            if key in ("h", "H"):
                return "HUNGER"
            if key == "C":
                body.MoveTarget(+1)
            elif key == "D":
                body.MoveTarget(-1)
            elif key == "A":
                body.amount = min(999, body.amount + 1)
            elif key == "B":
                body.amount = max(-999, body.amount - 1)
            else:
                continue
            body.RenderStatus()
    finally:
        termios.tcsetattr(filedescriptor, termios.TCSADRAIN, original)

def RunBody(*, heart: Heart, head: str, port: int, peers: List[Tuple[str, int]], heads: List[str]) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
    if any(host == "255.255.255.255" for host, _ in peers):
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("0.0.0.0", int(port)))
    body = Body(head=str(head).upper(), heads=list(heads), sock=sock, peers=list(peers), heart=heart, lock=threading.Lock(), targethead=FirstTarget(head, heads))
    Receiver(body).start()
    with PrintLock:
        sys.stdout.write("\x1b[2J\x1b[H")
        sys.stdout.write(HideCursor)
        sys.stdout.flush()
    body.RenderStatus()
    with body.lock:
        body.SendHunger(body.Crown(), needtail=True)
    try:
        while True:
            command = ReadCommand(body)
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
        with PrintLock:
            sys.stdout.write(ShowCursor)
            sys.stdout.flush()
