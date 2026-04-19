from __future__ import annotations
import json
import socket
import sys
import termios
import tty
from select import select
from typing import Dict, List, Optional, Set, Tuple

from .Plexus import Plexus
from .Body import ExitSignal, RunBody
from .Pulse import AwakeField, Clear, ExitLine, HideCursor, Index, Now, Phase, ReadKey, RenderCentered, RenderField, ShowCursor

BaseHeads = ["A", "B", "C", "D", "E"]
Fields = ["environment", "depth", "mutation", "head", "awakening"]
Labels = {
    "environment": "ENVIRONMENT",
    "depth": "DEPTH",
    "mutation": "MUTATIONS",
    "head": "HEADS",
    "awakening": "AWAKENING",
}
Options = {
    "environment": ["Den", "Swamp"],
    "mutation": ["1", "2", "3", "4", "5"],
    "head": list(BaseHeads),
}
RosterCache: Set[str] = set()
AwakeInterval = 0.75


def BuildDen(heads: List[str], depth: int, head: str) -> Tuple[int, List[Tuple[str, int]]]:
    ports = {item: depth + index for index, item in enumerate(heads)}
    return ports[head], [("127.0.0.1", ports[item]) for item in heads if item != head]


def BuildSwamp(heads: List[str], depth: int, head: str) -> Tuple[int, List[Tuple[str, int]]]:
    return depth, [("255.255.255.255", depth)]


def ExitScreen() -> None:
    filedescriptor = sys.stdin.fileno()
    original = termios.tcgetattr(filedescriptor)
    tty.setcbreak(filedescriptor)
    start = Now()
    lastpulse = None

    try:
        sys.stdout.write(HideCursor)
        Clear()
        while True:
            phase = Phase(start)
            pulse = Index(phase, 4)
            if pulse != lastpulse:
                RenderCentered([ExitLine(phase)], bias=0.5)
                lastpulse = pulse
            try:
                ready, _, _ = select([sys.stdin], [], [], 1 / 60)
            except KeyboardInterrupt:
                return
            if ready:
                try:
                    ReadKey()
                except KeyboardInterrupt:
                    pass
                return
    finally:
        termios.tcsetattr(filedescriptor, termios.TCSADRAIN, original)
        Clear()
        sys.stdout.write(ShowCursor)
        sys.stdout.flush()


def MutateShell() -> Dict[str, str]:
    state = {"environment": "Den", "depth": "12321", "mutation": "1", "head": "A"}
    fieldstep = 0
    awakensock: Optional[socket.socket] = None
    awakeheads: Set[str] = set()
    expectedheads: Set[str] = set()
    awakevalue = ""
    awakenready = False
    lastawakesent = 0.0

    def CurrentHeads() -> List[str]:
        return BaseHeads[:int(state["mutation"])]

    def ClampHead() -> None:
        heads = CurrentHeads()
        if state["head"] not in heads:
            state["head"] = heads[0]

    def Network() -> Tuple[int, List[Tuple[str, int]]]:
        heads = CurrentHeads()
        depth = int(state["depth"])
        return BuildDen(heads, depth, state["head"]) if state["environment"] == "Den" else BuildSwamp(heads, depth, state["head"])

    def AwakeSend() -> None:
        if awakensock is None:
            return
        message = json.dumps({"type": "AWAKE", "head": state["head"], "heads": CurrentHeads()}, separators=(",", ":")).encode("utf-8")
        _, peers = Network()
        for host, peerport in peers:
            try:
                awakensock.sendto(message, (host, peerport))
            except Exception:
                pass

    def OpenAwakening() -> None:
        nonlocal awakensock, awakeheads, expectedheads, awakevalue, awakenready, lastawakesent
        global RosterCache
        if awakensock is not None:
            return
        port, _ = Network()
        awakensock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        awakensock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if state["environment"] == "Swamp":
            awakensock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            awakensock.bind(("0.0.0.0", port))
        except OSError as exc:
            try:
                awakensock.close()
            except Exception:
                pass
            awakensock = None
            raise ExitSignal() from exc
        awakensock.setblocking(False)
        expectedheads = set(CurrentHeads())
        awakeheads = {item for item in RosterCache if item in expectedheads}
        awakeheads.add(state["head"])
        RosterCache = set(awakeheads)
        awakevalue = AwakeField(awakeheads)
        awakenready = False
        lastawakesent = 0.0
        for _ in range(3):
            AwakeSend()
        lastawakesent = Now()

    def CloseAwakening() -> None:
        nonlocal awakensock, awakeheads, expectedheads, awakevalue, awakenready, lastawakesent
        global RosterCache
        if awakeheads:
            RosterCache = set(awakeheads)
        if awakensock is not None:
            try:
                awakensock.close()
            except Exception:
                pass
        awakensock = None
        awakeheads = set()
        expectedheads = set()
        awakevalue = ""
        awakenready = False
        lastawakesent = 0.0

    def PollAwakening() -> None:
        nonlocal awakevalue, awakenready, lastawakesent, expectedheads, awakeheads
        if awakensock is None:
            return
        now = Now()
        if not awakenready and now - lastawakesent >= AwakeInterval:
            AwakeSend()
            lastawakesent = now
        while True:
            try:
                data, _ = awakensock.recvfrom(1024)
            except BlockingIOError:
                break
            except Exception:
                break
            try:
                message = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if not isinstance(message, dict):
                continue
            incoming = str(message.get("head", "") or "").strip().upper()
            incomingheads = {str(item).upper() for item in list(message.get("heads", []) or []) if str(item).strip()}
            messagetype = str(message.get("type", "") or "")
            if messagetype == "AWAKE":
                if incomingheads:
                    expectedheads = set(incomingheads)
                if incoming and incoming not in awakeheads:
                    awakeheads.add(incoming)
                    AwakeSend()
                    lastawakesent = Now()
                continue
            if messagetype == "ROSTER":
                if incomingheads:
                    expectedheads = set(incomingheads)
                    awakeheads.update(incomingheads)
                if incoming:
                    awakeheads.add(incoming)
        awakevalue = AwakeField(awakeheads)
        awakenready = bool(expectedheads) and awakeheads >= expectedheads

    filedescriptor = sys.stdin.fileno()
    original = termios.tcgetattr(filedescriptor)
    tty.setcbreak(filedescriptor)
    start = Now()
    lastpulse = None
    lastfield = None
    lastvalue = None

    try:
        sys.stdout.write(HideCursor)
        Clear()
        while True:
            field = Fields[fieldstep]
            if field == "awakening":
                OpenAwakening()
                PollAwakening()
            phase = Phase(start)
            value = awakevalue if field == "awakening" else state[field]
            pulse = Index(phase, 9)
            if pulse != lastpulse or field != lastfield or value != lastvalue:
                RenderField("Mutate", Labels[field], value, phase)
                lastpulse = pulse
                lastfield = field
                lastvalue = value
            if field == "awakening" and awakenready:
                CloseAwakening()
                return state
            try:
                ready, _, _ = select([sys.stdin], [], [], 1 / 60)
            except KeyboardInterrupt:
                raise ExitSignal
            if not ready:
                continue
            try:
                key = ReadKey()
            except KeyboardInterrupt:
                raise ExitSignal
            field = Fields[fieldstep]
            if key == "\x03" or field == "awakening":
                raise ExitSignal
            if key in ("\n", "\r", "C"):
                fieldstep = min(len(Fields) - 1, fieldstep + 1)
                continue
            if key == "D":
                fieldstep = max(0, fieldstep - 1)
                continue
            if key in ("A", "B"):
                direction = 1 if key == "A" else -1
                if field == "depth":
                    state["depth"] = f"{max(0, min(99999, int(state['depth']) + direction)):05d}"
                    continue
                fieldoptions = Options.get(field)
                if fieldoptions is not None:
                    index = fieldoptions.index(state[field])
                    if field == "head":
                        direction = -direction
                    state[field] = fieldoptions[(index + direction) % len(fieldoptions)]
                    if field == "mutation":
                        ClampHead()
                continue
            if not key.isprintable():
                continue
            if field == "mutation" and key in "12345":
                state["mutation"] = key
                ClampHead()
            elif field == "head":
                typed = key.upper()
                if typed in BaseHeads:
                    state["head"] = typed
                    ClampHead()
    finally:
        CloseAwakening()
        termios.tcsetattr(filedescriptor, termios.TCSADRAIN, original)
        Clear()
        sys.stdout.write(ShowCursor)
        sys.stdout.flush()


def Mutate() -> None:
    try:
        state = MutateShell()
        heads = BaseHeads[:int(state["mutation"])]
        head = state["head"] if state["head"] in heads else heads[0]
        depth = int(state["depth"])
        port, peers = BuildDen(heads, depth, head) if state["environment"] == "Den" else BuildSwamp(heads, depth, head)
        RunBody(heart=Plexus(head=head, heads=heads), head=head, port=port, peers=peers, heads=heads)
    except ExitSignal:
        ExitScreen()


if __name__ == "__main__":
    Mutate()
