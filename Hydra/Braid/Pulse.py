from __future__ import annotations
import re
import shutil
import sys
import time
from typing import Iterable, Sequence, Set, Tuple

Reset = "\x1b[0m"
Ash = "\x1b[90m"
Blue = "\x1b[36m"
Green = "\x1b[92m"
Teal = "\x1b[38;2;0;150;130m"
HideCursor = "\x1b[?25l"
ShowCursor = "\x1b[?25h"
AnsiPattern = re.compile(r"\x1b\[[0-9;]*m")
PulseRate = 7.0

BubbleFrames = [
    "",
    "0",
    "o0o",
    ".o0o.",
    "o.o0o.o",
    "0o.o0o.o0",
    "o0o.o0o.o0o",
    ".o0o.o0o.o0o.",
    "..o0o.o0o.o0o..",
]

ExitFrames = ["", ".", "..", "..."]


def Now() -> float:
    return time.monotonic()


def Step(rate: float = PulseRate) -> int:
    return int(Now() * float(rate or 0.0))


def Phase(start: float, speed: float = PulseRate) -> float:
    return (Now() - float(start or 0.0)) * float(speed or 0.0)


def Index(phase: float, frames: int) -> int:
    if frames <= 1:
        return 0
    span = (frames * 2) - 2
    index = int(phase) % span
    if index >= frames:
        index = span - index
    return index


def FlickerCycle(sequence: Sequence[str], rate: float = PulseRate, fallback: str = Teal) -> str:
    if not sequence:
        return fallback
    return sequence[Step(rate) % len(sequence)]


def Flicker1() -> str:
    return FlickerCycle((Green, Blue, Teal))


def Flicker2() -> str:
    return FlickerCycle((Green, Teal))


def Flicker3() -> str:
    return FlickerCycle((Teal, Blue))


def Flicker4() -> str:
    return FlickerCycle((Blue, Teal))


def VisibleLength(text: str) -> int:
    return len(AnsiPattern.sub("", str(text or "")))


def PadLine(text: str, width: int) -> str:
    built = str(text or "")
    visible = VisibleLength(built)
    if visible < width:
        built += " " * (width - visible)
    return built


def Center(text: str, width: int = 80) -> str:
    line = str(text or "")
    gap = max(0, (width - VisibleLength(line)) // 2)
    built = (" " * gap) + line
    visible = VisibleLength(built)
    if visible < width:
        built += " " * (width - visible)
    return built


def TerminalSize() -> Tuple[int, int]:
    size = shutil.get_terminal_size(fallback=(80, 24))
    return size.columns, size.lines


def VerticalOffset(lines: int, height: int = 24, bias: float = 0.35) -> int:
    return max(0, int((height - lines) * bias))


def Clear() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    sys.stdout.flush()


def CursorLeft(count: int) -> str:
    return f"\x1b[{count}D" if count > 0 else ""


def ReadKey() -> str:
    key = sys.stdin.read(1)
    if key == "\x1b" and sys.stdin.read(1) in ("[", "O"):
        return sys.stdin.read(1)
    return key


def RenderCentered(lines: Iterable[str], *, width: int = 80, minimumheight: int = 24, bias: float = 0.35) -> None:
    linelist = list(lines)
    _, terminalheight = TerminalSize()
    height = max(minimumheight, terminalheight)
    topgap = VerticalOffset(len(linelist), height, bias)
    sys.stdout.write("\x1b[H")
    if topgap:
        sys.stdout.write("\n" * topgap)
    sys.stdout.write("\n".join(Center(line, width) for line in linelist))
    sys.stdout.flush()


def TitleLine(text: str) -> str:
    return f"{Green}.:{Reset}{Teal}{text}{Reset}{Green}:.{Reset}"


def LabelLine(text: str) -> str:
    return f"{Ash}{text}{Reset}"


def DotField(value: str) -> str:
    return f"{Green}.{Reset}{Teal}{value}{Reset}{Green}.{Reset}"


def AwakeField(heads: Set[str]) -> str:
    return "" if not heads else f"{Teal}" + f"{Reset}{Green}.{Reset}{Teal}".join(sorted(heads))


def RenderField(title: str, label: str, value: str, phase: float, bias: float = 0.35) -> None:
    RenderCentered([
        TitleLine(title),
        "",
        BubbleLine(phase),
        LabelLine(label),
        DotField(value),
    ], bias=bias)


def BubbleLine(phase: float) -> str:
    return f"{Ash}{BubbleFrames[Index(phase, len(BubbleFrames))]}{Reset}"


def ExitLine(phase: float) -> str:
    dots = ExitFrames[Index(phase, len(ExitFrames))]
    message = f"{Teal}Sniff{Green}.{Teal}Snort{Green}..{Teal}RAWR{Green}...{Teal}bye{Reset}"
    return f"{Green}{dots}{Reset}{Teal}{message}{Reset}{Green}{dots}{Reset}"
