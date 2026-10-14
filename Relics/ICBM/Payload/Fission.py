from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import hashlib
import json
import os
import select
import socket
import sys
import termios
import time
import tty
from typing import Iterable, Optional

Width = 80
Height = 24
Frame = 1 / 60
Players = 5
FlashSeconds = 3.0
CountdownNS = 60_000_000_000

Title = "Interactive Consistency Broadcast Machine"
Countries = ("USA", "CHINA", "RUSSIA", "UK", "PAKISTAN", "ISRAEL", "FRANCE", "INDIA")
DefaultScenario = "NIGHTFALL"
DefaultSecret = "MIDNIGHT"

Reset = "\x1b[0m"
Ash = "\x1b[90m"
White = "\x1b[97m"
HideCursor = "\x1b[?25l"
ShowCursor = "\x1b[?25h"

Burst = 3
PayloadLimit = 65535
SimulationHost = "127.0.0.1"
SimulationBasePort = 19482
LivePort = 19582
LiveBroadcast = "255.255.255.255"


class Oblivion:
    """One UDP interface for Pact and Key projections; no stored field."""
    def __init__(self, scenario: str, mode: str) -> None:
        self.mode = mode
        self.mask = hashlib.sha256(scenario.encode("utf-8")).digest()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            sock.setsockopt(socket.SOL_SOCKET, option, 1 << 20)
        self.port: int | None = None
        try:
            if mode == "Simulation":
                for port in range(SimulationBasePort, SimulationBasePort + Players):
                    try:
                        sock.bind((SimulationHost, port))
                        self.port = port
                        break
                    except OSError:
                        continue
                if self.port is None:
                    raise RuntimeError("Simulation already has five occupied seats")
            else:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if hasattr(socket, "SO_REUSEPORT"):
                    try:
                        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                    except OSError:
                        pass
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                try:
                    sock.bind(("0.0.0.0", LivePort))
                except OSError as exc:
                    raise RuntimeError(f"Live port {LivePort} is unavailable") from exc
            sock.setblocking(False)
            self.sock: socket.socket | None = sock
        except Exception:
            sock.close()
            raise

    def Destinations(self) -> Iterable[tuple[str, int]]:
        if self.mode == "Simulation":
            return ((SimulationHost, port) for port in range(SimulationBasePort, SimulationBasePort + Players) if port != self.port)
        return ((LiveBroadcast, LivePort),)

    def Encode(self, payload: dict[str, object]) -> bytes:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return bytes(byte ^ self.mask[i % len(self.mask)] for i, byte in enumerate(body))

    def Decode(self, raw: bytes) -> dict[str, object]:
        body = bytes(byte ^ self.mask[i % len(self.mask)] for i, byte in enumerate(raw))
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
        return payload

    def Send(self, payload: dict[str, object]) -> None:
        if self.sock is None:
            return
        raw = self.Encode(payload)
        for host, port in self.Destinations():
            for copy in range(Burst):
                try:
                    self.sock.sendto(raw, (host, port))
                except OSError:
                    pass

    def Receive(self) -> list[dict[str, object]]:
        if self.sock is None:
            return []
        payloads: list[dict[str, object]] = []
        while True:
            try:
                raw = self.sock.recvfrom(PayloadLimit)[0]
            except (BlockingIOError, OSError):
                break
            try:
                payloads.append(self.Decode(raw))
            except (ValueError, UnicodeError, TypeError):
                continue
        return payloads

    def Close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None


def H(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def Commitment(country: str, code: str) -> str:
    return H(f"{country}\0{code}")

def ClaimOrder(claim: "Claim") -> tuple[str, str]:
    return claim.commitment, claim.player


def Visible(text: str) -> int:
    n = 0
    escape = False
    for ch in text:
        if escape:
            if ch == "m":
                escape = False
            continue
        if ch == "\x1b":
            escape = True
        else:
            n += 1
    return n


def Paint(text: str, color: str) -> str:
    return f"{color}{text}{Reset}"


def TitleText() -> str:
    return " ".join(Paint(word[0], White) + Paint(word[1:], Ash) for word in Title.split(" "))


def Center(text: str) -> str:
    gap = max(0, (Width - Visible(text)) // 2)
    return " " * gap + text


def Place(lines: list[str], row: int, text: str, *, center: bool = True) -> None:
    if 0 <= row < Height:
        lines[row] = Center(text) if center else text


def PlaceRight(lines: list[str], row: int, text: str, margin: int = 0) -> None:
    if 0 <= row < Height:
        lines[row] = " " * max(0, Width - margin - Visible(text)) + text


def ExitHint(lines: list[str]) -> None:
    PlaceRight(lines, Height - 2, Paint("Ctrl+C Exit", Ash), margin=1)


def Render(lines: list[str]) -> None:
    output: list[str] = []
    for line in lines[:Height]:
        pad = max(0, Width - Visible(line))
        output.append(line + " " * pad)
    while len(output) < Height:
        output.append(" " * Width)
    sys.stdout.write("\x1b[H" + "\n".join(output))
    sys.stdout.flush()


def ReadInput() -> str:
    fd = sys.stdin.fileno()
    key = os.read(fd, 1)
    if key != b"\x1b":
        return key.decode("utf-8", "ignore")
    if not select.select([fd], [], [], 0.10)[0]:
        return "ESC"
    second = os.read(fd, 1)
    if second not in (b"[", b"O") or not select.select([fd], [], [], 0.10)[0]:
        return ""
    third = os.read(fd, 1)
    key = {b"A": "UP", b"B": "DOWN", b"C": "RIGHT", b"D": "LEFT"}.get(third, "")
    if not key:
        while select.select([fd], [], [], 0)[0]:
            os.read(fd, 1)
    return key


def LockedFrame() -> None:
    ready = select.select([sys.stdin], [], [], Frame)[0]
    if ready:
        ReadInput()


def FlashWhite(elapsed: float) -> bool:
    return int(max(0.0, elapsed)) % 2 == 0


def WaitingText(elapsed: float) -> str:
    dots = int(max(0.0, elapsed)) % 4
    edge = "." * dots
    return f"{edge}Waiting for Genesis{edge}"


@dataclass(frozen=True)
class Claim:
    player: str
    version: int
    country: str
    commitment: str
    ready: str = ""
    @classmethod
    def FromPayload(cls, value: object) -> Optional["Claim"]:
        if not isinstance(value, dict):
            return None
        try:
            claim = cls(str(value["player"]), int(value["version"]),
                        str(value["country"]).upper(), str(value["commitment"]),
                        str(value.get("ready", "")))
        except (KeyError, ValueError, TypeError):
            return None
        if claim.country not in Countries or len(claim.commitment) != 64 or not claim.player:
            return None
        return claim


class Field:
    """One independent observer's Pact and five-position Key."""
    def __init__(self, player: str) -> None:
        self.player = player
        self.claims: dict[str, Claim] = {}
        self.key: list[str | None] = [None] * Players
        self.invalid = 0
        self.version = 0

    def Register(self, country: str, secret: str) -> None:
        self.version += 1
        self.claims[self.player] = Claim(self.player, self.version, country, Commitment(country, secret))

    def Winners(self) -> dict[str, Claim]:
        winners: dict[str, Claim] = {}
        for claim in self.claims.values():
            old = winners.get(claim.country)
            if old is None or ClaimOrder(claim) < ClaimOrder(old):
                winners[claim.country] = claim
        return winners

    def Selected(self) -> bool:
        own = self.claims.get(self.player)
        return bool(own and self.Winners().get(own.country) == own)

    def Genesis(self) -> Optional[tuple[Claim, ...]]:
        winners = self.Winners()
        if len(winners) < Players:
            return None
        chosen = sorted(winners.values(), key=ClaimOrder)[:Players]
        if self.player not in {claim.player for claim in chosen}:
            return None
        order = {country: index for index, country in enumerate(Countries)}
        return tuple(sorted(chosen, key=lambda claim: order[claim.country]))

    def TopSecret(self, genesis: tuple[Claim, ...]) -> str:
        # Readiness is not part of Genesis. Only immutable claims define its digest.
        body = [(c.player, c.version, c.country, c.commitment) for c in genesis]
        return H(json.dumps(body, separators=(",", ":")))

    def Agree(self, genesis: tuple[Claim, ...]) -> bool:
        own = self.claims[self.player]
        digest = self.TopSecret(genesis)
        if own.ready == digest:
            return False
        self.claims[self.player] = replace(own, ready=digest)
        return True

    def PactComplete(self, genesis: tuple[Claim, ...]) -> bool:
        digest = self.TopSecret(genesis)
        return all(self.claims.get(c.player) is not None and
                   self.claims[c.player].ready == digest for c in genesis)

    def Payload(self, phase: str, genesis: Optional[tuple[Claim, ...]] = None) -> dict[str, object]:
        # Same envelope, transport, and admission machinery for both resolutions.
        if phase == "Pact":
            return {"Phase": phase, "Slots": [asdict(c) for c in sorted(self.claims.values(), key=lambda c: c.player)]}
        assert genesis is not None
        return {"Phase": "Key", "TopSecret": self.TopSecret(genesis), "Slots": list(self.key)}

    def Revealed(self, genesis: tuple[Claim, ...]) -> set[str]:
        return {claim.country for claim, secret in zip(genesis, self.key) if secret is not None}

    def TurnKey(self) -> bool:
        return sum(secret is not None for secret in self.key) >= Players - 1


class Truth:
    """The single admission boundary for Pact and Key projections."""
    @staticmethod
    def Admit(field: Field, payload: dict[str, object], genesis: Optional[tuple[Claim, ...]], active: bool) -> tuple[bool, bool]:
        phase, slots = payload.get("Phase"), payload.get("Slots")
        accepted, rejected = False, False
        if phase == "Pact" and isinstance(slots, list):
            digest = field.TopSecret(genesis) if genesis else ""
            for item in slots:
                claim = Claim.FromPayload(item)
                if claim is None or claim.player == field.player:
                    continue
                old = field.claims.get(claim.player)
                if old is None or claim.version > old.version:
                    field.claims[claim.player] = claim
                    accepted = True
                elif (old.version == claim.version and old.country == claim.country
                      and old.commitment == claim.commitment and claim.ready == digest
                      and old.ready != digest):
                    field.claims[claim.player] = claim
                    accepted = True
        elif (phase == "Key" and active and genesis is not None
              and payload.get("TopSecret") == field.TopSecret(genesis)
              and isinstance(slots, list) and len(slots) == Players):
            for index, secret in enumerate(slots):
                if secret is None:
                    continue
                claim = genesis[index]
                if isinstance(secret, str) and Commitment(claim.country, secret) == claim.commitment:
                    if field.key[index] is None:
                        field.key[index] = secret
                        accepted = True
                else:
                    rejected = True
        return accepted, rejected
    @staticmethod
    def Submit(field: Field, genesis: tuple[Claim, ...], secret: str) -> dict[str, object]:
        return {"Phase": "Key", "TopSecret": field.TopSecret(genesis),
                "Slots": [secret if c.player == field.player else None for c in genesis]}


class App:
    def __init__(self) -> None:
        self.mode = "Simulation"
        self.scenario = DefaultScenario
        self.secret = DefaultSecret
        self.countryIndex = 0
        self.player = H(f"{os.getpid()}:{time.time_ns()}:{os.urandom(16).hex()}")[:16]
        self.field = Field(self.player)
        self.oblivion = None
        self.noticeUntil = 0.0
        self.genesis: Optional[tuple[Claim, ...]] = None
        self.topSecret = ""
        self.readySince: Optional[float] = None
        self.result = ""
        self.frozen = "00.00"
        self.intruderUntil = 0.0
        self.truthOpen = False

    def OpenOblivion(self) -> None:
        if self.oblivion is not None or not self.scenario:
            return
        self.oblivion = Oblivion(self.scenario, self.mode)
        self.Project()

    def ResetProjection(self) -> None:
        self.field = Field(self.player)
        self.genesis = None
        self.topSecret = ""
        self.readySince = None
        self.noticeUntil = 0.0

    def ReopenOblivion(self) -> None:
        self.Close()
        self.ResetProjection()
        self.OpenOblivion()

    def Change(self, name: str, value: str) -> None:
        if name == "mode" or name == "scenario":
            if getattr(self, name) != value:
                setattr(self, name, value)
                self.ReopenOblivion()

    def Close(self) -> None:
        if self.oblivion is not None:
            self.oblivion.Close()
            self.oblivion = None

    def Project(self, phase: str | None = None) -> None:
        if self.oblivion is not None:
            self.oblivion.Send(self.field.Payload(phase or ("Key" if self.truthOpen else "Pact"), self.genesis))

    def Admit(self, payload: dict[str, object]) -> bool:
        accepted, rejected = Truth.Admit(self.field, payload, self.genesis, self.truthOpen)
        if rejected:
            if time.monotonic() >= self.intruderUntil:
                self.field.invalid += 1
            self.intruderUntil = time.monotonic() + FlashSeconds
        # LINCHPIN: accepted truth mutates local state, then reprojects it.
        if accepted:
            self.Project(payload.get("Phase"))
        if self.truthOpen and payload.get("Phase") == "Pact" and self.oblivion is not None:
            # An observer still establishing Genesis can recover the final Pact.
            incoming = payload.get("Slots", [])
            known = {c.get("player"): c for c in incoming if isinstance(c, dict)} if isinstance(incoming, list) else {}
            if any(known.get(c.player) != asdict(c) for c in self.field.claims.values()):
                self.Project("Pact")
        return rejected

    def Observe(self) -> None:
        if self.oblivion is not None:
            for payload in self.oblivion.Receive():
                self.Admit(payload)

    def LostSeat(self) -> bool:
        own = self.field.claims.get(self.player)
        if own is None or self.field.Selected():
            return False
        self.noticeUntil = time.monotonic() + 2.0
        self.countryIndex = Countries.index(own.country)
        self.field.claims.pop(self.player, None)
        self.field.version += 1
        self.Project()
        return True

    def SetupLines(self, focus: int) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        simBracket = White if focus == 0 else Ash
        liveBracket = White if focus == 1 else Ash
        simText = White if self.mode == "Simulation" else Ash
        liveText = White if self.mode == "Live" else Ash
        Place(lines, 4, Paint("[", simBracket) + Paint("Simulation", simText) + Paint("]", simBracket))
        Place(lines, 5, Paint("[", liveBracket) + Paint("Live", liveText) + Paint("]", liveBracket))
        scenarioColor = White if focus == 2 else Ash
        Place(lines, 8, Paint("Doomsday Scenario", scenarioColor))
        Place(lines, 9, Paint(self.scenario, scenarioColor))
        codeColor = White if focus == 3 else Ash
        Place(lines, 11, Paint("Authorization Code", codeColor))
        Place(lines, 12, Paint("*" * len(self.secret), codeColor))
        Place(lines, 15, Paint("Which Country Are You?", White if focus == 4 else Ash))
        winners = self.field.Winners() if self.oblivion else {}
        cells: list[str] = []
        for i, country in enumerate(Countries):
            available = country not in winners or winners[country].player == self.player
            name = f"{country:^8}" if available else " " * 8
            bracket = White if focus == 4 and i == self.countryIndex else Ash
            cells.append(Paint("[", bracket) + Paint(name, White) + Paint("]", bracket))
        Place(lines, 17, "     ".join(cells[:4]))
        Place(lines, 19, "     ".join(cells[4:]))

        if time.monotonic() < self.noticeUntil:
            Place(lines, 20, Paint("Country Already Selected", White))
            Place(lines, 21, Paint("Choose Another One", Ash))
        ExitHint(lines)
        return lines

    def setup(self, startFocus: int = 0) -> None:
        focus = startFocus
        lastFrame = ""
        while True:
            if self.oblivion:
                self.Observe()
                if self.LostSeat():
                    focus = 4

            lines = self.SetupLines(focus)
            frame = "\n".join(lines)
            if frame != lastFrame:
                Render(lines)
                lastFrame = frame

            ready = select.select([sys.stdin], [], [], Frame)[0]
            if not ready:
                continue
            key = ReadInput()
            if key in ("\x03", "ESC"):
                raise KeyboardInterrupt

            enter = key in ("\r", "\n")
            if focus < 2:
                if enter:
                    self.Change("mode", ("Simulation", "Live")[focus])
                    focus = 2
                elif key == "UP" and focus == 1:
                    focus = 0
                elif key == "DOWN":
                    focus += 1
            elif focus in (2, 3):
                value = self.scenario if focus == 2 else self.secret
                if key == "UP":
                    focus -= 1
                elif (key == "DOWN" or enter) and value:
                    focus += 1
                elif key in ("\x7f", "\b"):
                    if focus == 2:
                        self.Change("scenario", value[:-1])
                    else:
                        self.secret = value[:-1]
                elif len(key) == 1 and key.isprintable() and len(value) < 10:
                    if focus == 2:
                        self.Change("scenario", value + key.upper())
                    else:
                        self.secret = value + key
            elif key == "UP":
                if self.countryIndex >= 4:
                    self.countryIndex -= 4
                else:
                    focus = 3
            elif key == "DOWN" and self.countryIndex < 4:
                self.countryIndex += 4
            elif key in ("LEFT", "RIGHT"):
                row = 0 if self.countryIndex < 4 else 4
                step = -1 if key == "LEFT" else 1
                self.countryIndex = row + ((self.countryIndex - row + step) % 4)
            elif enter:
                country = Countries[self.countryIndex]
                winners = self.field.Winners()
                if country not in winners or winners[country].player == self.player:
                    self.field.Register(country, self.secret)
                    self.Project()
                    return

    def LobbyLines(self, elapsed: float) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        Place(lines, 6, Paint("Hold Your Breath", Ash))
        winners = self.field.Winners()
        names = [country for country in Countries if country in winners][:Players]
        for offset, country in enumerate(names):
            Place(lines, 9 + offset, Paint(country, White))
        for offset in range(len(names), Players):
            Place(lines, 9 + offset, Paint("_", Ash))
        Place(lines, 17, Paint(WaitingText(elapsed), Ash))
        ExitHint(lines)
        return lines

    def lobby(self) -> None:
        start = time.monotonic()
        lastSecond = -1
        while True:
            self.Observe()
            if self.LostSeat():
                self.setup(4)
                start = time.monotonic()
                continue
            genesis = self.field.Genesis()
            if genesis:
                digest = self.field.TopSecret(genesis)
                if digest != self.topSecret:
                    self.genesis, self.topSecret = genesis, digest
                    self.readySince = time.monotonic()
                if self.field.Agree(genesis):
                    self.Project()
                if (self.field.PactComplete(genesis) and self.readySince
                        and time.monotonic() - self.readySince >= 0.25):
                    return
            second = int(time.monotonic() - start)
            if second != lastSecond:
                Render(self.LobbyLines(time.monotonic() - start))
                lastSecond = second
            ready = select.select([sys.stdin], [], [], Frame)[0]
            if ready and ReadInput() in ("\x03", "ESC"):
                raise KeyboardInterrupt

    def BoardLines(self, clock: str, phase: str, phaseElapsed: float, typed: str) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        nuclearColor = White if phase != "NUCLEAR" or FlashWhite(phaseElapsed) else Ash
        Place(lines, 5, Paint("Nuclear Launch Detected", nuclearColor))
        if phase != "NUCLEAR":
            clockColor = White
            if phase in ("CLOCK", "FINISH"):
                clockColor = White if FlashWhite(phaseElapsed) else Ash
            elif phase == "LIVE":
                # Steady to 30s; alternate every half-second; final 10s flash 3 times/sec.
                remaining = max(0.0, CountdownNS / 1_000_000_000 - phaseElapsed)
                if remaining <= 10.0:
                    clockColor = White if int(phaseElapsed * 6) % 2 == 0 else Ash
                elif remaining <= 30.0:
                    clockColor = White if int(phaseElapsed * 2) % 2 == 0 else Ash
            Place(lines, 7, Paint(clock, clockColor))
        if self.genesis:
            localCountry = next(c.country for c in self.genesis if c.player == self.player)
            blocks, statuses = [], []
            aborted = set(self.field.Revealed(self.genesis))
            resolved = len(aborted) >= 4
            for claim in self.genesis:
                override = resolved and claim.country not in aborted
                lit = claim.country in aborted or override
                flag = Paint("[", Ash) + Paint("▪" if lit else " ", White if lit else Ash) + Paint("]", Ash)
                nameColor = White if claim.country == localCountry else Ash
                blocks.append(flag + " " + Paint(f"{claim.country:^8}", nameColor))
                status = "Aborted" if claim.country in aborted else ("Override" if override else "")
                statuses.append(Paint(f"{status:<12}", Ash))
            row = "   ".join(blocks)
            statusrow = "   ".join(statuses)
            Place(lines, 10, row)
            Place(lines, 11, statusrow)
        Place(lines, 14, Paint("Truth", Ash if phase != "LIVE" else White))
        Place(lines, 15, Paint("*" * len(typed), Ash if phase != "LIVE" else White))
        if time.monotonic() < self.intruderUntil:
            Place(lines, 19, Paint("Intruder Detected", White if FlashWhite(self.intruderUntil - time.monotonic()) else Ash))
        return lines

    @staticmethod
    def FormatClock(ns: int) -> str:
        ns = max(0, ns)
        hundredths = min(5999, ns // 10_000_000)
        return f"{hundredths // 100:02d}.{hundredths % 100:02d}"

    def game(self) -> None:
        fd = sys.stdin.fileno()
        gameTerm = termios.tcgetattr(fd)
        lockedTerm = termios.tcgetattr(fd)
        lockedTerm[0] &= ~(termios.IXON | termios.IXOFF)
        lockedTerm[3] &= ~(termios.ISIG | termios.IEXTEN)
        termios.tcsetattr(fd, termios.TCSANOW, lockedTerm)

        for phase in ("NUCLEAR", "CLOCK"):
            phaseStart = time.monotonic()
            while time.monotonic() - phaseStart < FlashSeconds:
                self.Observe()
                elapsed = time.monotonic() - phaseStart
                Render(self.BoardLines("59.99", phase, elapsed, ""))
                LockedFrame()

        typed = ""
        self.truthOpen = True
        self.Project()
        deadline = time.perf_counter_ns() + CountdownNS
        finishStart: Optional[float] = None
        outcome = ""
        while finishStart is None:
            self.Observe()
            remaining = deadline - time.perf_counter_ns()
            clock = self.FormatClock(remaining)
            if self.field.TurnKey():
                self.frozen = clock
                outcome = "ABORTED"
                finishStart = time.monotonic()
                break
            if remaining <= 0:
                self.frozen = "00.00"
                outcome = "SUCCESSFUL"
                finishStart = time.monotonic()
                break
            Render(self.BoardLines(clock, "LIVE", (CountdownNS - remaining) / 1_000_000_000, typed))
            ready = select.select([sys.stdin], [], [], Frame)[0]
            if not ready:
                continue
            key = ReadInput()
            if key in ("\x7f", "\b"):
                typed = typed[:-1]
            elif key in ("\r", "\n"):
                if typed:
                    if self.genesis and self.truthOpen:
                        proposal = Truth.Submit(self.field, self.genesis, typed)
                        if self.Admit(proposal) and self.oblivion is not None:
                            # Expose rejected evidence; peers verify it independently.
                            self.oblivion.Send(proposal)
                    typed = ""
            elif len(key) == 1 and key.isprintable() and len(typed) < 10:
                typed += key

        assert finishStart is not None
        while time.monotonic() - finishStart < FlashSeconds:
            self.Observe()
            elapsed = time.monotonic() - finishStart
            Render(self.BoardLines(self.frozen, "FINISH", elapsed, ""))
            LockedFrame()
        self.result = outcome
        termios.tcsetattr(fd, termios.TCSANOW, gameTerm)

    def WaitScreen(self, row: int, message: str, subline: str = "") -> None:
        lines = [""] * Height
        Place(lines, 2 if subline else row, TitleText() if subline else "")
        Place(lines, row, Paint(message, White))
        if subline:
            Place(lines, row + 3, Paint(subline, Ash))
        Render(lines)
        while True:
            ready = select.select([sys.stdin], [], [], Frame)[0]
            if ready:
                ReadInput()
                return

    def WarRoomFullScreen(self) -> None:
        self.WaitScreen(9, "War Room Full", "Try Another Doomsday")

    def StalemateScreen(self) -> None:
        self.WaitScreen(Height // 2 - 2, "Stalemate")

    def ResultLines(self) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        headline, subline = (("LAUNCH ABORTED", "Go Call Someone You Love")
                             if self.result == "ABORTED" else ("LAUNCH SUCCESSFUL", "Oblivion Awaits"))
        Place(lines, 8, Paint(headline, White))
        Place(lines, 11, Paint(subline, Ash))
        Place(lines, 20, Paint("[L]ogs                         [Space]", Ash))
        return lines

    def LogLines(self) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        Place(lines, 5, Paint("Logs", White))
        Place(lines, 7, Paint(f"Genesis {self.topSecret[:16]}", Ash))
        if self.genesis:
            for i, claim in enumerate(self.genesis):
                status = "Aborted" if claim.country in self.field.Revealed(self.genesis) else "Override"
                Place(lines, 9 + i, Paint(f"{claim.country:<8} {claim.commitment[:12]}  {status:<8}", Ash))
        Place(lines, 15, Paint(f"Intruders {self.field.invalid}", Ash))
        Place(lines, 16, Paint(f"Clock {self.frozen}", Ash))
        Place(lines, 17, Paint(f"Result {self.result.title()}", Ash))
        Place(lines, 20, Paint("[Space]", Ash))
        return lines

    def ResultScreen(self) -> None:
        showLogs = False
        while True:
            Render(self.LogLines() if showLogs else self.ResultLines())
            ready = select.select([sys.stdin], [], [], Frame)[0]
            if not ready:
                continue
            key = ReadInput()
            if key in ("\x03", "ESC", " "):
                return
            if key in ("l", "L") and not showLogs:
                showLogs = True

    def run(self) -> None:
        self.OpenOblivion()
        self.setup()
        self.lobby()
        self.game()
        self.ResultScreen()


def Run() -> None:
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        raise SystemExit("ICBM requires an interactive terminal")
    original = termios.tcgetattr(fd)
    app = App()
    try:
        tty.setcbreak(fd)
        sys.stdout.write("\x1b[2J\x1b[H" + HideCursor)
        sys.stdout.flush()
        app.run()
    except KeyboardInterrupt:
        if app.genesis is None:
            app.Close()
            app.StalemateScreen()
    except RuntimeError as exc:
        if str(exc) == "Simulation already has five occupied seats":
            app.WarRoomFullScreen()
        else:
            raise
    finally:
        app.Close()
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        sys.stdout.write("\x1b[2J\x1b[H" + ShowCursor)
        sys.stdout.flush()
