from __future__ import annotations

from dataclasses import dataclass, asdict
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
FieldEvery = 0.25
FlashSeconds = 3.0
CountdownNS = 10_000_000_000

Title = "Interactive Consistency Broadcast Machine"
Countries = ("USA", "CHINA", "RUSSIA", "UK", "PAKISTAN", "ISRAEL", "FRANCE", "INDIA")
DefaultScenario = "NIGHTFALL"
DefaultCode = "MIDNIGHT"

Reset = "\x1b[0m"
Ash = "\x1b[90m"
White = "\x1b[97m"
HideCursor = "\x1b[?25l"
ShowCursor = "\x1b[?25h"

Burst = 3
PacketLimit = 65535
SimulationHost = "127.0.0.1"
SimulationBasePort = 19482
LivePort = 19582
LiveBroadcast = "255.255.255.255"

class Channel:
    def __init__(self, scenario: str, mode: str) -> None:
        self.mode = mode
        self.mask = hashlib.sha256(scenario.encode("utf-8")).digest()
        self.sock: socket.socket | None = None
        self.port: int | None = None

        if mode == "Simulation":
            for port in range(SimulationBasePort, SimulationBasePort + Players):
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
                try:
                    sock.bind((SimulationHost, port))
                    sock.setblocking(False)
                    self.sock, self.port = sock, port
                    break
                except OSError:
                    sock.close()
            if self.sock is None:
                raise RuntimeError("Simulation already has five occupied seats")
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if hasattr(socket, "SO_REUSEPORT"):
                try:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
                except OSError:
                    pass
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            try:
                sock.bind(("0.0.0.0", LivePort))
            except OSError as exc:
                sock.close()
                raise RuntimeError(f"Live port {LivePort} is unavailable") from exc
            sock.setblocking(False)
            self.sock = sock

    def destinations(self) -> Iterable[tuple[str, int]]:
        if self.mode == "Simulation":
            return ((SimulationHost, port) for port in range(SimulationBasePort, SimulationBasePort + Players) if port != self.port)
        return ((LiveBroadcast, LivePort),)

    def encode(self, message: dict[str, object]) -> bytes:
        body = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return bytes(byte ^ self.mask[i % len(self.mask)] for i, byte in enumerate(body))

    def decode(self, raw: bytes) -> dict[str, object]:
        body = bytes(byte ^ self.mask[i % len(self.mask)] for i, byte in enumerate(raw))
        message = json.loads(body.decode("utf-8"))
        if not isinstance(message, dict):
            raise ValueError("packet is not an object")
        return message

    def send(self, message: dict[str, object]) -> None:
        if self.sock is None:
            return
        raw = self.encode(message)
        for host, port in self.destinations():
            for _ in range(Burst):
                try:
                    self.sock.sendto(raw, (host, port))
                except OSError:
                    pass

    def receive(self) -> list[dict[str, object]]:
        if self.sock is None:
            return []
        messages: list[dict[str, object]] = []
        while True:
            try:
                raw, _ = self.sock.recvfrom(PacketLimit)
            except (BlockingIOError, OSError):
                break
            try:
                messages.append(self.decode(raw))
            except Exception:
                continue
        return messages

    def close(self) -> None:
        sock, self.sock = self.sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

def H(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def Commitment(country: str, code: str) -> str:
    return H(f"{country}\0{code}")

def ClaimKey(claim: "Claim") -> tuple[str, str]:
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
    words = Title.split(" ")
    return " ".join(Paint(word[0], White) + Paint(word[1:], Ash) for word in words)

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

def ReadKey() -> str:
    key = sys.stdin.read(1)
    if key != "\x1b":
        return key
    if not select.select([sys.stdin], [], [], 0.01)[0]:
        return "ESC"
    second = sys.stdin.read(1)
    if second not in ("[", "O"):
        return "ESC"
    if not select.select([sys.stdin], [], [], 0.01)[0]:
        return "ESC"
    third = sys.stdin.read(1)
    key = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(third, "ESC")
    if key == "ESC":
        while select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.read(1)
    return key

def LockedFrame() -> None:
    ready, _, _ = select.select([sys.stdin], [], [], Frame)
    if ready:
        ReadKey()

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

    @classmethod
    def from_wire(cls, value: object) -> Optional["Claim"]:
        if not isinstance(value, dict):
            return None
        try:
            country = str(value["country"]).upper()
            claim = cls(
                player=str(value["player"]),
                version=int(value["version"]),
                country=country,
                commitment=str(value["commitment"]),
            )
        except Exception:
            return None
        if country not in Countries or len(claim.commitment) != 64 or not claim.player:
            return None
        return claim

class Field:
    def __init__(self, player: str) -> None:
        self.player = player
        self.claims: dict[str, Claim] = {}
        self.readies: dict[str, str] = {}
        self.reveals: dict[str, str] = {}
        self.invalid = 0
        self.version = 0

    def local_claim(self, country: str, code: str) -> None:
        self.version += 1
        self.claims[self.player] = Claim(self.player, self.version, country, Commitment(country, code))

    def merge_claim(self, claim: Claim) -> None:
        if claim.player == self.player:
            return
        old = self.claims.get(claim.player)
        if old is not None and old.version >= claim.version:
            return
        self.claims[claim.player] = claim
        self.readies.pop(claim.player, None)

    def winners(self) -> dict[str, Claim]:
        winners: dict[str, Claim] = {}
        for claim in self.claims.values():
            old = winners.get(claim.country)
            if old is None or ClaimKey(claim) < ClaimKey(old):
                winners[claim.country] = claim
        return winners

    def local_won(self) -> bool:
        local = self.claims.get(self.player)
        return bool(local and self.winners().get(local.country) == local)

    def genesis(self) -> Optional[tuple[Claim, ...]]:
        winners = self.winners()
        if len(winners) < Players:
            return None
        chosen = sorted(winners.values(), key=ClaimKey)[:Players]
        if self.player not in {claim.player for claim in chosen}:
            return None
        order = {country: index for index, country in enumerate(Countries)}
        return tuple(sorted(chosen, key=lambda claim: order[claim.country]))

    def digest(self, genesis: tuple[Claim, ...]) -> str:
        body = [(c.player, c.version, c.country, c.commitment) for c in genesis]
        return H(json.dumps(body, separators=(",", ":")))

    def field_packet(self) -> dict[str, object]:
        return {
            "type": "FIELD",
            "claims": [asdict(c) for c in self.claims.values()],
            "reveals": dict(self.reveals),
        }

class App:
    def __init__(self) -> None:
        self.mode = "Simulation"
        self.scenario = DefaultScenario
        self.code = DefaultCode
        self.country_index = 0
        self.player = H(f"{os.getpid()}:{time.time_ns()}:{os.urandom(16).hex()}")[:16]
        self.field = Field(self.player)
        self.channel = None
        self.notice_until = 0.0
        self.last_send = 0.0
        self.genesis: Optional[tuple[Claim, ...]] = None
        self.genesis_digest = ""
        self.ready_since: Optional[float] = None
        self.result = ""
        self.frozen = "00.000"
        self.intruder_until = 0.0
        self.reveals_open = False

    def open_channel(self) -> None:
        if self.channel is not None or not self.scenario:
            return
        self.channel = Channel(self.scenario, self.mode)
        self.last_send = 0.0
        self.send_field(True)

    def reset_projection(self) -> None:
        self.field = Field(self.player)
        self.genesis = None
        self.genesis_digest = ""
        self.ready_since = None
        self.notice_until = 0.0
        self.last_send = 0.0

    def reopen_channel(self) -> None:
        self.close()
        self.reset_projection()
        self.open_channel()

    def select_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self.reopen_channel()

    def set_scenario(self, scenario: str) -> None:
        scenario = scenario[:10]
        if scenario == self.scenario:
            return
        self.scenario = scenario
        self.reopen_channel()

    def close(self) -> None:
        if self.channel is not None:
            self.channel.close()
            self.channel = None

    def send_field(self, force: bool = False) -> None:
        if self.channel is None:
            return
        now = time.monotonic()
        if force or now - self.last_send >= FieldEvery:
            self.channel.send(self.field.field_packet())
            if self.genesis_digest:
                self.channel.send({"type": "READY", "player": self.player, "digest": self.genesis_digest})
            self.last_send = now

    def pump(self) -> None:
        if self.channel is None:
            return
        for message in self.channel.receive():
            kind = str(message.get("type", "")).upper()
            if kind == "FIELD":
                raw = message.get("claims", [])
                if not isinstance(raw, list):
                    continue
                for item in raw:
                    claim = Claim.from_wire(item)
                    if claim:
                        self.field.merge_claim(claim)
                reveals = message.get("reveals", {})
                if self.genesis and self.reveals_open and isinstance(reveals, dict):
                    for country, code in reveals.items():
                        self.ingest_reveal(str(country).upper(), str(code))
            elif kind == "READY":
                player, digest = str(message.get("player", "")), str(message.get("digest", ""))
                if player and len(digest) == 64:
                    self.field.readies[player] = digest
            elif kind == "REVEAL" and self.genesis and self.reveals_open:
                self.ingest_reveal(str(message.get("country", "")).upper(), str(message.get("code", "")))
        self.send_field()

    def ingest_reveal(self, country: str, code: str) -> None:
        if not self.genesis:
            return
        match = next((claim for claim in self.genesis if claim.country == country), None)

        # ================= LINCHPIN ================= #
        admissible = match is not None and Commitment(country, code) == match.commitment
        # ============================================ #

        if not admissible:
            self.field.invalid += 1
            self.intruder_until = time.monotonic() + FlashSeconds
            return
        if country in self.field.reveals:
            return
        self.field.reveals[country] = code

    def submit_reveal(self, code: str) -> None:
        if not self.genesis or not self.reveals_open:
            return
        local = next((claim for claim in self.genesis if claim.player == self.player), None)
        if local is None:
            return
        packet = {"type": "REVEAL", "country": local.country, "code": code}
        self.ingest_reveal(local.country, code)
        if self.channel is not None:
            self.channel.send(packet)

    def setup_lines(self, focus: int) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())

        sim_bracket = White if focus == 0 else Ash
        live_bracket = White if focus == 1 else Ash
        sim_text = White if self.mode == "Simulation" else Ash
        live_text = White if self.mode == "Live" else Ash
        Place(lines, 4, Paint("[", sim_bracket) + Paint("Simulation", sim_text) + Paint("]", sim_bracket))
        Place(lines, 5, Paint("[", live_bracket) + Paint("Live", live_text) + Paint("]", live_bracket))

        scenario_color = White if focus == 2 else Ash
        Place(lines, 8, Paint("Doomsday Scenario", scenario_color))
        Place(lines, 9, Paint(self.scenario, scenario_color))

        code_color = White if focus == 3 else Ash
        Place(lines, 11, Paint("Authorization Code", code_color))
        Place(lines, 12, Paint("*" * len(self.code), code_color))

        Place(lines, 15, Paint("Which Country Are You?", White if focus == 4 else Ash))
        winners = self.field.winners() if self.channel else {}
        cells: list[str] = []
        for i, country in enumerate(Countries):
            available = country not in winners or winners[country].player == self.player
            name = f"{country:^8}" if available else " " * 8
            bracket = White if focus == 4 and i == self.country_index else Ash
            cells.append(Paint("[", bracket) + Paint(name, White) + Paint("]", bracket))
        Place(lines, 17, "     ".join(cells[:4]))
        Place(lines, 19, "     ".join(cells[4:]))

        if time.monotonic() < self.notice_until:
            Place(lines, 20, Paint("Country Already Selected", White))
            Place(lines, 21, Paint("Choose Another One", Ash))
        ExitHint(lines)
        return lines

    def setup(self, start_focus: int = 0) -> None:
        focus = start_focus
        last_frame = ""
        while True:
            if self.channel:
                self.pump()
                local = self.field.claims.get(self.player)
                if local and not self.field.local_won():
                    self.notice_until = time.monotonic() + 2.0
                    self.country_index = Countries.index(local.country)
                    self.field.claims.pop(self.player, None)
                    self.field.version += 1
                    focus = 4
                    self.send_field(True)

            lines = self.setup_lines(focus)
            frame = "\n".join(lines)
            if frame != last_frame:
                Render(lines)
                last_frame = frame

            ready, _, _ = select.select([sys.stdin], [], [], Frame)
            if not ready:
                continue
            key = ReadKey()
            if key in ("\x03", "ESC"):
                raise KeyboardInterrupt

            if focus == 0:
                if key == "DOWN":
                    focus = 1
                elif key in ("\r", "\n"):
                    self.select_mode("Simulation")
                    focus = 2

            elif focus == 1:
                if key == "UP":
                    focus = 0
                elif key == "DOWN":
                    focus = 2
                elif key in ("\r", "\n"):
                    self.select_mode("Live")
                    focus = 2

            elif focus == 2:
                if key == "UP":
                    focus = 1
                elif key == "DOWN" or key in ("\r", "\n"):
                    if self.scenario:
                        focus = 3
                elif key in ("\x7f", "\b"):
                    self.set_scenario(self.scenario[:-1])
                elif len(key) == 1 and key.isprintable() and len(self.scenario) < 10:
                    self.set_scenario(self.scenario + key.upper())

            elif focus == 3:
                if key == "UP":
                    focus = 2
                elif key == "DOWN" or key in ("\r", "\n"):
                    if self.code:
                        focus = 4
                elif key in ("\x7f", "\b"):
                    self.code = self.code[:-1]
                elif len(key) == 1 and key.isprintable() and len(self.code) < 10:
                    self.code += key

            else:
                if key == "UP":
                    if self.country_index >= 4:
                        self.country_index -= 4
                    else:
                        focus = 3
                elif key == "DOWN" and self.country_index < 4:
                    self.country_index += 4
                elif key == "LEFT":
                    row = 0 if self.country_index < 4 else 4
                    self.country_index = row + ((self.country_index - row - 1) % 4)
                elif key == "RIGHT":
                    row = 0 if self.country_index < 4 else 4
                    self.country_index = row + ((self.country_index - row + 1) % 4)
                elif key in ("\r", "\n"):
                    country = Countries[self.country_index]
                    winners = self.field.winners()
                    if country in winners and winners[country].player != self.player:
                        continue
                    self.field.local_claim(country, self.code)
                    self.send_field(True)
                    return

    def lobby_lines(self, elapsed: float) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        Place(lines, 6, Paint("Hold Your Breath", Ash))
        winners = self.field.winners()
        names = [country for country in Countries if country in winners]
        names = names[:Players]
        for offset, country in enumerate(names):
            Place(lines, 9 + offset, Paint(country, White))
        for offset in range(len(names), Players):
            Place(lines, 9 + offset, Paint("_", Ash))
        Place(lines, 17, Paint(WaitingText(elapsed), Ash))
        ExitHint(lines)
        return lines

    def lobby(self) -> None:
        start = time.monotonic()
        last_second = -1
        while True:
            self.pump()
            local = self.field.claims.get(self.player)
            if local and not self.field.local_won():
                self.notice_until = time.monotonic() + 2.0
                self.country_index = Countries.index(local.country)
                self.field.claims.pop(self.player, None)
                self.field.version += 1
                self.setup(4)
                start = time.monotonic()
                continue
            genesis = self.field.genesis()
            if genesis:
                digest = self.field.digest(genesis)
                if digest != self.genesis_digest:
                    self.genesis, self.genesis_digest = genesis, digest
                    self.field.readies = {self.player: digest}
                    self.ready_since = time.monotonic()
                self.send_field(True)
                players = {c.player for c in genesis}
                agreeing = {p for p, d in self.field.readies.items() if p in players and d == digest}
                if agreeing >= players and self.ready_since and time.monotonic() - self.ready_since >= 0.25:
                    return
            second = int(time.monotonic() - start)
            if second != last_second:
                Render(self.lobby_lines(time.monotonic() - start))
                last_second = second
            ready, _, _ = select.select([sys.stdin], [], [], Frame)
            if ready and ReadKey() in ("\x03", "ESC"):
                raise KeyboardInterrupt

    def board_lines(self, clock: str, phase: str, phase_elapsed: float, typed: str) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        nuclear_color = White if phase != "NUCLEAR" or FlashWhite(phase_elapsed) else Ash
        Place(lines, 5, Paint("Nuclear Launch Detected", nuclear_color))
        if phase != "NUCLEAR":
            clock_color = White
            if phase in ("CLOCK", "FINISH"):
                clock_color = White if FlashWhite(phase_elapsed) else Ash
            Place(lines, 7, Paint(clock, clock_color))
        if self.genesis:
            local_country = next(c.country for c in self.genesis if c.player == self.player)
            blocks, statuses = [], []
            aborted = set(self.field.reveals)
            resolved = len(aborted) >= 4
            for claim in self.genesis:
                override = resolved and claim.country not in aborted
                lit = claim.country in aborted or override
                flag = Paint("[", Ash) + Paint("▪" if lit else " ", White if lit else Ash) + Paint("]", Ash)
                name_color = White if claim.country == local_country and claim.country not in aborted and not resolved else Ash
                blocks.append(flag + " " + Paint(f"{claim.country:^8}", name_color))
                status = "Aborted" if claim.country in aborted else ("Override" if override else "")
                statuses.append(Paint(f"{status:<12}", Ash))
            row = "   ".join(blocks)
            statusrow = "   ".join(statuses)
            Place(lines, 10, row)
            Place(lines, 11, statusrow)
        Place(lines, 14, Paint("Authorization Code", Ash if phase != "LIVE" else White))
        Place(lines, 15, Paint("*" * len(typed), Ash if phase != "LIVE" else White))
        if time.monotonic() < self.intruder_until:
            Place(lines, 19, Paint("Intruder Detected", White if FlashWhite(self.intruder_until - time.monotonic()) else Ash))
        return lines

    @staticmethod
    def format_clock(ns: int) -> str:
        ns = max(0, ns)
        millis = ns // 1_000_000
        return f"{millis // 1000:02d}.{millis % 1000:03d}"

    def game(self) -> None:
        fd = sys.stdin.fileno()
        game_term = termios.tcgetattr(fd)
        locked_term = termios.tcgetattr(fd)
        locked_term[0] &= ~(termios.IXON | termios.IXOFF)
        locked_term[3] &= ~(termios.ISIG | termios.IEXTEN)
        termios.tcsetattr(fd, termios.TCSANOW, locked_term)

        phase_start = time.monotonic()
        while time.monotonic() - phase_start < FlashSeconds:
            self.pump()
            elapsed = time.monotonic() - phase_start
            Render(self.board_lines("10.000", "NUCLEAR", elapsed, ""))
            LockedFrame()

        phase_start = time.monotonic()
        while time.monotonic() - phase_start < FlashSeconds:
            self.pump()
            elapsed = time.monotonic() - phase_start
            Render(self.board_lines("10.000", "CLOCK", elapsed, ""))
            LockedFrame()

        typed = ""
        self.reveals_open = True
        deadline = time.perf_counter_ns() + CountdownNS
        finish_start: Optional[float] = None
        outcome = ""
        while finish_start is None:
            self.pump()
            remaining = deadline - time.perf_counter_ns()
            clock = self.format_clock(remaining)
            if len(self.field.reveals) >= 4:
                self.frozen = clock
                outcome = "ABORTED"
                self.reveals_open = False
                finish_start = time.monotonic()
                break
            if remaining <= 0:
                self.frozen = "00.000"
                outcome = "SUCCESSFUL"
                self.reveals_open = False
                finish_start = time.monotonic()
                break
            Render(self.board_lines(clock, "LIVE", 0.0, typed))
            ready, _, _ = select.select([sys.stdin], [], [], Frame)
            if not ready:
                continue
            key = ReadKey()
            if key in ("\x7f", "\b"):
                typed = typed[:-1]
            elif key in ("\r", "\n"):
                if typed:
                    self.submit_reveal(typed)
                    typed = ""
            elif len(key) == 1 and key.isprintable() and len(typed) < 10:
                typed += key

        assert finish_start is not None
        while time.monotonic() - finish_start < FlashSeconds:
            self.pump()
            elapsed = time.monotonic() - finish_start
            Render(self.board_lines(self.frozen, "FINISH", elapsed, ""))
            LockedFrame()
        self.result = outcome
        termios.tcsetattr(fd, termios.TCSANOW, game_term)

    def stalemate_lines(self) -> list[str]:
        lines = [""] * Height
        Place(lines, Height // 2 - 2, Paint("Stalemate", White))
        return lines

    def stalemate_screen(self) -> None:
        Render(self.stalemate_lines())
        while True:
            try:
                ready, _, _ = select.select([sys.stdin], [], [], Frame)
                if ready:
                    ReadKey()
                    return
            except KeyboardInterrupt:
                return

    def result_lines(self) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        if self.result == "ABORTED":
            Place(lines, 8, Paint("LAUNCH ABORTED", White))
            Place(lines, 11, Paint("Go Call Someone You Love", Ash))
        else:
            Place(lines, 8, Paint("LAUNCH SUCCESSFUL", White))
            Place(lines, 11, Paint("Oblivion Awaits", Ash))
        Place(lines, 20, Paint("[L]ogs                         [Space]", Ash))
        return lines

    def log_lines(self) -> list[str]:
        lines = [""] * Height
        Place(lines, 2, TitleText())
        Place(lines, 5, Paint("Logs", White))
        Place(lines, 7, Paint(f"Genesis {self.genesis_digest[:16]}", Ash))
        if self.genesis:
            for i, claim in enumerate(self.genesis):
                status = "Aborted" if claim.country in self.field.reveals else "Override"
                Place(lines, 9 + i, Paint(f"{claim.country:<8} {claim.commitment[:12]}  {status:<8}", Ash))
        Place(lines, 15, Paint(f"Intruders {self.field.invalid}", Ash))
        Place(lines, 16, Paint(f"Clock {self.frozen}", Ash))
        Place(lines, 17, Paint(f"Result {self.result.title()}", Ash))
        Place(lines, 20, Paint("[Space]", Ash))
        return lines

    def result_screen(self) -> None:
        show_logs = False
        while True:
            Render(self.log_lines() if show_logs else self.result_lines())
            ready, _, _ = select.select([sys.stdin], [], [], Frame)
            if not ready:
                continue
            key = ReadKey()
            if key in ("\x03", "ESC", " "):
                return
            if key in ("l", "L") and not show_logs:
                show_logs = True

    def run(self) -> None:
        self.open_channel()
        self.setup()
        self.lobby()
        self.game()
        self.result_screen()

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
            app.stalemate_screen()
    finally:
        app.close()
        termios.tcsetattr(fd, termios.TCSADRAIN, original)
        sys.stdout.write("\x1b[2J\x1b[H" + ShowCursor)
        sys.stdout.flush()
