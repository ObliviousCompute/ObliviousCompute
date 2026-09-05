from __future__ import annotations

import hashlib
import os
import select
import shutil
import sys
import termios
import tty
from contextlib import contextmanager

from .Catacomb import (
    BonePile,
    BonesPerHead,
    Catacomb,
    GenesisChild,
    Head,
    PublicKeyHex,
    Result,
    StateKey,
    Tag,
    ValidKey,
    ZeroHash,
    ZeroSign,
)
from .BoneYard import BoneYard

Uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MaxHeads = 9
NameMax = 8
Width = 80
Height = 24
BoardGap = "     "
Guardians = (
    "Bandit", "Baxter", "Bear", "Bingo", "Buddy", "Buster",
    "Chief", "Coco", "Copper", "Duke", "Fido", "Gizmo",
    "Goose", "Hank", "Jasper", "Lucky", "Marley", "Milo",
    "Otis", "Rex", "Riley", "Rocky", "Scout", "Teddy",
    "Toby", "Ziggy", "Bruno", "Daisy", "Finn", "Harley",
    "Jax", "Moose", "Pepper", "Rusty", "Shadow", "Spot",
)

Clear = "\x1b[2J\x1b[H"
Hide = "\x1b[?25l"
Show = "\x1b[?25h"

Up, Down, Left, Right = "UP", "DOWN", "LEFT", "RIGHT"
Enter, Space, HungerKey, Other = "ENTER", "SPACE", "HUNGER", "OTHER"


class ExitCerberus(Exception):
    pass


def Heads(count: int = MaxHeads) -> list[str]:
    return list(Uppercase[:max(1, min(MaxHeads, int(count)))])


def HeadCountHash(counted: dict[str, str]) -> str:
    keys = sorted((str(key) for key in counted), reverse=True)
    body = b"".join(bytes.fromhex(key) for key in keys)
    return hashlib.sha256(b"CERBERUS::HEADCOUNT::V1::" + body).hexdigest()


def HashRank(seed: str, domain: str, value: str) -> str:
    body = f"CERBERUS::{domain}::V1::{seed}::{value}".encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def Canvas() -> list[str]:
    return [" " * Width] * Height


def Put(card: list[str], row: int, text: str, col: int = 0) -> None:
    if not 0 <= row < Height or not 0 <= col < Width:
        return
    text = str(text)[:Width - col]
    line = card[row]
    card[row] = line[:col] + text + line[col + len(text):]


def Center(card: list[str], row: int, text: str) -> None:
    text = str(text)[:Width]
    Put(card, row, text, max(0, (Width - len(text)) // 2))


def Paint(card: list[str]) -> None:
    columns, rows = shutil.get_terminal_size(fallback=(Width, Height))
    prefix = " " * max((columns - Width) // 2, 0)
    screen = [""] * max((rows - Height) // 2, 0) + [prefix + line.rstrip() for line in card]
    sys.stdout.write(Clear + "\n".join(screen))
    sys.stdout.flush()


def ReadKey(fd: int) -> str:
    first = os.read(fd, 1)
    if not first or first == b"\x03":
        raise ExitCerberus
    if first in (b"\r", b"\n"):
        return Enter
    if first == b" ":
        return Space
    if first in (b"h", b"H"):
        return HungerKey
    if first != b"\x1b":
        return Other
    ready, _, _ = select.select([fd], [], [], 0.04)
    if not ready or os.read(fd, 1) != b"[":
        return Other
    ready, _, _ = select.select([fd], [], [], 0.04)
    if not ready:
        return Other
    return {b"A": Up, b"B": Down, b"C": Right, b"D": Left}.get(os.read(fd, 1), Other)


@contextmanager
def Terminal():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    sys.stdout.write(Hide)
    sys.stdout.flush()
    try:
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(Show)
        sys.stdout.flush()


def EntryCard(question: str, value: str, bracket: bool = False) -> None:
    card = Canvas()
    Center(card, 6, "Welcome To")
    Center(card, 8, "Cerberus")
    Center(card, 11, question)
    shown = f"[{str(value).ljust(NameMax)[:NameMax]}]" if bracket else str(value)
    Center(card, 13, shown)
    Paint(card)


def Field(
    fd: int,
    question: str,
    value: str = "",
    limit: int = NameMax,
    bracket: bool = False,
) -> tuple[str, str]:
    chars = list(str(value)[:limit])
    while True:
        EntryCard(question, "".join(chars), bracket)
        raw = os.read(fd, 1)
        if not raw or raw == b"\x03":
            raise ExitCerberus
        if raw == b"\x1b":
            ready, _, _ = select.select([fd], [], [], 0.04)
            if ready and os.read(fd, 1) == b"[":
                ready, _, _ = select.select([fd], [], [], 0.04)
                arrow = os.read(fd, 1) if ready else b""
                if arrow == b"D":
                    return "".join(chars).strip(), Left
                if arrow == b"C" and "".join(chars).strip():
                    return "".join(chars).strip(), Right
                continue
        if raw in (b"\r", b"\n"):
            text = "".join(chars).strip()
            if text:
                return text, Right
        elif raw in (b"\x7f", b"\x08"):
            if chars:
                chars.pop()
        elif 32 <= raw[0] <= 126 and len(chars) < limit:
            chars.append(raw.decode("ascii"))


def ChooseHeads(fd: int, count: int = 1) -> tuple[int, str]:
    count = max(1, min(MaxHeads, int(count)))
    while True:
        EntryCard("How Many Heads?", str(count))
        key = ReadKey(fd)
        if key == Up:
            count = min(MaxHeads, count + 1)
        elif key == Down:
            count = max(1, count - 1)
        elif key == Left:
            return count, Left
        elif key in (Right, Enter):
            return count, Right


def Setup(fd: int) -> tuple[str, int, str, str]:
    fields = ["Fluffyyy", 1, "Backbite", "Paradise"]
    step = 0
    while step < len(fields):
        if step == 0:
            fields[0], move = Field(fd, "What's Its Name", str(fields[0]))
        elif step == 1:
            fields[1], move = ChooseHeads(fd, int(fields[1]))
        elif step == 2:
            fields[2], move = Field(
                fd, "What's Your DogTag", str(fields[2])
            )
        else:
            fields[3], move = Field(fd, "Claim Your BonePile", str(fields[3]))
        step = max(0, step - 1) if move == Left else step + 1
    return str(fields[0]), int(fields[1]), str(fields[2]), str(fields[3])


def OblivionCard(counted: dict[str, str]) -> None:
    card = Canvas()
    Center(card, 4, "ENTERING OBLIVION")
    for row, dogtag in enumerate((tag for _, tag in sorted(counted.items(), reverse=True)), 7):
        Center(card, row, str(dogtag)[:NameMax])
    Paint(card)


def LeaveScreen(fd: int) -> None:
    card = Canvas()
    Center(card, 9, "YOU ARE NOW LEAVING OBLIVION")
    Center(card, 12, "Remember Your DogTags If You Want To Rejoin")
    Paint(card)
    try:
        ReadKey(fd)
    except ExitCerberus:
        pass
    finally:
        sys.stdout.write(Clear)
        sys.stdout.flush()


def PortTakenScreen(fd: int) -> None:
    card = Canvas()
    Center(card, 9, "Stop chasing your tail.")
    Center(card, 12, "This port is already taken.")
    Paint(card)
    try:
        ReadKey(fd)
    except ExitCerberus:
        return
    LeaveScreen(fd)


def HeadTile(name: str, cell: Head) -> tuple[str, str]:
    top = f"{str(name)[:NameMax]:<{NameMax}} {int(cell.bones):>2}"
    bottom = "Dirty------" if cell.clawcount else f"{cell.tag.child[:5]}------"
    return top, bottom


def BoardRows(names: dict[str, str], state: BonePile) -> list[str]:
    rows: list[str] = []
    heads = Heads()
    for start in range(0, MaxHeads, 3):
        top, bottom = [], []
        for head in heads[start:start + 3]:
            first, second = HeadTile(names.get(head, head), state[head])
            top.append(first)
            bottom.append(second)
        rows.extend((BoardGap.join(top), BoardGap.join(bottom)))
        if start < 6:
            rows.append("")
    return rows


class Guardian:

    def __init__(self, cerberus: str):
        self.cerberus = str(cerberus)[:NameMax]
        self.count = 0
        self.dogtag = ""
        self.bonepile = ""
        self.secret = ""
        self.privatekey = None
        self.publickey = ""
        self.counted: dict[str, str] = {}
        self.names: dict[str, str] = {}
        self.headcounthash = ""

        self.heads: list[str] = []
        self.expected: set[str] = set()
        self.head = ""
        self.target = ""
        self.amount = 1
        self.notice: str | None = None

        self.catacomb: Catacomb | None = None
        self.state: BonePile = {}
        self.boneyard = BoneYard(self.cerberus, HeadCountIn=self.HeadCount, NoticeOut=self.Notice)

    def Open(self, count: int) -> None:
        self.count = max(1, min(MaxHeads, int(count)))
        self.boneyard.Open(self.count)

    def Identity(self, dogtag: str, bonepile: str) -> None:
        if self.count < 1:
            raise RuntimeError("HeadCount must be chosen before identity.")
        self.dogtag = str(dogtag).strip()[:NameMax]
        self.bonepile = str(bonepile).strip()[:NameMax]
        self.secret = f"{self.dogtag}|{self.bonepile}"
        self.privatekey = StateKey(self.secret)
        self.publickey = PublicKeyHex(self.privatekey)
        self.counted = {self.publickey: self.dogtag}
        self.HeadCount()
        if len(self.counted) == self.count and self.catacomb is None:
            self.Genesis()

    def HeadCount(self, value: object = None) -> bool:
        if value is None:
            if not self.publickey:
                return False
            heads = [
                {"dogtag": tag, "key": key}
                for key, tag in sorted(self.counted.items(), reverse=True)
            ]
            self.boneyard.HeadCount({"count": self.count, "heads": heads})
            return False
        if not self.publickey or not isinstance(value, dict):
            return False
        try:
            if int(value.get("count", 0)) != self.count:
                return False
            rawheads = value.get("heads", [])
            if not isinstance(rawheads, (list, tuple)):
                return False
            incoming: dict[str, str] = {}
            for item in rawheads:
                if not isinstance(item, dict):
                    return False
                dogtag = str(item.get("dogtag", "")).strip()[:NameMax]
                key = str(item.get("key", "")).strip()
                if not dogtag or not ValidKey(key):
                    return False
                if key in incoming and incoming[key] != dogtag:
                    return False
                incoming[key] = dogtag
        except Exception:
            return False
        if len(incoming) > self.count:
            return False

        before = dict(self.counted)
        merged = dict(before)
        for key, dogtag in incoming.items():
            if key in merged and merged[key] != dogtag:
                return False
            merged[key] = dogtag
        if len(merged) > self.count:
            return False

        changed = merged != before
        if changed:
            self.counted = dict(sorted(merged.items(), reverse=True))
            self.HeadCount()

        complete = len(self.counted) == self.count
        incomingkeys = set(incoming)
        if complete and not changed and 0 < len(incomingkeys) < self.count and incomingkeys <= set(self.counted):
            self.HeadCount()
        if complete and self.catacomb is None:
            self.Genesis()
        return changed

    def Genesis(self) -> None:
        if self.catacomb is not None or len(self.counted) != self.count:
            return
        active = sorted(self.counted.items(), reverse=True)
        activekeys = {key for key, _dogtag in active}
        if self.publickey not in activekeys:
            raise RuntimeError("Cerberus is full.")

        self.headcounthash = HeadCountHash(self.counted)
        usednames = {dogtag.casefold() for _key, dogtag in active}
        dogpool = sorted(
            (name for name in Guardians if name.casefold() not in usednames),
            key=lambda name: HashRank(self.headcounthash, "GUARDIANNAME", name),
            reverse=True,
        )
        needed = MaxHeads - len(active)
        roster = list(active)
        for index, name in enumerate(dogpool[:needed]):
            guardiansecret = f"Guardian|{self.headcounthash}|{index}|{name}"
            guardprivate = StateKey(guardiansecret)
            guardkey = PublicKeyHex(guardprivate)
            del guardprivate
            roster.append((guardkey, name))

        if len(roster) != MaxHeads or len({key for key, _name in roster}) != MaxHeads:
            raise RuntimeError("Guardian could not fill Genesis.")
        roster.sort(
            key=lambda item: HashRank(self.headcounthash, "HEADORDER", item[0]),
            reverse=True,
        )

        self.heads = Heads()
        self.expected = set(self.heads)
        slotbykey = {key: self.heads[index] for index, (key, _name) in enumerate(roster)}
        self.head = slotbykey[self.publickey]
        self.names = {self.heads[index]: name for index, (_key, name) in enumerate(roster)}
        self.target = next(head for head in self.heads if head != self.head)

        genesis: BonePile = {}
        for index, head in enumerate(self.heads):
            key = roster[index][0]
            tag = Tag(ZeroHash, GenesisChild(head, key))
            genesis[head] = Head(head, key, BonesPerHead, tag, ZeroSign)

        self.catacomb = Catacomb(self.heads, self.head, self.secret, GuardianOut=self.Catacomb)
        self.boneyard.Attach(
            self.heads,
            self.head,
            CatacombIn=self.catacomb.BoneYard,
            BonePileIn=self.catacomb.ReceiveBonePile,
            BonePileOut=lambda: self.catacomb.BonePile,
        )
        self.catacomb.BoneYardOut = self.boneyard.Catacomb
        self.catacomb.HungerOut = self.boneyard.Hunger
        self.catacomb.ProjectOut = self.boneyard.SendBonePile
        if self.catacomb.Seed(genesis).status == "BAD BONEPILE":
            raise RuntimeError("Guardian could not generate Genesis.")
        self.state = self.catacomb.BonePile
        if self.count > 1:
            self.catacomb.Hunger()

    def Catacomb(self, pile: BonePile, result: Result) -> None:
        self.state = pile
        self.notice = None
        self.ClampAmount()

    def Notice(self, text: str) -> None:
        self.notice = str(text)

    def ClampAmount(self) -> None:
        if not self.head or self.head not in self.state:
            return
        available = max(0, int(self.state[self.head].bones))
        self.amount = 0 if available == 0 else max(1, min(self.amount, available))

    def MoveTarget(self, step: int) -> None:
        targets = [head for head in self.heads if head != self.head] or [self.head]
        if self.target not in targets:
            self.target = targets[0]
            return
        self.target = targets[(targets.index(self.target) + step) % len(targets)]

    def Game(self) -> None:
        self.ClampAmount()
        card = Canvas()
        Center(card, 2, "OBLIVION")

        board = BoardRows(self.names, self.state)
        boardwidth = max(map(len, board))
        left = max(0, (Width - boardwidth) // 2)
        for row, text in enumerate(board, 5):
            Put(card, row, text, left)

        noun = "bone" if self.amount == 1 else "bones"
        source = self.names.get(self.head, self.head)
        target = self.names.get(self.target, self.target)
        action = self.notice or f"{source} lets {target} steal {self.amount} {noun}"
        Put(card, 15, action, left)

        total = sum(int(self.state[head].bones) for head in self.heads)
        expected = BonesPerHead * MaxHeads
        Put(card, 17, f"Bones: {total} / {expected}", left)
        Put(card, 18, f"BonePile: {self.bonepile}", left)
        Put(card, 19, f"Cerberus: {self.cerberus}", left)
        Put(card, 21, "(H)unger for a fresh BonePile", left)
        Paint(card)

    def Commit(self) -> bool:
        if self.catacomb is None:
            return False
        result = self.catacomb.Guardian(self.target, self.amount)
        if result.status == "IDEMPOTENT":
            return False
        if result.status == "BAD BONE":
            self.notice = "BAD BONE"
            return True
        return bool(result.changed)

    def Hunger(self) -> bool:
        if self.catacomb is None:
            return False
        self.catacomb.Hunger()
        self.notice = "HUNGER"
        return True

    def Run(self, fd: int) -> None:
        if self.boneyard.sock is None or not self.publickey:
            raise RuntimeError("Guardian has not entered Oblivion.")
        OblivionCard(self.counted)
        while self.catacomb is None or self.catacomb.Hungry:
            ready, _, _ = select.select([self.boneyard.sock, fd], [], [])
            if fd in ready:
                try:
                    ReadKey(fd)
                except ExitCerberus:
                    pass
                LeaveScreen(fd)
                return
            if self.boneyard.sock in ready and self.boneyard.Pump():
                OblivionCard(self.counted)

        self.state = self.catacomb.BonePile
        self.notice = None
        termios.tcflush(fd, termios.TCIFLUSH)
        self.Game()

        while True:
            ready, _, _ = select.select([self.boneyard.sock, fd], [], [])
            redraw = self.boneyard.Pump() if self.boneyard.sock in ready else False
            if fd in ready:
                try:
                    key = ReadKey(fd)
                except ExitCerberus:
                    LeaveScreen(fd)
                    return
                if key in (Up, Down, Left, Right, Enter, HungerKey):
                    self.notice = None
                if key == Up:
                    self.MoveTarget(+1)
                    redraw = True
                elif key == Down:
                    self.MoveTarget(-1)
                    redraw = True
                elif key == Right:
                    self.amount += 1
                    redraw = True
                elif key == Left:
                    self.amount -= 1
                    redraw = True
                elif key == Enter:
                    redraw = self.Commit() or redraw
                elif key == HungerKey:
                    redraw = self.Hunger() or redraw
                elif key == Space:
                    LeaveScreen(fd)
                    return
            if redraw:
                self.Game()

    def Close(self) -> None:
        self.boneyard.Close()


def Run() -> None:
    if not sys.stdin.isatty():
        raise RuntimeError("Cerberus needs a terminal for arrow-key input.")
    guardian: Guardian | None = None
    with Terminal() as fd:
        try:
            cerberus, count, dogtag, bonepile = Setup(fd)
            guardian = Guardian(cerberus)
            try:
                guardian.Open(count)
            except RuntimeError as error:
                if "No clean mouth available" not in str(error):
                    raise
                PortTakenScreen(fd)
                return
            guardian.Identity(dogtag, bonepile)
            guardian.Run(fd)
        except (ExitCerberus, KeyboardInterrupt):
            pass
        finally:
            if guardian is not None:
                guardian.Close()
    sys.stdout.write(Clear)
    sys.stdout.flush()
