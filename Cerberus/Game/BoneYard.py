from __future__ import annotations

from collections import deque
from dataclasses import asdict
import hashlib
import json
import socket
from typing import Callable, Iterable, Optional

from .Catacomb import (
    Bone,
    BonePile,
    Head,
    Result,
    Tag,
)

Host = "127.0.0.1"
BonePilePort = 9000
Burst = 3
CacheLimit = 4096


def BoneKey(bone: Bone) -> tuple[object, ...]:
    return (bone.head, bone.key, bone.target, bone.bones, bone.tag.parent, bone.tag.child, bone.locksign, bone.sign)


def BoneToWire(bone: Bone) -> dict[str, object]:
    return asdict(bone)


def BoneFromWire(value: object) -> Bone:
    if not isinstance(value, dict) or not isinstance(value.get("tag"), dict):
        raise ValueError("Bone has bad shape")
    tag = value["tag"]
    return Bone(
        head=str(value.get("head", "")).upper(),
        key=str(value.get("key", "")),
        target=str(value.get("target", "")).upper(),
        bones=int(value.get("bones", 0)),
        tag=Tag(str(tag.get("parent", "")), str(tag.get("child", ""))),
        locksign=str(value.get("locksign", "")),
        sign=str(value.get("sign", "")),
    )


def HeadToWire(cell: Head) -> dict[str, object]:
    return asdict(cell)


def HeadFromWire(value: object) -> Head:
    if not isinstance(value, dict) or not isinstance(value.get("tag"), dict):
        raise ValueError("BonePile Head has bad shape")
    tag = value["tag"]
    receipts = value.get("receipts", [])
    if not isinstance(receipts, (list, tuple)) or len(receipts) > 2:
        raise ValueError("BonePile Head has bad receipts")
    return Head(
        head=str(value.get("head", "")).upper(),
        key=str(value.get("key", "")),
        bones=int(value.get("bones", -1)),
        tag=Tag(str(tag.get("parent", "")), str(tag.get("child", ""))),
        locksign=str(value.get("locksign", "")),
        receipts=tuple(BoneFromWire(item) for item in receipts),
        clawcount=value.get("clawcount"),
    )


def BonePileToWire(pile: BonePile) -> dict[str, object]:
    return {head: HeadToWire(cell) for head, cell in pile.items()}


def BonePileFromWire(value: object, heads: Iterable[str]) -> BonePile:
    expected = tuple(heads)
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("BonePile has the wrong heads")
    pile = {head: HeadFromWire(value[head]) for head in expected}
    if any(pile[head].head != head for head in expected):
        raise ValueError("BonePile Cell label does not match its slot")
    return pile


class BoneYard:

    def __init__(
        self,
        ring: str,
        *,
        HeadCountIn: Optional[Callable[[object], bool]] = None,
        NoticeOut: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.mask = hashlib.sha256(str(ring).encode("utf-8")).digest()
        self.HeadCountIn = HeadCountIn
        self.NoticeOut = NoticeOut

        self.mouthcount = 0
        self.bindport: Optional[int] = None
        self.sock: Optional[socket.socket] = None

        self.heads: tuple[str, ...] = ()
        self.expected: set[str] = set()
        self.count = 0
        self.head = ""
        self.ready = False

        self.CatacombIn: Optional[Callable[[Bone], Result]] = None
        self.BonePileIn: Optional[Callable[[BonePile], Result]] = None
        self.BonePileOut: Optional[Callable[[], BonePile]] = None

        self.seenorder: deque[tuple[object, ...]] = deque()
        self.seen: set[tuple[object, ...]] = set()

    def Open(self, count: int) -> None:
        if self.sock is not None:
            return
        self.mouthcount = max(1, int(count))
        lasterror: Optional[Exception] = None
        for port in self.Mouths():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            try:
                sock.bind((Host, port))
                sock.setblocking(False)
                self.bindport = port
                self.sock = sock
                return
            except OSError as exc:
                lasterror = exc
                sock.close()
        raise RuntimeError(f"No clean mouth available in {self.Mouths()}.") from lasterror

    def Mouths(self) -> list[int]:
        return [BonePilePort + index for index in range(self.mouthcount)]

    def Peers(self) -> list[int]:
        return [port for port in self.Mouths() if port != self.bindport]

    def Close(self) -> None:
        sock = self.sock
        self.sock = None
        if sock is None:
            return
        try:
            sock.close()
        except Exception:
            pass

    def Encrypt(self, message: dict[str, object]) -> bytes:
        body = json.dumps(message, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return bytes(byte ^ self.mask[index % len(self.mask)] for index, byte in enumerate(body))

    def Decrypt(self, raw: bytes) -> dict[str, object]:
        body = bytes(byte ^ self.mask[index % len(self.mask)] for index, byte in enumerate(raw))
        message = json.loads(body.decode("utf-8"))
        if not isinstance(message, dict):
            raise TypeError("packet must decode to dict")
        return message

    def Send(self, message: dict[str, object]) -> None:
        if self.sock is None:
            return
        raw = self.Encrypt(message)
        for port in self.Peers():
            for _shot in range(Burst):
                try:
                    self.sock.sendto(raw, (Host, port))
                except OSError:
                    pass

    def Receive(self) -> list[dict[str, object]]:
        if self.sock is None:
            return []
        messages = []
        while True:
            try:
                raw, address = self.sock.recvfrom(65535)
            except (BlockingIOError, OSError):
                break
            if address[0] != Host:
                continue
            try:
                messages.append(self.Decrypt(raw))
            except Exception:
                continue
        return messages

    def HeadCount(self, headcount: object) -> None:
        self.Send({"type": "HEADCOUNT", "headcount": headcount})

    def Attach(
        self,
        heads: Iterable[str],
        head: str,
        *,
        CatacombIn: Callable[[Bone], Result],
        BonePileIn: Callable[[BonePile], Result],
        BonePileOut: Callable[[], BonePile],
    ) -> None:
        heads = tuple(str(item).upper() for item in heads)
        head = str(head).upper()
        if head not in heads:
            raise ValueError("local head is not in this BoneYard")
        self.heads = heads
        self.expected = set(heads)
        self.count = len(heads)
        self.head = head
        self.CatacombIn = CatacombIn
        self.BonePileIn = BonePileIn
        self.BonePileOut = BonePileOut
        self.ready = True

    def SendBonePile(self, pile: Optional[BonePile] = None) -> None:
        if not self.ready or self.BonePileOut is None:
            return
        pile = self.BonePileOut() if pile is None else pile
        if set(pile) != self.expected:
            return
        self.Send({
            "type": "BONEPILE",
            "count": self.count,
            "bonepile": BonePileToWire(pile),
        })

    def Hunger(self) -> None:
        if self.ready:
            self.Send({"type": "HUNGER", "count": self.count, "head": self.head})

    def Seen(self, bone: Bone) -> bool:
        return BoneKey(bone) in self.seen

    def Remember(self, bone: Bone) -> None:
        key = BoneKey(bone)
        if key in self.seen:
            return
        self.seen.add(key)
        self.seenorder.append(key)
        while len(self.seenorder) > CacheLimit:
            self.seen.discard(self.seenorder.popleft())

    def Catacomb(self, bone: Bone, result: Result) -> None:
        if not self.ready or not isinstance(bone, Bone):
            return
        if result.status == "GROWL":
            self.Remember(bone)
            self.Send({"type": "BONE", "count": self.count, "head": self.head, "bone": BoneToWire(bone)})
            return
        if result.snapshot is not None:
            self.SendBonePile(result.snapshot)
        if not result.changed:
            return
        self.Remember(bone)
        self.Send({"type": "BONE", "count": self.count, "head": self.head, "bone": BoneToWire(bone)})
        if result.reproject:
            self.SendBonePile()

    def Pump(self) -> bool:
        redraw = False
        for message in self.Receive():
            redraw = self.Handle(message) or redraw
        return redraw

    def Handle(self, message: dict[str, object]) -> bool:
        kind = str(message.get("type", "")).upper()
        if kind == "HEADCOUNT":
            return bool(self.HeadCountIn and self.HeadCountIn(message.get("headcount")))
        if not self.ready:
            return False
        try:
            if int(message.get("count", 0)) != self.count:
                return False
        except Exception:
            return False
        if kind == "BONE":
            try:
                bone = BoneFromWire(message.get("bone"))
            except Exception:
                if self.NoticeOut:
                    self.NoticeOut("BAD BONE")
                return True
            if self.Seen(bone):
                return False
            if self.CatacombIn is None:
                return False
            result = self.CatacombIn(bone)
            if result.status == "BAD BONE":
                if self.NoticeOut:
                    self.NoticeOut("BAD BONE")
                return True
            if result.status == "HUNGRY":
                return True
            self.Remember(bone)
            return False if result.status in ("IDEMPOTENT", "DOGHOUSE") else bool(result.changed)

        if kind == "BONEPILE":
            try:
                pile = BonePileFromWire(message.get("bonepile"), self.heads)
            except Exception:
                if self.NoticeOut:
                    self.NoticeOut("BAD BONEPILE")
                return True
            if self.BonePileIn is None:
                return False
            result = self.BonePileIn(pile)
            if result.status == "LOCKED":
                return False
            if result.status == "IDEMPOTENT":
                return True
            if result.status == "BAD BONEPILE":
                if self.NoticeOut:
                    self.NoticeOut("BAD BONEPILE")
                return True
            return bool(result.changed)

        if kind == "HUNGER":
            self.SendBonePile()
        return False
