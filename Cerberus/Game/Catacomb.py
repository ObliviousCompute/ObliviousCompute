from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from typing import Callable, Iterable, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

Uppercase = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BonesPerHead = 11
HashBytes = 32
PublicKeyBytes = 32
SignatureBytes = 64
HashHexWidth = HashBytes * 2
PublicKeyHexWidth = PublicKeyBytes * 2
SignatureHexWidth = SignatureBytes * 2
ZeroHash = "0" * HashHexWidth
ZeroSign = "0" * SignatureHexWidth


def HexShape(value: object, width: int) -> bool:
    if not isinstance(value, str) or len(value) != width:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True

def ValidHash(value: object) -> bool:
    return HexShape(value, HashHexWidth)

def ValidKey(value: object) -> bool:
    return HexShape(value, PublicKeyHexWidth)

def ValidSign(value: object) -> bool:
    return HexShape(value, SignatureHexWidth)

def HashBody(domain: str, *parts: object) -> bytes:
    return "|".join((str(domain), *(str(part) for part in parts))).encode("utf-8")

def HashHex(domain: str, *parts: object) -> str:
    return hashlib.sha256(HashBody(domain, *parts)).hexdigest()

def StateKey(secret: str) -> Ed25519PrivateKey:
    seed = hashlib.sha256(
        f"Cerberus::Dog::V1::{str(secret)}".encode("utf-8")
    ).digest()
    return Ed25519PrivateKey.from_private_bytes(seed)

def PublicKeyHex(privatekey: Ed25519PrivateKey) -> str:
    return privatekey.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    ).hex()

def SignDigest(privatekey: Ed25519PrivateKey, digesthex: str) -> str:
    if not ValidHash(digesthex):
        raise ValueError("digest must be a SHA-256 hex string")
    return privatekey.sign(bytes.fromhex(digesthex)).hex()

def VerifyDigest(keyhex: str, digesthex: str, signhex: str) -> None:
    if not ValidKey(keyhex):
        raise ValueError("bad Ed25519 public key")
    if not ValidHash(digesthex):
        raise ValueError("bad SHA-256 digest")
    if not ValidSign(signhex):
        raise ValueError("bad Ed25519 signature")
    key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(keyhex))
    try:
        key.verify(bytes.fromhex(signhex), bytes.fromhex(digesthex))
    except InvalidSignature as exc:
        raise ValueError("signature verification failed") from exc


@dataclass(frozen=True)
class Tag:
    parent: str
    child: str

    def __post_init__(self) -> None:
        if not ValidHash(self.parent):
            raise ValueError("tag.parent must be a 32-byte SHA-256 hash")
        if not ValidHash(self.child):
            raise ValueError("tag.child must be a 32-byte SHA-256 hash")

@dataclass(frozen=True)
class Bone:
    head: str
    key: str
    target: str
    bones: int
    tag: Tag
    locksign: str
    sign: str

    def __post_init__(self) -> None:
        if not isinstance(self.head, str) or len(self.head) != 1 or self.head not in Uppercase:
            raise ValueError("bone.head must be A-Z")
        if not ValidKey(self.key):
            raise ValueError("bone.key must be a 32-byte Ed25519 public key")
        if not isinstance(self.target, str) or len(self.target) != 1 or self.target not in Uppercase:
            raise ValueError("bone.target must be A-Z")
        if not isinstance(self.bones, int):
            raise TypeError("bone.bones must be int")
        if self.bones <= 0:
            raise ValueError("bone.bones must be positive")
        if not isinstance(self.tag, Tag):
            raise TypeError("bone.tag must be Tag")
        if not ValidSign(self.locksign):
            raise ValueError("bone.locksign must be a 64-byte Ed25519 signature")
        if not ValidSign(self.sign):
            raise ValueError("bone.sign must be a 64-byte Ed25519 signature")

@dataclass(frozen=True)
class Head:
    head: str
    key: str
    bones: int
    tag: Tag
    locksign: str
    receipts: tuple[Bone, ...] = ()
    clawcount: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.head, str) or len(self.head) != 1 or self.head not in Uppercase:
            raise ValueError("cell.head must be A-Z")
        if not ValidKey(self.key):
            raise ValueError("cell.key must be a 32-byte Ed25519 public key")
        if not isinstance(self.bones, int):
            raise TypeError("cell.bones must be int")
        if self.bones < 0:
            raise ValueError("cell.bones cannot be negative")
        if not isinstance(self.tag, Tag):
            raise TypeError("cell.tag must be Tag")
        if not ValidSign(self.locksign):
            raise ValueError("cell.locksign must be a 64-byte Ed25519 signature")
        if not isinstance(self.receipts, tuple) or len(self.receipts) > 2:
            raise ValueError("cell.receipts must contain zero, one, or two Bones")
        if any(not isinstance(receipt, Bone) for receipt in self.receipts):
            raise TypeError("cell.receipts must contain Bones")
        if self.clawcount is not None:
            if not isinstance(self.clawcount, int) or self.clawcount < 0:
                raise ValueError("cell.clawcount must be a non-negative int or None")

BonePile = dict[str, Head]

@dataclass(frozen=True)
class Result:
    status: str
    changed: bool = False
    reproject: bool = False
    bone: Optional[Bone] = None
    snapshot: Optional[BonePile] = None


def LockHash(tag: Tag) -> str:
    return HashHex("CERBERUS::LOCK::V1", tag.parent, tag.child)

def GenesisChild(head: str, key: str) -> str:
    return HashHex("CERBERUS::GENESIS::V1", head, key, BonesPerHead, ZeroHash)

def ChildHash(head: str, key: str, parent: str, target: str, bones: int) -> str:
    return HashHex(
        "CERBERUS::CHILD::V1",
        head,
        key,
        parent,
        target,
        int(bones),
    )

def ReceiptHash(bone: Bone) -> str:
    return HashHex(
        "CERBERUS::BONE::V1",
        bone.head,
        bone.key,
        bone.target,
        int(bone.bones),
        bone.tag.parent,
        bone.tag.child,
        bone.locksign,
    )

def ForkHash(*receipts: Bone) -> str:
    pair = CanonicalReceipts(*receipts)
    if len(pair) != 2 or pair[0].head != pair[1].head or pair[0].tag.parent != pair[1].tag.parent:
        raise ValueError("fork hash needs two same-parent sibling Bones")
    return HashHex(
        "CERBERUS::FORK::V1",
        pair[0].head,
        pair[0].tag.parent,
        pair[0].tag.child,
        pair[1].tag.child,
    )

def BonePileHash(projector: str, pile: BonePile) -> str:
    parts: list[object] = [str(projector).upper()]
    for head in sorted(pile):
        cell = pile[head]
        parts.extend((
            cell.head,
            cell.key,
            int(cell.bones),
            cell.tag.parent,
            cell.tag.child,
            cell.locksign,
            "" if cell.clawcount is None else int(cell.clawcount),
            len(cell.receipts),
        ))
        for receipt in cell.receipts:
            parts.extend((ReceiptHash(receipt), receipt.sign))
    return HashHex("CERBERUS::BONEPILE::V1", *parts)


def VerifyBoneProof(bone: Bone) -> Bone:
    expectedchild = ChildHash(
        bone.head,
        bone.key,
        bone.tag.parent,
        bone.target,
        bone.bones,
    )
    if bone.tag.child != expectedchild:
        raise ValueError("Bone child hash does not match its contents")
    VerifyDigest(bone.key, LockHash(bone.tag), bone.locksign)
    VerifyDigest(bone.key, ReceiptHash(bone), bone.sign)
    return bone

def CanonicalReceipts(*receipts: Bone) -> tuple[Bone, ...]:
    unique: dict[str, Bone] = {}
    for receipt in receipts:
        unique[ReceiptHash(receipt)] = receipt
    ordered = tuple(sorted(unique.values(), key=lambda item: (item.tag.child, ReceiptHash(item))))
    if len(ordered) > 2:
        raise ValueError("a convicted Cell retains exactly two sibling receipts")
    return ordered
def VerifyCellLock(cell: Head) -> Head:
    if cell.locksign == ZeroSign and not cell.receipts and cell.clawcount is None and cell.tag == Tag(ZeroHash, GenesisChild(cell.head, cell.key)):
        return cell
    VerifyDigest(cell.key, LockHash(cell.tag), cell.locksign)
    return cell

class Catacomb:
    def __init__(
        self,
        heads: Iterable[str],
        head: str,
        secret: str,
        *,
        GuardianOut: Optional[Callable[[BonePile, Result], None]] = None,
        BoneYardOut: Optional[Callable[[Bone, Result], None]] = None,
        HungerOut: Optional[Callable[[], None]] = None,
    ) -> None:
        ordered = tuple(str(h).upper() for h in heads)
        if not ordered:
            raise ValueError("Cerberus needs at least one head")
        if len(set(ordered)) != len(ordered):
            raise ValueError("head identities must be unique")
        if any(len(h) != 1 or h not in Uppercase for h in ordered):
            raise ValueError("heads must be A-Z")

        self.heads = ordered
        self.expected = set(ordered)
        self.head = str(head).upper()
        if self.head not in self.expected:
            raise ValueError("local head is not in this Cerberus")

        self.privatekey = StateKey(secret)
        self.publickey = PublicKeyHex(self.privatekey)
        self.Authority: dict[str, str] = {}

        self.GuardianOut = GuardianOut
        self.BoneYardOut = BoneYardOut
        self.HungerOut = HungerOut

        self.BuriedBonePile: BonePile = {self.head: self.GenesisCell}
        self.SortingBonePile: Optional[BonePile] = None
        self.Hungry = False

    @property
    def GenesisCell(self) -> Head:
        tag = Tag(ZeroHash, GenesisChild(self.head, self.publickey))
        return Head(self.head, self.publickey, BonesPerHead, tag, ZeroSign)

    def FreezeAuthority(self, pile: BonePile) -> None:
        authority = {head: pile[head].key for head in self.heads}
        if authority[self.head] != self.publickey:
            raise ValueError("Genesis tried to replace the local Head key")
        if len(set(authority.values())) != len(authority):
            raise ValueError("two Heads cannot share one public key")
        self.Authority = authority

    def CopyBonePile(self, source: Optional[BonePile] = None) -> BonePile:
        return dict(self.BuriedBonePile if source is None else source)

    def SignBonePile(self, pile: BonePile) -> str:
        return SignDigest(self.privatekey, BonePileHash(self.head, pile))

    def VerifyBonePile(self, pile: BonePile) -> BonePile:
        if not isinstance(pile, dict):
            raise TypeError("BonePile must be dict[str, Head]")
        if set(pile) != self.expected:
            raise ValueError("BonePile has missing or unknown heads")

        keys: list[str] = []
        for head in self.heads:
            cell = pile[head]
            if not isinstance(cell, Head):
                raise TypeError("BonePile values must be Head")
            if cell.head != head:
                raise ValueError("BonePile Cell label does not match its slot")
            VerifyCellLock(cell)

            receipts = CanonicalReceipts(*cell.receipts)
            if receipts != cell.receipts:
                raise ValueError("Cell receipts are not canonical")
            if cell.clawcount is not None and len(receipts) != 2:
                raise ValueError("clawcount requires a sibling pair")
            if cell.clawcount and cell.bones != 0:
                raise ValueError("a clawed Head must be empty")
            if receipts:
                for receipt in receipts:
                    VerifyBoneProof(receipt)
                    if receipt.head != head or receipt.key != cell.key:
                        raise ValueError("Cell receipt does not belong to its Head")
                    if receipt.target not in self.expected or receipt.target == head:
                        raise ValueError("Cell receipt names an invalid target")
                parents = {receipt.tag.parent for receipt in receipts}
                children = {receipt.tag.child for receipt in receipts}
                if len(receipts) == 2 and (len(parents) != 1 or len(children) != 2):
                    raise ValueError("two Cell receipts must be conflicting siblings")
                canonical = receipts[0]
                if len(receipts) == 2 and cell.clawcount == 0:
                    applied = any(
                        cell.tag == receipt.tag and cell.locksign == receipt.locksign
                        for receipt in receipts
                    )
                    untouched = cell.tag.child == receipts[0].tag.parent
                    if not applied and not untouched:
                        raise ValueError("virgin clawback surface does not expose its fork")
                elif cell.tag != canonical.tag or cell.locksign != canonical.locksign:
                    raise ValueError("Cell surface must follow its lower recorded child")
            else:
                genesis = Tag(parent=ZeroHash, child=GenesisChild(head, cell.key))
                if cell.tag != genesis:
                    raise ValueError("a receipt-free Cell must still be at Genesis")

            keys.append(cell.key)
            if self.Authority and cell.key != self.Authority[head]:
                raise ValueError("BonePile tried to replace a frozen Head key")

        if len(set(keys)) != len(keys):
            raise ValueError("BonePile reuses one public key for multiple Heads")
        if pile[self.head].key != self.publickey:
            raise ValueError("BonePile does not contain our own public key")

        total = sum(pile[head].bones for head in self.heads)
        expected = BonesPerHead * len(self.heads)
        if total != expected:
            raise ValueError(f"BonePile invariant violated: {total} != {expected}")
        return pile

    @property
    def BonePile(self) -> BonePile:
        return self.CopyBonePile(self.BuriedBonePile)

    def Seed(self, pile: BonePile) -> Result:
        try:
            candidate = self.CopyBonePile(pile)
            self.VerifyBonePile(candidate)
            if not self.Authority:
                self.FreezeAuthority(candidate)
                self.VerifyBonePile(candidate)
        except Exception:
            return Result(status="BAD BONEPILE")

        if candidate == self.BuriedBonePile:
            return Result(status="IDEMPOTENT")

        self.BuriedBonePile = candidate
        self.SortingBonePile = None
        result = Result(status="BURIED", changed=True)
        if self.GuardianOut is not None:
            self.GuardianOut(self.CopyBonePile(), result)
        return result

    def Hunger(self) -> Result:
        self.Hungry = True
        if self.HungerOut is not None:
            self.HungerOut()
        return Result(status="HUNGRY")

    def ReceiveBonePile(self, pile: BonePile, projector: str) -> Result:
        try:
            candidate = self.CopyBonePile(pile)
            self.VerifyBonePile(candidate)
        except Exception:
            return Result(status="BAD BONEPILE")

        projector = str(projector).upper()
        if projector in self.BuriedBonePile and self.Doghouse(self.BuriedBonePile[projector]):
            return Result(status="DOGHOUSE")

        virgins = [head for head in self.heads if candidate[head].clawcount == 0]
        if virgins:
            if len(virgins) != 1:
                return Result(status="BAD BONEPILE")
            dirty = virgins[0]
            if projector == dirty:
                return Result(status="DOGHOUSE")
            release = self.Clawback(candidate, dirty)
            if release is None:
                return Result(status="BAD BONEPILE")
            repaired = release

            mydogs = self.Dogs(self.BuriedBonePile)
            theirdogs = self.Dogs(repaired)
            if not mydogs <= theirdogs:
                return Result(status="LOCKED")

            local = self.BuriedBonePile[dirty]
            if self.Doghouse(local):
                incomingrank = (
                    ForkHash(*candidate[dirty].receipts),
                    int(repaired[dirty].clawcount or 0),
                    BonePileHash("", repaired),
                )
                localrank = (
                    ForkHash(*local.receipts),
                    int(local.clawcount or 0),
                    BonePileHash("", self.BuriedBonePile),
                )
                if incomingrank >= localrank:
                    return Result(status="LOCKED")

            self.Hungry = False
            self.BuriedBonePile = repaired
            self.SortingBonePile = None
            result = Result(status="CLAWED", changed=True)
            if self.GuardianOut is not None:
                self.GuardianOut(self.CopyBonePile(), result)
            return result

        if not self.Hungry:
            return Result(status="LOCKED")

        mydogs = self.Dogs(self.BuriedBonePile)
        theirdogs = self.Dogs(candidate)
        if not mydogs <= theirdogs:
            return Result(status="LOCKED")

        self.Hungry = False
        if candidate == self.BuriedBonePile:
            result = Result(status="IDEMPOTENT")
            if self.GuardianOut is not None:
                self.GuardianOut(self.CopyBonePile(), result)
            return result

        self.BuriedBonePile = candidate
        self.SortingBonePile = None

        result = Result(status="BURIED", changed=True)
        if self.GuardianOut is not None:
            self.GuardianOut(self.CopyBonePile(), result)
        return result


    def Mint(self, target: str, bones: int) -> Bone:
        if not self.Authority:
            raise ValueError("Genesis authority has not been frozen")

        target = str(target).upper()
        bones = int(bones)
        current = self.BuriedBonePile[self.head]
        if current.key != self.publickey:
            raise ValueError("local Cell key does not match local private key")
        if self.Doghouse(current):
            raise ValueError("Head is in the Doghouse")
        if target in self.BuriedBonePile and self.Doghouse(self.BuriedBonePile[target]):
            raise ValueError("target Head is in the Doghouse")

        parent = current.tag.child
        child = ChildHash(self.head, self.publickey, parent, target, bones)
        tag = Tag(parent=parent, child=child)
        locksign = SignDigest(self.privatekey, LockHash(tag))
        proto = Bone(
            head=self.head,
            key=self.publickey,
            target=target,
            bones=bones,
            tag=tag,
            locksign=locksign,
            sign=ZeroSign,
        )
        return replace(proto, sign=SignDigest(self.privatekey, ReceiptHash(proto)))

    def VerifyBone(self, bone: Bone) -> Bone:
        if not isinstance(bone, Bone):
            raise TypeError("expected Bone")
        if bone.head not in self.expected:
            raise ValueError("Bone names an unknown head")
        if bone.target not in self.expected:
            raise ValueError("Bone names an unknown target")
        if bone.head == bone.target:
            raise ValueError("a Head cannot give bones to itself")
        if bone.bones <= 0:
            raise ValueError("Bone amount must be positive")
        if not self.Authority:
            raise ValueError("Head authority is not established")
        if bone.key != self.Authority[bone.head]:
            raise ValueError("Bone public key does not own that Head")

        return VerifyBoneProof(bone)

    def Guardian(self, target: str, bones: int) -> Result:
        try:
            bone = self.Mint(target, bones)
        except Exception:
            return Result(status="BAD BONE")
        return self.ReceiveBone(bone)

    def BoneYard(self, bone: Bone) -> Result:
        return self.ReceiveBone(bone)


    @staticmethod
    def Recorded(cell: Head, bone: Bone) -> bool:
        receiptid = ReceiptHash(bone)
        return any(ReceiptHash(receipt) == receiptid for receipt in cell.receipts)

    @staticmethod
    def Doghouse(cell: Head) -> bool:
        return bool(cell.clawcount)

    def Dogs(self, pile: BonePile) -> set[str]:
        return {head for head in self.heads if self.Doghouse(pile[head])}

    def RaiseForClawback(
        self,
        candidate: BonePile,
        head: str,
        needed: int,
        unwound: set[str],
        trail: set[str],
    ) -> bool:
        cell = candidate[head]
        if cell.bones >= needed:
            return True
        if head in trail or not cell.receipts:
            return False

        nexttrail = set(trail)
        nexttrail.add(head)
        for edge in reversed(cell.receipts):
            edgeid = ReceiptHash(edge)
            if edgeid in unwound or edge.target not in candidate or edge.target == head:
                continue
            before = self.CopyBonePile(candidate)
            beforeunwound = set(unwound)
            if not self.RaiseForClawback(candidate, edge.target, edge.bones, unwound, nexttrail):
                candidate.clear()
                candidate.update(before)
                unwound.clear()
                unwound.update(beforeunwound)
                continue
            target = candidate[edge.target]
            if target.bones < edge.bones:
                continue
            candidate[edge.target] = replace(target, bones=target.bones - edge.bones)
            candidate[head] = replace(candidate[head], bones=candidate[head].bones + edge.bones)
            unwound.add(edgeid)
            if candidate[head].bones >= needed:
                return True
        return False

    def Clawback(self, virgin: BonePile, dirty: str) -> Optional[BonePile]:
        candidate = self.CopyBonePile(virgin)
        cell = candidate[dirty]
        pair = CanonicalReceipts(*cell.receipts)
        if cell.clawcount != 0 or len(pair) != 2:
            return None

        applied = next(
            (receipt for receipt in pair if cell.tag == receipt.tag and cell.locksign == receipt.locksign),
            None,
        )
        if applied is None and cell.tag.child != pair[0].tag.parent:
            return None
        estatebefore = cell.bones + (applied.bones if applied is not None else 0)
        if sum(receipt.bones for receipt in pair) <= estatebefore:
            return None

        unwound: set[str] = set()
        if applied is not None:
            if not self.RaiseForClawback(candidate, applied.target, applied.bones, unwound, {dirty}):
                return None
            victim = candidate[applied.target]
            if victim.bones < applied.bones:
                return None
            candidate[applied.target] = replace(victim, bones=victim.bones - applied.bones)
            source = candidate[dirty]
            candidate[dirty] = replace(source, bones=source.bones + applied.bones)

        source = candidate[dirty]
        estate = int(source.bones)
        canonical = pair[0]
        claws = set(unwound)
        if applied is not None:
            claws.add(ReceiptHash(applied))
        candidate[dirty] = Head(
            head=source.head,
            key=source.key,
            bones=0,
            tag=canonical.tag,
            locksign=canonical.locksign,
            receipts=pair,
            clawcount=max(1, len(claws)),
        )

        q, r = divmod(estate, len(pair))
        for index, receipt in enumerate(pair):
            share = q + (1 if index < r else 0)
            if share:
                target = candidate[receipt.target]
                candidate[receipt.target] = replace(target, bones=target.bones + share)

        try:
            self.VerifyBonePile(candidate)
        except Exception:
            return None
        return candidate

    def DirtyDog(self, first: Bone, second: Bone) -> Result:
        pair = CanonicalReceipts(first, second)
        if len(pair) != 2 or pair[0].tag.parent != pair[1].tag.parent:
            return Result(status="BAD BONE", bone=second)

        virgin = self.CopyBonePile()
        source = virgin[second.head]
        virgin[second.head] = Head(
            head=source.head,
            key=source.key,
            bones=source.bones,
            tag=source.tag,
            locksign=source.locksign,
            receipts=pair,
            clawcount=0,
        )
        try:
            self.VerifyBonePile(virgin)
        except Exception:
            return Result(status="BAD BONE", bone=second)

        release = self.Clawback(virgin, second.head)
        if release is None:
            self.Hungry = True
            if self.HungerOut is not None:
                self.HungerOut()
            result = Result(status="HUNGRY", bone=second, snapshot=virgin)
            if self.BoneYardOut is not None:
                self.BoneYardOut(second, result)
            return result

        repaired = release
        self.SortingBonePile = repaired
        return self.Bury(second, snapshot=virgin)

    def ReceiveBone(self, bone: Bone) -> Result:
        try:
            self.VerifyBone(bone)
        except Exception:
            return Result(status="BAD BONE", bone=bone if isinstance(bone, Bone) else None)

        current = self.BuriedBonePile[bone.head]

        if self.Recorded(current, bone):
            return Result(status="IDEMPOTENT", bone=bone)

        if self.Doghouse(current) or self.Doghouse(self.BuriedBonePile[bone.target]):
            return Result(status="DOGHOUSE", bone=bone)

        held = current.receipts[0] if len(current.receipts) == 1 else None

        # ================= LINCHPIN ================= #
        normal = bone.tag.parent == current.tag.child
        sibling = bool(
            held is not None
            and bone.tag.parent == held.tag.parent
            and bone.tag.child != held.tag.child
        )
        if not normal and not sibling:
            return Result(status="BAD BONE", bone=bone)

        if bone.bones > current.bones:
            if sibling and held is not None:
                return self.DirtyDog(held, bone)
            result = Result(status="GROWL", bone=bone)
            if self.BoneYardOut is not None:
                self.BoneYardOut(bone, result)
            return result
        # ============================================ #

        self.SortingBonePile = self.CopyBonePile()
        source = self.SortingBonePile[bone.head]
        target = self.SortingBonePile[bone.target]

        if sibling and held is not None:
            receipts = CanonicalReceipts(held, bone)
            canonical = receipts[0]
            chosentag = canonical.tag
            chosenlock = canonical.locksign
            reproject = True
        else:
            receipts = (bone,)
            chosentag = bone.tag
            chosenlock = bone.locksign
            reproject = False

        self.SortingBonePile[bone.head] = Head(
            head=source.head,
            key=source.key,
            bones=source.bones - bone.bones,
            tag=chosentag,
            locksign=chosenlock,
            receipts=receipts,
        )
        self.SortingBonePile[bone.target] = replace(
            target,
            bones=target.bones + bone.bones,
        )

        try:
            self.VerifyBonePile(self.SortingBonePile)
        except Exception:
            self.SortingBonePile = None
            return Result(status="BAD BONE", bone=bone)

        return self.Bury(bone, reproject=reproject)


    def Bury(self, bone: Bone, *, reproject: bool = False, snapshot: Optional[BonePile] = None) -> Result:
        if self.SortingBonePile is None:
            return Result(status="BAD BONE", bone=bone)

        candidate = self.CopyBonePile(self.SortingBonePile)
        try:
            self.VerifyBonePile(candidate)
        except Exception:
            self.SortingBonePile = None
            return Result(status="BAD BONE", bone=bone)

        if candidate == self.BuriedBonePile:
            self.SortingBonePile = None
            return Result(status="IDEMPOTENT", bone=bone)

        self.BuriedBonePile = candidate
        self.SortingBonePile = None

        result = Result(
            status="BURIED",
            changed=True,
            reproject=bool(reproject),
            bone=bone,
            snapshot=self.CopyBonePile(snapshot) if snapshot is not None else None,
        )

        if self.GuardianOut is not None:
            self.GuardianOut(self.CopyBonePile(), result)
        if self.BoneYardOut is not None:
            self.BoneYardOut(bone, result)
        return result
