from __future__ import annotations
from dataclasses import dataclass, replace
import hashlib
from typing import Callable, Iterable, Optional
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
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
class BonePile(dict[str, Head]):
    def __init__(self, source: Optional[dict[str, Head]] = None) -> None:
        super().__init__({} if source is None else source)
    def clone(self) -> "BonePile":
        return BonePile(dict(self))
class Maul(BonePile):
    def __init__(self, source: BonePile) -> None:
        super().__init__(dict(source))
        self.changedpairs: set[str] = set()
        self.frontier: dict[str, tuple[Bone, ...]] = {}
        self.incoming: Optional[BonePile] = None
        self.dirt: set[str] = set()
        self.forward = False
        self.solvent: set[str] = set()
        self.dirtydogs: set[str] = set()
        self.dogpile: set[str] = set()
        self.equivocation = False
        self.changed = False
        self.ossified = False
        self.bone: Optional[Bone] = None
@dataclass(frozen=True)
class Result:
    status: str
    changed: bool = False
    reproject: bool = False
    bone: Optional[Bone] = None
def LockHash(tag: Tag) -> str:
    return HashHex("CERBERUS::LOCK::V1", tag.parent, tag.child)
def GenesisChild(head: str, key: str) -> str:
    return HashHex("CERBERUS::GENESIS::V1", head, key, BonesPerHead, ZeroHash)
def ChildHash(head: str, key: str, parent: str, target: str, bones: int) -> str:
    return HashHex(
        "CERBERUS::CHILD::V1", head, key, parent, target, int(bones),
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
        raise ValueError("a Head retains at most two canonical sibling receipts")
    return ordered
def FreshBones(*receipts: Bone) -> tuple[Bone, Bone]:
    """Return the two freshest siblings: lowest child hashes, never arrival order."""
    unique: dict[str, Bone] = {}
    for receipt in receipts:
        VerifyBoneProof(receipt)
        unique[ReceiptHash(receipt)] = receipt
    ordered = tuple(sorted(unique.values(), key=lambda item: (item.tag.child, ReceiptHash(item))))
    if len(ordered) < 2:
        raise ValueError("fork evidence needs at least two distinct Bones")
    head = ordered[0].head
    key = ordered[0].key
    parent = ordered[0].tag.parent
    if any(item.head != head or item.key != key or item.tag.parent != parent for item in ordered):
        raise ValueError("fork evidence must share one Head, key, and parent")
    if len({item.tag.child for item in ordered}) != len(ordered):
        raise ValueError("fork evidence children must be distinct")
    return (ordered[0], ordered[1])
def FreshHashes(*receipts: Bone) -> tuple[str, str]:
    pair = FreshBones(*receipts)
    return (pair[0].tag.child, pair[1].tag.child)
def VerifyCellLock(cell: Head) -> Head:
    if cell.locksign == ZeroSign and not cell.receipts and cell.tag == Tag(ZeroHash, GenesisChild(cell.head, cell.key)):
        return cell
    VerifyDigest(cell.key, LockHash(cell.tag), cell.locksign)
    return cell
class Catacomb:
    def __init__(
        self,
        heads: Iterable[str],
        head: str,
        publickey: str,
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
        if not ValidKey(publickey):
            raise ValueError("local Head needs a valid Ed25519 public key")
        self.publickey = str(publickey)
        self.Authority: dict[str, str] = {}
        self.GuardianOut = GuardianOut
        self.BoneYardOut = BoneYardOut
        self.HungerOut = HungerOut
        self.ProjectOut: Optional[Callable[[BonePile], None]] = None
        self.BuriedBonePile = BonePile({self.head: self.GenesisCell})
        self.Strays: dict[str, Bone] = {}
        self.Hungry = False
    @property
    def GenesisCell(self) -> Head:
        tag = Tag(ZeroHash, GenesisChild(self.head, self.publickey))
        return Head(self.head, self.publickey, BonesPerHead, tag, ZeroSign)
    def CopyBonePile(self, source: Optional[dict[str, Head]] = None) -> BonePile:
        source = self.BuriedBonePile if source is None else source
        return BonePile(dict(source))
    def FreezeAuthority(self, pile: BonePile) -> None:
        authority = {head: pile[head].key for head in self.heads}
        if authority[self.head] != self.publickey:
            raise ValueError("Genesis tried to replace the local Head key")
        if len(set(authority.values())) != len(authority):
            raise ValueError("two Heads cannot share one public key")
        self.Authority = authority
    def NinetyNine(self, pile: dict[str, Head]) -> dict[str, Head]:
        if set(pile) != self.expected:
            raise ValueError("BonePile has missing or unknown heads")
        if any(not isinstance(pile[head], Head) or pile[head].bones < 0 for head in self.heads):
            raise ValueError("BonePile contains an invalid/negative Head")
        total = sum(int(pile[head].bones) for head in self.heads)
        expected = BonesPerHead * len(self.heads)
        if total != expected:
            raise ValueError(f"NinetyNine invariant violated: {total} != {expected}")
        return pile
    def Check(self, pile: BonePile) -> BonePile:
        if not isinstance(pile, dict):
            raise TypeError("BonePile must be dict-like")
        if set(pile) != self.expected:
            raise ValueError("BonePile has missing or unknown heads")
        keys: list[str] = []
        for head in self.heads:
            cell = pile[head]
            if not isinstance(cell, Head):
                raise TypeError("BonePile values must be Head")
            if cell.head != head:
                raise ValueError("BonePile Head label does not match its slot")
            VerifyCellLock(cell)
            receipts = CanonicalReceipts(*cell.receipts)
            if receipts != cell.receipts:
                raise ValueError("Head receipts are not canonical")
            if receipts:
                for receipt in receipts:
                    self.VerifyBone(receipt)
                    if receipt.head != head or receipt.key != cell.key:
                        raise ValueError("receipt does not belong to its Head")
                if len(receipts) == 2:
                    parents = {receipt.tag.parent for receipt in receipts}
                    children = {receipt.tag.child for receipt in receipts}
                    if len(parents) != 1 or len(children) != 2:
                        raise ValueError("two receipts must be conflicting siblings")
                canonical = receipts[0]
                if cell.tag != canonical.tag or cell.locksign != canonical.locksign:
                    raise ValueError("Head surface must follow its freshest retained child")
            else:
                genesis = Tag(parent=ZeroHash, child=GenesisChild(head, cell.key))
                if cell.tag != genesis:
                    raise ValueError("receipt-free Head must be at Genesis")
            keys.append(cell.key)
            if self.Authority and cell.key != self.Authority[head]:
                raise ValueError("BonePile tried to replace a frozen Head key")
        if len(set(keys)) != len(keys):
            raise ValueError("BonePile reuses one public key for multiple Heads")
        if pile[self.head].key != self.publickey:
            raise ValueError("BonePile does not contain our own public key")
        return pile
    @property
    def BonePile(self) -> BonePile:
        return self.CopyBonePile(self.BuriedBonePile)
    def Seed(self, pile: dict[str, Head]) -> Result:
        try:
            candidate = self.CopyBonePile(pile)
            if not self.Authority:
                self.FreezeAuthority(candidate)
            self.Check(candidate)
            self.NinetyNine(candidate)
        except Exception:
            return Result(status="BAD BONEPILE")
        if candidate == self.BuriedBonePile:
            return Result(status="IDEMPOTENT")
        self.BuriedBonePile = candidate
        self.Strays = {head: bone for head, bone in self.Strays.items() if bone.tag.parent == candidate[head].tag.child}
        result = Result(status="BURIED", changed=True)
        if self.GuardianOut is not None:
            self.GuardianOut(self.BonePile, result)
        return result
    def Hunger(self) -> Result:
        self.Hungry = True
        if self.HungerOut is not None:
            self.HungerOut()
        return Result(status="HUNGRY")
    def VerifyBone(self, bone: Bone) -> Bone:
        if not isinstance(bone, Bone):
            raise TypeError("expected Bone")
        if bone.head not in self.expected or bone.target not in self.expected:
            raise ValueError("Bone names an unknown Head")
        if bone.head == bone.target:
            raise ValueError("a Head cannot give bones to itself")
        if bone.bones <= 0:
            raise ValueError("Bone amount must be positive")
        if not self.Authority:
            raise ValueError("Head authority is not established")
        if bone.key != self.Authority[bone.head]:
            raise ValueError("Bone public key does not own that Head")
        return VerifyBoneProof(bone)
    @staticmethod
    def Recorded(cell: Head, bone: Bone) -> bool:
        receiptid = ReceiptHash(bone)
        return any(ReceiptHash(receipt) == receiptid for receipt in cell.receipts)
    @staticmethod
    def SameFrontier(first: Head, second: Head) -> bool:
        return (
            first.head == second.head
            and first.key == second.key
            and first.tag == second.tag
            and first.locksign == second.locksign
            and first.receipts == second.receipts
        )
    def SetFrontier(self, pile: dict[str, Head], head: str, receipts: tuple[Bone, ...]) -> None:
        cell = pile[head]
        if not receipts:
            pile[head] = replace(cell, receipts=())
            return
        receipts = CanonicalReceipts(*receipts)
        canonical = receipts[0]
        pile[head] = Head(cell.head, cell.key, cell.bones, canonical.tag, canonical.locksign, receipts)
    def Apply(self, maul: Maul, bone: Bone) -> bool:
        source = maul[bone.head]
        target = maul[bone.target]
        if bone.tag.parent != source.tag.child or source.bones < bone.bones:
            return False
        maul[bone.head] = Head(
            source.head,
            source.key,
            source.bones - bone.bones,
            bone.tag,
            bone.locksign,
            (bone,),
        )
        maul[bone.target] = replace(target, bones=target.bones + bone.bones)
        maul.changed = True
        return True
    def Pair(self, mine: Head, theirs: Head) -> Optional[tuple[Bone, Bone]]:
        evidence = list(mine.receipts) + list(theirs.receipts)
        if len(evidence) < 2:
            return None
        groups: dict[tuple[str, str, str], list[Bone]] = {}
        for receipt in evidence:
            key = (receipt.head, receipt.key, receipt.tag.parent)
            groups.setdefault(key, []).append(receipt)
        pairs: list[tuple[Bone, Bone]] = []
        for group in groups.values():
            try:
                pairs.append(FreshBones(*group))
            except Exception:
                continue
        if not pairs:
            return None
        pairs.sort(key=lambda pair: FreshHashes(*pair))
        return pairs[0]
    def Dirt(self, incoming: BonePile) -> set[str]:
        dirt: set[str] = set()
        for head in self.heads:
            mine = self.BuriedBonePile[head]
            theirs = incoming[head]
            if mine.bones <= 0 or theirs.bones != 0:
                continue
            if self.Pair(mine, theirs) is not None:
                dirt.add(head)
        return dirt
    def Seek(self, packet: Bone | BonePile) -> tuple[Optional[Maul], Result]:
        maul = Maul(self.BuriedBonePile)
        try:
            self.Check(self.BuriedBonePile)
        except Exception:
            return None, Result(status="BAD BONEPILE")
        if isinstance(packet, Bone):
            maul.bone = packet
            try:
                self.VerifyBone(packet)
            except Exception:
                return None, Result(status="BAD BONE", bone=packet)
            current = maul[packet.head]
            if self.Recorded(current, packet):
                return None, Result(status="IDEMPOTENT", bone=packet)
            held = tuple(current.receipts)
            stray = self.Strays.get(packet.head)
            evidence = held + ((stray,) if stray and stray.tag.parent == packet.tag.parent else ())
            sibling = bool(
                evidence
                and all(receipt.tag.parent == packet.tag.parent for receipt in evidence)
                and packet.tag.child not in {receipt.tag.child for receipt in evidence}
            )
            if sibling:
                try:
                    pair = FreshBones(*(evidence + (packet,)))
                except Exception:
                    return None, Result(status="BAD BONE", bone=packet)
                if len(held) == 2 and FreshHashes(*pair) >= FreshHashes(*held):
                    return None, Result(status="IDEMPOTENT", bone=packet)
                self.Strays.pop(packet.head, None)
                maul.equivocation = maul.forward = maul.changed = True
                maul.changedpairs.add(packet.head)
                maul.frontier[packet.head] = pair
                return maul, Result(status="MAULED", changed=True, reproject=True, bone=packet)
            dirtydogs = self.DirtyDogs(self.BuriedBonePile)
            if packet.head in dirtydogs:
                return None, Result(status="IDEMPOTENT", bone=packet)
            if packet.tag.parent != current.tag.child:
                return None, Result(status="BAD BONE", bone=packet)
            if packet.bones > current.bones:
                return None, Result(status="GROWL", bone=packet)
            if packet.target in dirtydogs:
                self.Strays[packet.head] = packet
                return None, Result(status="STRAY", bone=packet)
            if not self.Apply(maul, packet):
                return None, Result(status="BAD BONE", bone=packet)
            self.Strays.pop(packet.head, None)
            maul.forward = True
            self.Check(maul)
            return maul, Result(status="MAULED", changed=True, bone=packet)
        try:
            incoming = self.CopyBonePile(packet)
            self.Check(incoming)
        except Exception:
            return None, Result(status="BAD BONEPILE")
        if incoming == self.BuriedBonePile:
            return None, Result(status="IDEMPOTENT")
        maul.incoming = incoming
        maul.dirt = self.Dirt(incoming)
        ordinary: dict[str, Bone] = {}
        for head in self.heads:
            mine = self.BuriedBonePile[head]
            theirs = incoming[head]
            if mine.key != theirs.key:
                return None, Result(status="BAD BONEPILE")
            if self.SameFrontier(mine, theirs):
                if head in maul.dirt:
                    pair = self.Pair(mine, theirs)
                    if pair is not None:
                        maul.equivocation = True
                        maul.changed = True
                        maul.changedpairs.add(head)
                        maul.frontier[head] = pair
                continue
            mineids = {ReceiptHash(item) for item in mine.receipts}
            pair = self.Pair(mine, theirs)
            pairevidence = bool(
                pair and any(ReceiptHash(item) not in mineids for item in pair)
            )
            if pairevidence or len(theirs.receipts) == 2:
                if pair is None:
                    continue
                currentpair: tuple[Bone, ...] = ()
                if len(mine.receipts) == 2:
                    try:
                        currentpair = FreshBones(*mine.receipts)
                    except Exception:
                        currentpair = ()
                if currentpair and FreshHashes(*pair) >= FreshHashes(*currentpair):
                    if head in maul.dirt:
                        maul.equivocation = True
                        maul.changed = True
                        maul.changedpairs.add(head)
                        maul.frontier[head] = currentpair
                    continue
                maul.equivocation = True
                maul.forward = maul.forward or pairevidence
                maul.changed = True
                maul.changedpairs.add(head)
                maul.frontier[head] = pair
                continue
            if len(theirs.receipts) == 1:
                ordinary[head] = theirs.receipts[0]
        pending = dict(ordinary)
        while pending:
            progress = False
            for head, edge in tuple(pending.items()):
                if self.Recorded(maul[head], edge):
                    pending.pop(head, None)
                    progress = True
                    continue
                try:
                    self.VerifyBone(edge)
                except Exception:
                    return None, Result(status="BAD BONEPILE")
                if self.Apply(maul, edge):
                    maul.forward = True
                    pending.pop(head, None)
                    progress = True
            if not progress:
                break
        if not maul.changed:
            return None, Result(status="LOCKED")
        return maul, Result(status="MAULED", changed=True, reproject=maul.equivocation)
    @staticmethod
    def Shares(total: int, heads: set[str]) -> dict[str, int]:
        total = int(total)
        if total < 0:
            raise ValueError("Spoils cannot be negative")
        ordered = sorted(set(heads))
        if not ordered:
            if total:
                raise ValueError("Spoils have no eligible DogPile Heads")
            return {}
        q, r = divmod(total, len(ordered))
        return {
            head: q + (1 if index < r else 0)
            for index, head in enumerate(ordered)
        }
    @staticmethod
    def DirtyDogs(pile: dict[str, Head]) -> set[str]:
        return {
            head
            for head, cell in pile.items()
            if int(cell.bones) == 0 and len(cell.receipts) == 2
        }
    def FinalFrontier(self, maul: Maul, head: str) -> tuple[Bone, ...]:
        if head in maul.frontier:
            return FreshBones(*maul.frontier[head])
        return CanonicalReceipts(*maul[head].receipts)
    def DogPile(self, maul: Maul, dirtydogs: set[str]) -> set[str]:
        dogpile: set[str] = set()
        for dog in dirtydogs:
            pair = self.FinalFrontier(maul, dog)
            if len(pair) != 2:
                raise ValueError("dirty Dog has no retained sibling pair")
            dogpile.update(receipt.target for receipt in pair)
        return dogpile
    def Clawback(self, maul: Maul) -> Maul:
        if not isinstance(maul, Maul):
            raise TypeError("Clawback expects Maul")
        dirtydogs = self.DirtyDogs(self.BuriedBonePile)
        solvent: set[str] = set()
        for head in sorted(maul.changedpairs):
            pair = FreshBones(*maul.frontier[head])
            maul.frontier[head] = pair
            if head in dirtydogs:
                continue
            current = self.BuriedBonePile[head]
            old = tuple(current.receipts)
            sameparent = tuple(
                receipt for receipt in old if receipt.tag.parent == pair[0].tag.parent
            )
            estate = int(maul[head].bones) + sum(int(r.bones) for r in sameparent)
            claims = sum(int(r.bones) for r in pair)
            if claims > estate:
                dirtydogs.add(head)
            else:
                solvent.add(head)
        changed = True
        while changed:
            changed = False
            for head in self.heads:
                if head in dirtydogs:
                    continue
                pair = self.FinalFrontier(maul, head)
                if len(pair) == 2 and any(receipt.target in dirtydogs for receipt in pair):
                    solvent.discard(head)
                    dirtydogs.add(head)
                    changed = True
                    break
        maul.solvent = solvent
        maul.dirtydogs = dirtydogs
        maul.dogpile = self.DogPile(maul, dirtydogs)
        return maul
    def Retract(
        self,
        maul: Maul,
        balances: dict[str, int],
        head: str,
    ) -> None:
        for receipt in CanonicalReceipts(*self.BuriedBonePile[head].receipts):
            balances[head] += int(receipt.bones)
            balances[receipt.target] -= int(receipt.bones)
    def Spoils(
        self,
        maul: Maul,
        balances: dict[str, int],
        dirtydogs: set[str],
        newdirty: set[str],
    ) -> dict[str, int]:
        base = {head: int(balances[head]) for head in self.heads}
        spoils = sum(int(base[dog]) for dog in newdirty)
        if spoils < 0:
            raise ValueError("dirty Dogs cannot contribute negative Spoils")
        for dog in newdirty:
            base[dog] = 0
        surface = self.DogPile(maul, set(newdirty))
        trailed: set[str] = set()
        while True:
            resolved = dict(base)
            for head, share in self.Shares(spoils, surface - dirtydogs).items():
                resolved[head] += share
            negative = {head for head, value in resolved.items() if value < 0}
            if not negative:
                break
            progress = False
            for head in sorted(negative - dirtydogs - trailed):
                receipts = CanonicalReceipts(*maul[head].receipts)
                if not receipts:
                    continue
                for receipt in receipts:
                    base[head] += int(receipt.bones)
                    base[receipt.target] -= int(receipt.bones)
                    surface.add(receipt.target)
                surface.add(head)
                trailed.add(head)
                progress = True
            if not progress:
                raise ValueError("negative Head cannot be ossified")
        maul.dogpile.update(surface)
        expected = BonesPerHead * len(self.heads)
        if sum(resolved.values()) != expected:
            raise ValueError("Spoils did not conserve the field")
        return resolved
    def Underbelly(self, maul: Maul) -> Optional[Maul]:
        if maul.incoming is None or not maul.dirt:
            return None
        dirtydogs = set(maul.dirtydogs) | set(maul.dirt)
        try:
            dogpile = self.DogPile(maul, dirtydogs)
            balances = {head: int(maul[head].bones) for head in self.heads}
            newdirty = dirtydogs - set(maul.dirtydogs)
            for head in sorted(newdirty):
                self.Retract(maul, balances, head)
            resolved = self.Spoils(maul, balances, dirtydogs, newdirty)
            for head in self.heads:
                maul[head] = replace(maul[head], bones=int(resolved[head]))
        except Exception:
            return None
        maul.dirtydogs, maul.ossified = dirtydogs, True
        maul.dogpile.update(dogpile)
        if not maul.forward and BonePile(dict(maul)) != maul.incoming:
            return None
        return maul
    def Dig(self, maul: Maul) -> Optional[Maul]:
        if not isinstance(maul, Maul):
            raise TypeError("Dig expects Maul")
        balances = {head: int(maul[head].bones) for head in self.heads}
        try:
            for head in sorted(maul.solvent):
                self.Retract(maul, balances, head)
                pair = FreshBones(*maul.frontier[head])
                claims = sum(int(receipt.bones) for receipt in pair)
                if balances[head] < claims:
                    return None
                for receipt in pair:
                    balances[head] -= int(receipt.bones)
                    balances[receipt.target] += int(receipt.bones)
                self.SetFrontier(maul, head, pair)
            burieddirty = self.DirtyDogs(self.BuriedBonePile)
            newdirty = maul.dirtydogs - burieddirty
            for head in sorted(newdirty):
                self.Retract(maul, balances, head)
            for head in sorted(maul.dirtydogs):
                if head in maul.frontier:
                    self.SetFrontier(maul, head, FreshBones(*maul.frontier[head]))
            if maul.dirtydogs:
                balances = self.Spoils(maul, balances, maul.dirtydogs, newdirty)
            elif any(value < 0 for value in balances.values()):
                return None
            for head in self.heads:
                maul[head] = replace(maul[head], bones=int(balances[head]))
            maul.ossified = True
        except Exception:
            return None
        if maul.incoming is not None and maul.dirt:
            if not (maul.dirt - maul.dirtydogs):
                return maul
            if BonePile(dict(maul)) == maul.incoming:
                return maul
            return self.Underbelly(maul)
        return maul
    def Bury(self, maul: Maul, *, status: str = "BURIED") -> Result:
        bone = maul.bone
        bad = "BAD BONE" if bone is not None else "BAD BONEPILE"
        try:
            # ================= LINCHPIN ================= #
            if maul.equivocation and not maul.ossified:
                raise ValueError("equivocation Maul reached Bury before full ossification")
            self.NinetyNine(maul)
            candidate = BonePile(dict(maul))
            self.Check(candidate)
            # ============================================ #
        except Exception:
            return Result(status=bad, bone=bone)
        if candidate == self.BuriedBonePile:
            return Result(status="IDEMPOTENT", bone=bone)
        self.BuriedBonePile = candidate
        self.Strays = {head: bone for head, bone in self.Strays.items() if bone.tag.parent == candidate[head].tag.child}
        self.Hungry = False
        result = Result(
            status=status,
            changed=True,
            reproject=maul.equivocation,
            bone=bone,
        )
        if self.GuardianOut is not None:
            self.GuardianOut(self.BonePile, result)
        if maul.equivocation:
            self.Project(self.BonePile)
        elif bone is not None and self.BoneYardOut is not None:
            self.BoneYardOut(bone, result)
        return result
    def Project(self, pile: BonePile) -> None:
        if self.ProjectOut is not None:
            self.ProjectOut(self.CopyBonePile(pile))
    def ReceiveBone(self, bone: Bone) -> Result:
        maul, result = self.Seek(bone)
        if maul is None:
            if result.status == "GROWL" and self.BoneYardOut is not None and isinstance(bone, Bone):
                self.BoneYardOut(bone, result)
            return result
        if maul.equivocation:
            try:
                self.Clawback(maul)
            except Exception:
                return Result(status="BAD BONE", bone=bone)
            maul = self.Dig(maul)
            if maul is None:
                self.Hungry = True
                if self.HungerOut is not None:
                    self.HungerOut()
                return Result(status="HUNGRY", bone=bone)
            return self.Bury(maul, status="MAULED")
        return self.Bury(maul, status="BURIED")
    def FetchBonePile(self, pile: BonePile) -> Result:
        maul, result = self.Seek(pile)
        if maul is None:
            return result
        if maul.equivocation:
            try:
                self.Clawback(maul)
            except Exception:
                return Result(status="BAD BONEPILE")
            maul = self.Dig(maul)
            if maul is None:
                return Result(status="LOCKED")
            return self.Bury(maul, status="MAULED")
        return self.Bury(maul, status="BURIED")
