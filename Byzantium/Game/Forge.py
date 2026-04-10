from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from wcwidth import wcwidth as wc
except Exception:
    wc = None


class Geometry:
    cols: int = 4
    rows: int = 6
    cells: int = cols * rows
    term: int = 80
    framepad: int = 1
    inner: int = term - framepad * 2
    name: int = 8
    msg: int = 60
    cost: int = 8

    @classmethod
    def File(cls, q: int) -> int:
        return int(q) // cls.rows

    @classmethod
    def Rank(cls, q: int) -> int:
        return int(q) % cls.rows

    @classmethod
    def General(cls, q: int) -> bool:
        return cls.Rank(q) == 0

    @classmethod
    def Move(cls, q: int, arrow: str) -> int:
        row = cls.Rank(q)
        col = cls.File(q)
        if arrow == 'A':
            row = (row - 1) % cls.rows
        elif arrow == 'B':
            row = (row + 1) % cls.rows
        elif arrow == 'D':
            col = (col - 1) % cls.cols
        elif arrow == 'C':
            col = (col + 1) % cls.cols
        return col * cls.rows + row

    @classmethod
    def Split(cls, total: int, cells: Sequence[Any], *, by=None) -> Tuple[Tuple[str, int], ...]:
        total = max(0, int(total or 0))
        live = [cell for cell in cells if Key(cell)]
        count = len(live)
        if total <= 0 or count <= 0:
            return ()
        ordered = sorted(live, key=Amount if by is None else by)
        base = total // count
        rem = total % count
        out: List[Tuple[str, int]] = []
        for i, cell in enumerate(ordered):
            share = base + (1 if i < rem else 0)
            if share > 0:
                out.append((Key(cell), share))
        return tuple(out)


geometry = Geometry
Columns = geometry.cols
Rows = geometry.rows
TerminalWidth = geometry.term
InnerWidth = geometry.inner
NameWidth = geometry.name
MessageMax = geometry.msg
SaltWidth = geometry.cost
Hline = '=' * TerminalWidth
BodyFillLines = 23
AnsiRe = re.compile(r'\x1b\[[0-9;]*m')

Reset = '\x1b[0m'
Ash = '\x1b[90m'
Salt = '\x1b[97m'
Ember = '\x1b[38;5;130m'
Flame = '\x1b[38;5;208m'
Flare = '\x1b[38;5;214m'


def FlickerCycle(sequence: Tuple[str, ...], phase: int) -> str:
    if not sequence:
        return Ash
    return sequence[int(phase or 0) % len(sequence)]


def Crucible(cache: Any = None) -> Dict[str, str]:
    raw = getattr(cache, 'flamephase', None) if cache is not None else None
    try:
        phase = int(raw) if raw is not None else 0
    except Exception:
        phase = 0
    phase += int(time.monotonic() * 8.0)
    flicker1 = FlickerCycle((Flame, Flare), phase)
    flicker2 = FlickerCycle((Flare, Flame), phase)
    flicker3 = FlickerCycle((Flame, Flame, Flare), phase)
    flicker4 = FlickerCycle((Flare, Flare, Flame, Ember), phase)
    flicker5 = FlickerCycle((Ember, Ember, Ember, Flame, Flame, Flare), phase)
    flicker6 = FlickerCycle((Ash, Ember, Ember, Flare, Ember, Ember), phase)

    return {
        'Ash': Ash,
        'Salt': Salt,
        'Ember': Ember,
        'Flame': Flame,
        'Flare': Flare,
        'Flicker1': flicker1,
        'Flicker2': flicker2,
        'Flicker3': flicker3,
        'Flicker4': flicker4,
        'Flicker5': flicker5,
        'Flicker6': flicker6,
        'Reset': Reset,
    }


@dataclass(frozen=True)
class Cell:
    soul: str = ''
    key: str = ''
    salt: int = 0
    purge: Any = None
    lock: Any = None
    sign: Any = None


@dataclass(frozen=True)
class State:
    cells: Tuple[Cell, ...] = ()
    self: Tuple[str, str] = ('', '')
    monument: Tuple[str, ...] = ()


@dataclass
class Q:
    self: Optional[int] = None
    target: Optional[int] = None
    city: int = 0


class Focus(str, Enum):
    Title = 'title'
    Menu = 'menu'
    TableMove = 'tablemove'
    TableLock = 'tablelock'
    Spine = 'spine'


class Action(str, Enum):
    Purge = 'purge'
    Whisper = 'whisper'
    Rally = 'rally'
    Wrath = 'wrath'
    Defect = 'defect'
    Monument = 'monument'
    Lore = 'lore'
    Exit = 'exit'


Menu: List[Action] = [
    Action.Whisper,
    Action.Rally,
    Action.Wrath,
    Action.Defect,
    Action.Purge,
    Action.Monument,
    Action.Lore,
    Action.Exit,
]


@dataclass
class Intent:
    focus: Focus = Focus.Menu
    action: Action = Action.Purge
    q: Q = field(default_factory=Q)
    amount: int = 1
    text: str = ''
    kind: str = 'purge'
    pairs: Tuple[Tuple[str, int], ...] = ()
    lock: Any = None

    def __post_init__(self) -> None:
        self.amount = max(0, int(self.amount or 0))
        self.text = CleanDraft(self.text)
        self.kind = str(self.kind or getattr(self.action, 'value', self.action) or '').lower()
        self.pairs = tuple((str(k or '').strip(), int(v or 0)) for k, v in tuple(self.pairs or ()))


@dataclass
class Cache:
    feed: list
    name: str
    monuments: Optional[list] = None
    state: Optional[Any] = None
    intent: Intent = field(default_factory=Intent)
    focus: Focus = Focus.Menu
    menuq: int = 0
    stateq: int = 0
    targetq: Optional[int] = None
    statekey: str = ''
    targetkey: str = ''
    salt: int = 1
    text: str = ''
    feedcount: int = 0
    flamephase: int = 0
    flamefed: bool = False
    activerequest: Optional[str] = None
    banner: bool = True
    lore: bool = False
    showdebug: bool = False
    lorescroll: int = 0
    mode: str = 'Siege'
    gate: str = '9000'
    skeleton: str = 'Skeleton'
    secret: str = 'Password'
    genesis: str = '1'
    soul: str = ''
    titleselect: int = 0

    def SyncIntent(self) -> Intent:
        self.intent.focus = self.focus
        self.intent.action = Menu[self.menuq % len(Menu)]
        self.intent.q.city = int(self.stateq or 0)
        self.intent.q.target = self.targetq
        self.intent.amount = max(0, int(self.salt or 0))
        self.intent.text = CleanDraft(self.text)
        self.intent.kind = str(getattr(self.intent.action, 'value', self.intent.action) or '').lower()
        return self.intent


ActionMap: Dict[Action, Dict[str, object]] = {
    Action.Whisper: {'floor': 1, 'desc': 'send salt + a private message', 'preview': "You Didn't Hear This From Me", 'label': 'WHISPER'},
    Action.Rally: {'floor': 100, 'desc': 'spend salt on your column', 'preview': 'You Got To Pump It Up', 'label': 'COHORT'},
    Action.Wrath: {'floor': 1000, 'desc': 'spend salt on everyone', 'preview': 'Show No Mercy', 'label': 'LEGION'},
    Action.Defect: {'floor': 0, 'desc': 'rank-dependent cost to swap seats', 'preview': 'Friends Are Friends Until The End', 'label': None},
    Action.Purge: {'floor': 0, 'desc': '', 'preview': 'Restore Formation', 'label': None},
    Action.Monument: {'floor': 0, 'desc': '', 'preview': 'Memory Set In Stone', 'label': None},
    Action.Lore: {'floor': 0, 'desc': '', 'preview': 'HisStory', 'label': None},
    Action.Exit: {'floor': 0, 'desc': '', 'preview': 'Abandon Post', 'label': None},
}


def MakeCell(raw: Any) -> Cell:
    if isinstance(raw, Cell):
        return raw
    return Cell(
        soul=str(getattr(raw, 'soul', getattr(raw, 'Soul', '')) or ''),
        key=str(getattr(raw, 'key', '') or ''),
        salt=int(getattr(raw, 'salt', 0) or 0),
        purge=getattr(raw, 'purge', None),
        lock=getattr(raw, 'lock', None),
        sign=getattr(raw, 'sign', None),
    )


def MakeSelf(raw: Any) -> Tuple[str, str]:
    if raw is None:
        return ('', '')
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return (str(raw[0] or ''), str(raw[1] or '').strip())
    if isinstance(raw, dict):
        return (str(raw.get('soul', '') or ''), str(raw.get('key', '') or '').strip())
    return (str(getattr(raw, 'soul', '') or ''), str(getattr(raw, 'key', '') or '').strip())


def MakeState(raw: Any, selfraw: Any = None) -> State:
    if isinstance(raw, State) and selfraw is None:
        return raw
    cells = tuple(MakeCell(cell) for cell in getattr(raw, 'cells', ()) or ())
    me = MakeSelf(selfraw if selfraw is not None else getattr(raw, 'self', None))
    monument = tuple(getattr(raw, 'monument', ()) or ())
    return State(cells=cells, self=me, monument=monument)


def VisLen(text: str) -> int:
    raw = AnsiRe.sub('', str(text or ''))
    if wc is None:
        return len(raw)
    total = 0
    for ch in raw:
        width = wc(ch)
        if width and width > 0:
            total += width
    return total


def ClipWidth(text: str, width: int) -> str:
    text = str(text or '')
    width = max(0, int(width or 0))
    out: List[str] = []
    seen = 0
    i = 0
    while i < len(text):
        if text[i] == '\x1b' and i + 1 < len(text) and text[i + 1] == '[':
            j = i + 2
            while j < len(text) and text[j] != 'm':
                j += 1
            if j < len(text):
                out.append(text[i:j + 1])
                i = j + 1
                continue
        if seen >= width:
            break
        ch = text[i]
        step = 1 if wc is None else wc(ch)
        if step is None or step < 0:
            step = 0
        if seen + step > width:
            break
        out.append(ch)
        seen += step
        i += 1
    return ''.join(out)


def CenterWidth(text: str, width: int) -> str:
    text = str(text or '')
    seen = VisLen(text)
    if seen >= width:
        return ClipWidth(text, width)
    return ' ' * ((int(width) - seen) // 2) + text


def PadWidth(text: str, width: int) -> str:
    text = str(text or '')
    seen = VisLen(text)
    if seen >= width:
        return ClipWidth(text, width)
    return text + ' ' * (int(width) - seen)


def ClipTerm(text: str, *, term: int = TerminalWidth) -> str:
    return ClipWidth(text, term)


def CenterTerm(text: str, *, term: int = TerminalWidth) -> str:
    return CenterWidth(text, term)


def FrameLines(lines: List[str], *, inner: int = InnerWidth) -> str:
    return '\n'.join(' ' + PadWidth(ClipWidth(line, inner), inner) + ' ' for line in lines) + Reset


def MastLines(lux: Dict[str, str], title: str = 'BYZANTIUM') -> List[str]:
    return [
        CenterTerm(''),
        CenterTerm(''),
        CenterTerm(lux['Ash'] + '.' + lux['Reset']),
        CenterTerm(lux['Ash'] + '.' + lux['Reset'] + lux['Flicker1'] + '+' + lux['Reset'] + lux['Ash'] + '.' + lux['Reset']),
        CenterTerm(lux['Ash'] + '.   .   .   .' + lux['Reset']),
        CenterTerm(lux['Ash'] + lux['Flicker2'] + '+' + lux['Reset'] + f' {title} ' + lux['Reset'] + lux['Flicker3'] + '+' + lux['Reset']),
        CenterTerm(lux['Ash'] + '·   · ·   · ·   ·' + lux['Reset']),
        CenterTerm(lux['Ash'] + '·' + lux['Reset'] + lux['Flicker4'] + '+' + lux['Reset'] + lux['Ash'] + '·' + lux['Reset']),
        CenterTerm(lux['Ash'] + '·' + lux['Reset']),
    ]


def ValueField(lux: Dict[str, str], value: str = '') -> str:
    value = str(value or '')[:8]
    left = lux['Flicker1'] + ':' + lux['Reset']
    right = lux['Flicker2'] + ':' + lux['Reset']
    if not value:
        return left + right
    return left + lux['Salt'] + value + lux['Reset'] + right


def FrameTextScreen(lines: List[str], *, fill: int = BodyFillLines, term: int = TerminalWidth, inner: int = InnerWidth) -> str:
    body = [ClipTerm(str(line or ''), term=inner) for line in list(lines or ())]
    while len(body) < int(fill):
        body.append('')
    body.append(ClipTerm(Ash + '=' * term + Reset, term=inner))
    return FrameLines(body, inner=inner)


def CleanDraft(text: object) -> str:
    return str(text or '').replace('\r', ' ').replace('\n', ' ')


def MessageNorm(text: str, *, maxlen: int = MessageMax) -> str:
    return CleanDraft(text).strip()[:max(0, int(maxlen))]


def SpineCost(cost: int, *, width: int = SaltWidth, signed: bool = True) -> str:
    value = int(cost or 0)
    text = f'{value:+,}' if signed else f'{value:,}'
    if signed and not text.startswith(('+', '-')):
        text = '+' + text
    return text.rjust(int(width or 0))


def ParseMonument(line: str, *, name: int = NameWidth):
    if len(line) < 10:
        return (None, None, line)
    head = line[:name].strip()
    tail = line[name:].strip()
    match = re.match(r'^([+-]?[\d,]+):\s*(.*)', tail)
    return (head, match.group(1), match.group(2)) if match else (head, None, tail)


def MonumentAnchorCol(monuments: List[str], anchor: str, *, name: int = NameWidth) -> int:
    parsed = [ParseMonument(m, name=name) for m in monuments]
    for head, score, _ in parsed:
        if head == anchor and score is not None:
            return len(f'{head.ljust(name)[:name]} {score}')
    widths = [len(f'{head.ljust(name)[:name]} {score}') for head, score, _ in parsed if head is not None]
    return max(widths, default=0)


def Key(raw: Any) -> str:
    if raw is None:
        return ''
    value = getattr(raw, 'key', None)
    if value is None and isinstance(raw, dict):
        value = raw.get('key', '')
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value or '').strip()


def Amount(cell: Any) -> int:
    return int(getattr(cell, 'salt', 0) or 0)


def SelfKey(state: Any) -> str:
    return MakeSelf(getattr(state, 'self', None))[1]


def QxKey(state: Any, raw: Any) -> Optional[int]:
    needle = str(raw or '').strip()
    if not needle:
        return None
    cells = list(getattr(state, 'cells', []) or [])
    for i, cell in enumerate(cells):
        if Key(cell) == needle:
            return i
    return None


def SelfQ(state: Any) -> Optional[int]:
    mine = SelfKey(state)
    return None if not mine else QxKey(state, mine)


def ResolveRank(state: Any) -> Optional[int]:
    q = SelfQ(state)
    return None if q is None else q + 1


def KeyxQ(state: Any, q: int) -> str:
    cells = list(getattr(state, 'cells', []) or [])
    return Key(cells[int(q)]) if 0 <= int(q) < len(cells) else ''


def ActionPreview(action: Action) -> str:
    return str(ActionMap.get(action, {}).get('preview', action.value))


def ActionDesc(action: Action) -> str:
    return str(ActionMap.get(action, {}).get('desc', ''))


def ActionFloor(action: Action, *, rank: Optional[int] = None, q: Optional[int] = None, city: Optional[int] = None) -> int:
    seat = q if q is not None else city
    if action == Action.Defect:
        if seat is None:
            seat = None if rank is None else int(rank) - 1
        if seat is None:
            return 1000
        return 10000 if geometry.General(seat) else 1000
    return int(ActionMap.get(action, {}).get('floor', 0) or 0)


def ActionSpineLabel(action: Action, *, rank: Optional[int] = None, q: Optional[int] = None, city: Optional[int] = None) -> Optional[str]:
    label = ActionMap.get(action, {}).get('label')
    if label:
        return str(label)
    if action == Action.Defect:
        return 'MYRIAD' if ActionFloor(action, rank=rank, q=q, city=city) >= 10000 else 'COHORT'
    return None


def DefectViable(state: Any, mine: int, target: int) -> bool:
    if target == mine:
        return False
    cells = list(getattr(state, 'cells', []) or [])
    if not (0 <= mine < len(cells) and 0 <= target < len(cells)):
        return False
    return Amount(cells[target]) < Amount(cells[mine]) and geometry.File(target) != geometry.File(mine)


def MoveTable(state: Any, city: int, arrow: str, current: Action) -> int:
    old = int(city or 0) % geometry.cells
    me = SelfQ(state)
    mine = 0 if me is None else int(me) % geometry.cells
    if current == Action.Defect:
        row = geometry.Rank(old)
        col = geometry.File(old)
        if arrow in ('C', 'D'):
            step = 1 if arrow == 'C' else -1
            for hop in range(1, geometry.cols + 1):
                q = (col + step * hop) % geometry.cols * geometry.rows + row
                if DefectViable(state, mine, q):
                    return q
            return old
        if arrow in ('A', 'B'):
            step = -1 if arrow == 'A' else 1
            for hop in range(1, geometry.rows + 1):
                q = col * geometry.rows + (row + step * hop) % geometry.rows
                if DefectViable(state, mine, q):
                    return q
            return old
        return old
    cand = geometry.Move(old, arrow)
    if current in (Action.Whisper, Action.Purge) and cand == mine:
        q = cand
        for _ in range(geometry.cells - 1):
            q = geometry.Move(q, arrow)
            if q != mine:
                return q
    return cand


def Targets(state: Any, mine: int, current: Action, target: Optional[int]) -> Tuple[int, ...]:
    cells = list(getattr(state, 'cells', []) or [])
    if current in (Action.Whisper, Action.Purge):
        return () if target is None else (int(target),)
    if current == Action.Defect:
        myfile = geometry.File(int(mine))
        return tuple(q for q, _cell in enumerate(cells) if q != int(mine) and geometry.File(q) == myfile)
    if current == Action.Rally:
        myfile = geometry.File(int(mine))
        return tuple(q for q, _cell in enumerate(cells) if q != int(mine) and geometry.File(q) == myfile)
    if current == Action.Wrath:
        return tuple(q for q, _cell in enumerate(cells) if q != int(mine))
    return ()


def PurgeViable(state: Any, mine: int, target: Optional[int]) -> bool:
    cells = list(getattr(state, 'cells', []) or [])
    if target is None:
        return False
    q = int(target)
    return 0 <= q < len(cells) and q != int(mine) and bool(Key(cells[q]))


def PurgeTarget(state: Any, q: Optional[int]) -> str:
    return '' if q is None else KeyxQ(state, int(q))


def PurgeFlavor() -> str:
    return 'Friends are friends until the end.'


def DefectTargetKey(state: Any, q: Optional[int]) -> str:
    return '' if q is None else KeyxQ(state, int(q))
