from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
try:
    from wcwidth import wcwidth as wc
except Exception:
    wc = None

class Geometry:
    cols: int = 4
    rows: int = 6
    cells: int = cols * rows
    term: int = 80
    framePad: int = 1
    inner: int = term - framePad * 2
    name: int = 8
    msg: int = 60
    cost: int = 8

    @classmethod
    def file(cls, q: int) -> int:
        return int(q) // cls.rows

    @classmethod
    def rank(cls, q: int) -> int:
        return int(q) % cls.rows

    @classmethod
    def general(cls, q: int) -> bool:
        return cls.rank(q) == 0

    @classmethod
    def wrap(cls, q: int, delta: int, *, count: Optional[int]=None) -> int:
        n = cls.cells if count is None else max(0, int(count))
        if n <= 0:
            return 0
        return (int(q) + int(delta)) % n

    @classmethod
    def move(cls, q: int, arrow: str) -> int:
        row = cls.rank(q)
        col = cls.file(q)
        if arrow == 'A':
            row, col = ((row - 1) % cls.rows, col)
        elif arrow == 'B':
            row, col = ((row + 1) % cls.rows, col)
        elif arrow == 'D':
            row, col = (row, (col - 1) % cls.cols)
        elif arrow == 'C':
            row, col = (row, (col + 1) % cls.cols)
        return col * cls.rows + row

    @classmethod
    def split(cls, total: int, cells: Sequence[Any], *, by=None) -> Tuple[Tuple[str, int], ...]:
        total = max(0, int(total or 0))
        cells = [cell for cell in cells if str(key(cell) or '').strip()]
        n = len(cells)
        if total <= 0 or n <= 0:
            return ()
        if by is None:
            by = amount
        ordered = sorted(cells, key=by)
        base = total // n
        rem = total % n
        out: List[Tuple[str, int]] = []
        for i, cell in enumerate(ordered):
            share = base + (1 if i < rem else 0)
            if share > 0:
                out.append((key(cell), share))
        return tuple(out)
geometry = Geometry
BOARD_COLS = geometry.cols
BOARD_ROWS = geometry.rows
CELL_COUNT = geometry.cells
TERM_W = geometry.term
FRAME_PAD = geometry.framePad
INNER_W = geometry.inner
NAME_W = geometry.name
MSG_MAX = geometry.msg
COST_W = geometry.cost
ANSI_RE = re.compile('\\x1b\\[[0-9;]*m')


RESET = '\x1b[0m'
ASH = '\x1b[90m'
WHITE = '\x1b[97m'
SALT = WHITE
EMBER = '\x1b[38;5;130m'
FLICKER1 = '\x1b[38;5;208m'
FLICKER2 = '\x1b[38;5;214m'

def flickerPair(phase: int) -> Tuple[str, str]:
    return (FLICKER1, FLICKER2) if int(phase or 0) % 2 else (FLICKER2, FLICKER1)

def palette(cache: Any=None) -> Dict[str, str]:
    raw_phase = getattr(cache, 'flame_phase', None) if cache is not None else None
    try:
        phase = int(raw_phase) if raw_phase is not None else 0
    except Exception:
        phase = 0
    phase += int(__import__('time').monotonic() * 8.0)
    flicker1, flicker2 = flickerPair(phase)
    return {
        'reset': RESET,
        'ash': ASH,
        'white': WHITE,
        'salt': SALT,
        'ember': EMBER,
        'flicker1': flicker1,
        'flicker2': flicker2,
    }


@dataclass(frozen=True)
class Cell:
    soul: str = ''
    key: str = ''
    salt: int = 0
    reserve: int = 0
    purge: Any = None
    lock: Any = None
    sign: Any = None
    lockset: Any = None
    debit: Any = None
    credit: Any = None

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
    TITLE = 'TITLE'
    MENU = 'MENU'
    TABLE_MOVE = 'TABLE_MOVE'
    TABLE_LOCK = 'TABLE_LOCK'
    SPINE = 'SPINE'

class Action(str, Enum):
    PURGE = 'PURGE'
    WHISPER = 'WHISPER'
    RALLY = 'RALLY'
    WRATH = 'WRATH'
    DEFECT = 'DEFECT'
    MONUMENT = 'MONUMENT'
    LORE = 'LORE'
    EXIT = 'EXIT'
MENU: List[Action] = [Action.WHISPER, Action.RALLY, Action.WRATH, Action.DEFECT, Action.PURGE, Action.MONUMENT, Action.LORE, Action.EXIT]

@dataclass
class Intent:
    focus: Focus = Focus.MENU
    action: Action = Action.PURGE
    Q: Q = field(default_factory=Q)
    amount: int = 1
    text: str = ''
    kind: str = 'PURGE'
    pairs: Tuple[Tuple[str, int], ...] = ()
    lock: Any = None

    def __post_init__(self):
        self.amount = max(0, int(self.amount or 0))
        self.text = cleanDraft(self.text)
        self.kind = str(self.kind or getattr(self.action, 'value', self.action) or '').upper()
        self.pairs = tuple(((str(k or '').strip(), int(v or 0)) for k, v in tuple(self.pairs or ())))

@dataclass
class UiCache:
    feed: list
    local_name: str
    monuments: Optional[list] = None
    state: Optional[Any] = None
    intent: Intent = field(default_factory=Intent)
    focus: Focus = Focus.MENU
    menuQ: int = 0
    stateQ: int = 0
    targetQ: Optional[int] = None
    stateKey: str = ''
    targetKey: str = ''
    salt: int = 1
    text: str = ''
    visible_feed_count: int = 0
    flame_phase: int = 0
    flame_fed: bool = False
    pending_request: Optional[str] = None
    show_banner: bool = True
    show_lore: bool = False
    show_debug: bool = False
    lore_offset: int = 0
    mode: str = 'Siege'
    gate: str = '9000'
    skeleton: str = 'Skeleton'
    secret: str = 'Password'
    genesis: str = '1'
    soul: str = ''
    title_idx: int = 0

    def syncIntent(self) -> Intent:
        self.intent.focus = self.focus
        self.intent.action = MENU[self.menuQ % len(MENU)]
        self.intent.Q.city = int(self.stateQ or 0)
        self.intent.Q.target = self.targetQ
        self.intent.amount = max(0, int(self.salt or 0))
        self.intent.text = cleanDraft(self.text)
        self.intent.kind = str(getattr(self.intent.action, 'value', self.intent.action) or '').upper()
        return self.intent
ACTION: Dict[Action, Dict[str, object]] = {Action.WHISPER: {'floor': 1, 'desc': 'send salt + a private message', 'preview': "You Didn't Hear This From Me", 'label': 'WHISPER', 'needsTarget': True, 'arm': False}, Action.RALLY: {'floor': 100, 'desc': 'spend salt on your column', 'preview': 'You Got To Pump It Up', 'label': 'COHORT', 'needsTarget': False, 'arm': True}, Action.WRATH: {'floor': 1000, 'desc': 'spend salt on everyone', 'preview': 'Show No Mercy', 'label': 'LEGION', 'needsTarget': False, 'arm': True}, Action.DEFECT: {'floor': 0, 'desc': 'rank-dependent cost to swap seats', 'preview': 'Friends Are Friends Until The End', 'label': None, 'needsTarget': True, 'arm': False}, Action.PURGE: {'floor': 0, 'desc': '', 'preview': 'Restore Formation', 'label': None, 'needsTarget': True, 'arm': False}, Action.MONUMENT: {'floor': 0, 'desc': '', 'preview': 'Memory Set In Stone', 'label': None, 'needsTarget': False, 'arm': False}, Action.LORE: {'floor': 0, 'desc': '', 'preview': 'HisStory', 'label': None, 'needsTarget': False, 'arm': False}, Action.EXIT: {'floor': 0, 'desc': '', 'preview': 'Abandon Post', 'label': None, 'needsTarget': False, 'arm': False}}

def makeCell(raw: Any) -> Cell:
    if isinstance(raw, Cell):
        return raw
    reserve0 = getattr(raw, 'reserve', None)
    salt0 = getattr(raw, 'salt', 0)
    return Cell(soul=str(getattr(raw, 'soul', '') or ''), key=str(getattr(raw, 'key', getattr(raw, 'pubkey', '')) or ''), salt=int(salt0 or 0), reserve=int((reserve0 if reserve0 is not None else salt0) or 0), purge=getattr(raw, 'purge', None), lock=getattr(raw, 'lock', None), sign=getattr(raw, 'sign', None), lockset=getattr(raw, 'lockset', None), debit=getattr(raw, 'debit', None), credit=getattr(raw, 'credit', None))

def makeSelf(raw: Any) -> Tuple[str, str]:
    if raw is None:
        return ('', '')
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return (str(raw[0] or ''), str(raw[1] or '').strip())
    if isinstance(raw, dict):
        return (str(raw.get('soul', '') or ''), str(raw.get('key', raw.get('pubkey', '')) or '').strip())
    return (str(getattr(raw, 'soul', '') or ''), str(getattr(raw, 'key', getattr(raw, 'pubkey', '')) or '').strip())

def makeState(raw: Any, selfRaw: Any=None) -> State:
    if isinstance(raw, State) and selfRaw is None:
        return raw
    cells = tuple((makeCell(cell) for cell in getattr(raw, 'cells', ()) or ()))
    me = makeSelf(selfRaw if selfRaw is not None else getattr(raw, 'self', None))
    monument = tuple(getattr(raw, 'monument', ()) or ())
    return State(cells=cells, self=me, monument=monument)

def vislen(text: str) -> int:
    raw = ANSI_RE.sub('', str(text or ''))
    if wc is None:
        return len(raw)
    total = 0
    for ch in raw:
        width = wc(ch)
        if width and width > 0:
            total += width
    return total

def clipw(text: str, width: int) -> str:
    text = str(text or '')
    width = max(0, int(width or 0))
    out: List[str] = []
    seen = 0
    i = 0
    while i < len(text):
        if text[i] == '\x1b' and i + 1 < len(text) and (text[i + 1] == '['):
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

def centerw(text: str, width: int) -> str:
    text = str(text or '')
    seen = vislen(text)
    if seen >= width:
        return clipw(text, width)
    return ' ' * ((int(width) - seen) // 2) + text

def padw(text: str, width: int) -> str:
    text = str(text or '')
    seen = vislen(text)
    if seen >= width:
        return clipw(text, width)
    return text + ' ' * (int(width) - seen)

def clipTerm(text: str, *, term: int=TERM_W) -> str:
    return clipw(text, term)

def centerTerm(text: str, *, term: int=TERM_W) -> str:
    return centerw(text, term)


def frameTextScreen(lines: List[str], *, fill: int=23, term: int=TERM_W, inner: int=INNER_W) -> str:
    body = [clipTerm(str(line or ''), term=inner) for line in list(lines or ())]
    while len(body) < int(fill):
        body.append('')
    body.append(clipTerm(ASH + '=' * term + RESET, term=inner))
    return '\n'.join((' ' + padw(clipw(line, inner), inner) + ' ' for line in body)) + RESET

def cleanDraft(text: object) -> str:
    return str(text or '').replace('\r', ' ').replace('\n', ' ')

def msgNorm(text: str, *, maxLen: int=MSG_MAX) -> str:
    return cleanDraft(text).strip()[:max(0, int(maxLen))]

def fmtSpineCost(cost: int, *, width: int=COST_W, signed: bool=True) -> str:
    value = int(cost or 0)
    text = f'{value:+,}' if signed else f'{value:,}'
    if signed and (not text.startswith(('+', '-'))):
        text = '+' + text
    return text.rjust(int(width or 0))

def parseMonument(line: str, *, name: int=NAME_W):
    if len(line) < 10:
        return (None, None, line)
    head = line[:name].strip()
    tail = line[name:].strip()
    match = re.match('^([+-]?[\\d,]+):\\s*(.*)', tail)
    return (head, match.group(1), match.group(2)) if match else (head, None, tail)

def monumentAnchorCol(monuments: List[str], anchor: str, *, name: int=NAME_W) -> int:
    parsed = [parseMonument(m, name=name) for m in monuments]
    for head, score, _ in parsed:
        if head == anchor and score is not None:
            return len(f'{head.ljust(name)[:name]} {score}')
    widths = [len(f'{head.ljust(name)[:name]} {score}') for head, score, _ in parsed if head is not None]
    return max(widths, default=0)

def key(raw: Any) -> str:
    if raw is None:
        return ''
    value = getattr(raw, 'key', None)
    if value is None:
        value = getattr(raw, 'pubkey', None)
    if value is None and isinstance(raw, dict):
        value = raw.get('key', raw.get('pubkey', ''))
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value or '').strip()

def reserve(cell: Any) -> int:
    value = getattr(cell, 'reserve', None)
    if value is None:
        value = getattr(cell, 'salt', 0)
    return int(value or 0)

def amount(cell: Any) -> int:
    return reserve(cell)

def id6(raw: Any) -> str:
    text = key(raw) if not isinstance(raw, str) else str(raw).strip()
    if not text:
        return ''
    if len(text) == 64:
        try:
            bytes.fromhex(text)
            return text[:16]
        except Exception:
            pass
    return text[:NAME_W]

def currentLock(cellstate: Any, *, key0: Any=None, idx: Optional[int]=None) -> str:
    cells = list(getattr(cellstate, 'cells', []) or [])
    if idx is not None and 0 <= int(idx) < len(cells):
        cell = cells[int(idx)]
    else:
        found = str(key0 or '').strip()
        cell = None
        for candidate in cells:
            if key(candidate) == found:
                cell = candidate
                break
    if cell is None:
        return ''
    lock = getattr(cell, 'lock', None)
    if lock is not None:
        child = getattr(lock, 'child', None)
        if child is not None:
            return str(child or '')
    lockset = getattr(cell, 'lockset', None)
    if lockset is None:
        return ''
    child = getattr(lockset, 'child_lock', None)
    if child is None:
        child = getattr(lockset, 'child_dream_lock', None)
    if child is None:
        child = getattr(lockset, 'cell_lock', None)
    return str(child or '')

def selfKey(state: Any) -> str:
    return makeSelf(getattr(state, 'self', None))[1]

def selfSoul(state: Any) -> str:
    return makeSelf(getattr(state, 'self', None))[0]



def qByKey(state: Any, raw: Any) -> Optional[int]:
    needle = str(raw or '').strip()
    if not needle:
        return None
    cells = list(getattr(state, 'cells', []) or [])
    for i, cell in enumerate(cells):
        if key(cell) == needle:
            return i
    return None

def Qof(state: Any) -> Optional[int]:
    cells = list(getattr(state, 'cells', []) or [])
    mine = selfKey(state)
    if not cells or not mine:
        return None
    for i, cell in enumerate(cells):
        if key(cell) == mine:
            return i
    return None

def resolveRank(state: Any) -> Optional[int]:
    q = Qof(state)
    return None if q is None else q + 1

def keyAt(state: Any, q: int) -> str:
    cells = list(getattr(state, 'cells', []) or [])
    return key(cells[int(q)]) if 0 <= int(q) < len(cells) else ''

def actionPreview(action: Action) -> str:
    return str(ACTION.get(action, {}).get('preview', action.value))

def actionDesc(action: Action) -> str:
    return str(ACTION.get(action, {}).get('desc', ''))

def actionFloor(action: Action, *, rank: Optional[int]=None, q: Optional[int]=None, Q: Optional[int]=None) -> int:
    seat_q = q if q is not None else Q
    if action == Action.DEFECT:
        seat = seat_q if seat_q is not None else None if rank is None else int(rank) - 1
        if seat is None:
            return 1000
        return 10000 if geometry.general(seat) else 1000
    try:
        return int(ACTION.get(action, {}).get('floor', 0) or 0)
    except Exception:
        return 0

def actionSpineLabel(action: Action, *, rank: Optional[int]=None, q: Optional[int]=None, Q: Optional[int]=None) -> Optional[str]:
    label = ACTION.get(action, {}).get('label')
    if label:
        return str(label)
    if action == Action.DEFECT:
        return 'MYRIAD' if actionFloor(action, rank=rank, q=q, Q=Q) >= 10000 else 'COHORT'
    return None

def actionNeedsTarget(action: Action) -> bool:
    return bool(ACTION.get(action, {}).get('needsTarget', False))

def actionHasArmPhase(action: Action) -> bool:
    return bool(ACTION.get(action, {}).get('arm', False))

def defectViable(state: Any, mine: int, target: int) -> bool:
    if target == mine:
        return False
    cells = list(getattr(state, 'cells', []) or [])
    if not (0 <= mine < len(cells) and 0 <= target < len(cells)):
        return False
    return amount(cells[target]) < amount(cells[mine]) and geometry.file(target) != geometry.file(mine)

def moveBoard(state: Any, city: int, arrow: str, current: Action) -> int:
    old = int(city or 0) % geometry.cells
    mine = 0 if Qof(state) is None else int(Qof(state)) % geometry.cells
    if current == Action.DEFECT:
        row = geometry.rank(old)
        col = geometry.file(old)
        if arrow in ('C', 'D'):
            step = 1 if arrow == 'C' else -1
            for hop in range(1, geometry.cols + 1):
                q = (col + step * hop) % geometry.cols * geometry.rows + row
                if defectViable(state, mine, q):
                    return q
            return old
        if arrow in ('A', 'B'):
            step = -1 if arrow == 'A' else 1
            for hop in range(1, geometry.rows + 1):
                q = col * geometry.rows + (row + step * hop) % geometry.rows
                if defectViable(state, mine, q):
                    return q
            return old
        return old
    cand = geometry.move(old, arrow)
    if current in (Action.WHISPER, Action.PURGE) and cand == mine:
        q = cand
        for _ in range(geometry.cells - 1):
            q = geometry.move(q, arrow)
            if q != mine:
                return q
    return cand

def targets(state: Any, mine: int, current: Action, target: Optional[int]) -> Tuple[int, ...]:
    cells = list(getattr(state, 'cells', []) or [])
    if current == Action.WHISPER:
        return () if target is None else (int(target),)
    if current == Action.DEFECT:
        myfile = geometry.file(int(mine))
        return tuple((q for q, _cell in enumerate(cells) if q != int(mine) and geometry.file(q) == myfile))
    if current == Action.PURGE:
        return () if target is None else (int(target),)
    if current == Action.RALLY:
        myfile = geometry.file(int(mine))
        return tuple((q for q, _cell in enumerate(cells) if q != int(mine) and geometry.file(q) == myfile))
    if current == Action.WRATH:
        return tuple((q for q, _cell in enumerate(cells) if q != int(mine)))
    return ()

def purgeViable(state: Any, mine: int, target: Optional[int]) -> bool:
    cells = list(getattr(state, 'cells', []) or [])
    if target is None:
        return False
    q = int(target)
    if not 0 <= q < len(cells):
        return False
    if q == int(mine):
        return False
    return bool(key(cells[q]))

def purgeTarget(state: Any, q: Optional[int]) -> str:
    if q is None:
        return ''
    return keyAt(state, int(q))

def purgeFlavor() -> str:
    return 'Friends are friends until the end.'

def defecttargetkey(state: Any, q: Optional[int]) -> str:
    if q is None:
        return ''
    return keyAt(state, int(q))

def pairs(state: Any, qs: Iterable[int], total: int) -> Tuple[Tuple[str, int], ...]:
    cells = list(getattr(state, 'cells', []) or [])
    picked = [cells[int(q)] for q in qs if 0 <= int(q) < len(cells)]
    if not picked or total <= 0:
        return ()
    if len(picked) == 1:
        found = key(picked[0])
        return () if not found else ((found, int(total)),)
    return geometry.split(int(total), picked, by=amount)
__all__ = ['geometry', 'Geometry', 'Cell', 'State', 'Q', 'Intent', 'UiCache', 'Focus', 'Action', 'MENU', 'ACTION', 'BOARD_COLS', 'BOARD_ROWS', 'CELL_COUNT', 'TERM_W', 'FRAME_PAD', 'INNER_W', 'NAME_W', 'MSG_MAX', 'COST_W', 'ANSI_RE', 'RESET', 'ASH', 'WHITE', 'SALT', 'EMBER', 'FLICKER1', 'FLICKER2', 'flickerPair', 'palette', 'makeCell', 'makeSelf', 'makeState', 'vislen', 'clipw', 'centerw', 'padw', 'clipTerm', 'centerTerm', 'frameTextScreen', 'cleanDraft', 'msgNorm', 'fmtSpineCost', 'parseMonument', 'monumentAnchorCol', 'key', 'reserve', 'amount', 'id6', 'currentLock', 'selfKey', 'selfSoul', 'qByKey', 'Qof', 'resolveRank', 'keyAt', 'actionPreview', 'actionDesc', 'actionFloor', 'actionSpineLabel', 'actionNeedsTarget', 'actionHasArmPhase', 'defectViable', 'moveBoard', 'targets', 'purgeViable', 'purgeTarget', 'purgeFlavor', 'pairs']
