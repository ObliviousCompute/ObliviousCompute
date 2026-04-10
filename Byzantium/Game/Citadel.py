from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import Forge
from Forge import Action, Focus, Menu, Cache, geometry

Core = None


@dataclass
class Relay:
    cache: Any = None
    statecache: Any = None
    ash: list | None = None

    def BindCache(self, cache: Any) -> Any:
        self.cache = cache
        if self.ash is None:
            self.ash = []
        if cache is not None:
            ApplyAsh(cache, self.ash)
            if self.statecache is not None:
                Project(cache, self.statecache)
        return cache

    def SurfaceState(self, value: Any) -> Any:
        self.statecache = Forge.MakeState(value)
        if self.cache is not None:
            Project(self.cache, self.statecache)
            ApplyAsh(self.cache, self.ash or [])
        return self.statecache

    @property
    def State(self) -> Any:
        return self.statecache

    @State.setter
    def State(self, value: Any) -> None:
        self.SurfaceState(value)

    def Intent(self, value: Any) -> Any:
        return Send(value)

    def Ashfall(self, value: Any) -> Any:
        if self.ash is None:
            self.ash = []
        if not isinstance(value, dict):
            return None
        sender = str(value.get('sender', '') or '').strip()
        kind = str(value.get('action', '') or value.get('kind', '') or '').strip().lower()
        text = str(value.get('text', '') or '')
        rawtext = str(value.get('rawtext', '') or '')
        total = int(value.get('total', 0) or 0)
        entry = {
            'kind': kind or 'ash',
            'sender': sender,
            'name': sender.ljust(Forge.NameWidth)[:Forge.NameWidth],
            'left': 'Defected' if kind == 'defect' else Forge.SpineCost(total, width=Forge.SaltWidth, signed=True),
            'text': text,
            'rawtext': rawtext,
            'total': total,
        }
        self.ash.append(entry)
        self.ash = self.ash[-7:]
        if self.cache is not None:
            ApplyAsh(self.cache, self.ash)
        return entry

    def PeerDisplay(self, key: str) -> str:
        key = str(key or '').strip()
        if not key:
            return ''
        state = self.statecache
        if state is None:
            return key
        q = Forge.QxKey(state, key)
        if q is None:
            return key
        cells = list(getattr(state, 'cells', []) or [])
        if not (0 <= int(q) < len(cells)):
            return key
        soul = str(getattr(cells[int(q)], 'soul', getattr(cells[int(q)], 'Soul', '')) or '').strip()
        return soul or key


Citadel = Relay()


def ApplyAsh(cache: Any, ash: list) -> None:
    payload = list(ash or [])
    cache.feed = payload
    cache.feedcount = len(cache.feed)


def KeyxQ(state: Any, q: int) -> str:
    return str(Forge.KeyxQ(state, int(q)) or '').strip()


def SelfQ(state: Any) -> int:
    q = Forge.SelfQ(state)
    return 0 if q is None else int(q)


def MenuAction(cache: Cache) -> Action:
    return Menu[int(cache.menuq) % len(Menu)]


def Floor(current: Action, state: Any) -> int:
    return int(Forge.ActionFloor(current, rank=Forge.ResolveRank(state), q=Forge.SelfQ(state)) or 0)


def DefectWrap(state: Any, mine: int, start: int) -> Optional[int]:
    cells = list(getattr(state, 'cells', []) or [])
    count = len(cells)
    if count <= 0:
        return None
    q = int(start) % count
    for hop in range(count):
        seat = (q + hop) % count
        if Forge.DefectViable(state, int(mine), seat):
            return seat
    return None


def BindState(cache: Cache, state: Any, q: Optional[int] = None) -> Optional[int]:
    if q is None:
        q = getattr(cache, 'stateq', 0)
    cache.stateq = int(q)
    cache.statekey = KeyxQ(state, cache.stateq)
    return cache.stateq


def BindTarget(cache: Cache, state: Any, q: Optional[int] = None) -> Optional[int]:
    if q is None:
        cache.targetq = None
        cache.targetkey = ''
        return None
    cache.targetq = int(q)
    cache.targetkey = KeyxQ(state, cache.targetq)
    return cache.targetq


def Project(cache: Cache, state: Any) -> None:
    cache.state = state
    cache.activerequest = ''
    cache.monuments = list(getattr(state, 'monument', ()) or ())
    cache.intent.q.self = Forge.SelfQ(state)
    RebindFocus(cache, state)
    cache.SyncIntent()


def RebindFocus(cache: Cache, state: Any) -> None:
    cells = list(getattr(state, 'cells', []) or [])
    count = len(cells)
    if count <= 0:
        cache.stateq = 0
        cache.statekey = ''
        cache.targetq = None
        cache.targetkey = ''
        return

    mine = SelfQ(state) % count
    current = MenuAction(cache)
    stateq = Forge.QxKey(state, getattr(cache, 'statekey', ''))
    targetq = Forge.QxKey(state, getattr(cache, 'targetkey', ''))

    if current in (Action.Rally, Action.Wrath):
        BindState(cache, state, mine)
        BindTarget(cache, state, None)
        return

    if current == Action.Whisper:
        if targetq is None and stateq is not None:
            targetq = stateq
        if targetq is None or targetq == mine:
            targetq = int(getattr(cache, 'stateq', mine) or mine) % count
            if targetq == mine:
                targetq = (mine + 1) % count
        BindState(cache, state, int(targetq))
        BindTarget(cache, state, None if int(targetq) == mine else int(targetq))
        return

    if current == Action.Purge:
        if stateq is None:
            stateq = mine
        BindState(cache, state, int(stateq))
        BindTarget(cache, state, int(stateq))
        return

    if current == Action.Defect:
        if targetq is None and stateq is not None and stateq != mine:
            targetq = stateq
        if targetq is not None and not Forge.DefectViable(state, mine, int(targetq)):
            targetq = DefectWrap(state, mine, int(targetq))
        if targetq is None or int(targetq) == mine:
            start = int(getattr(cache, 'stateq', mine) or mine) % count
            if start == mine:
                start = (mine + 1) % count
            targetq = DefectWrap(state, mine, start)
        BindState(cache, state, mine if targetq is None else int(targetq))
        BindTarget(cache, state, None if targetq is None else int(targetq))
        return

    BindState(cache, state, mine if stateq is None else int(stateq))
    BindTarget(cache, state, None if targetq is None else int(targetq))


def Say(cache: Cache, chan: str, line: str) -> None:
    feed = list(getattr(cache, 'feed', []) or [])
    feed.append((chan, line))
    cache.feed = feed[-7:]
    cache.feedcount = len(cache.feed)


def Sync(cache: Cache, state: Any) -> Any:
    state = Forge.MakeState(state)
    cache.focus = cache.focus or Focus.Menu
    cache.intent.focus = cache.focus
    cache.intent.action = MenuAction(cache)
    cache.intent.q.self = Forge.SelfQ(state)
    Project(cache, state)
    cache.intent.q.city = int(getattr(cache, 'stateq', 0) or 0)
    cache.intent.q.target = getattr(cache, 'targetq', None)
    cache.intent.amount = max(0, int(getattr(cache, 'salt', 0) or 0))
    cache.intent.text = Forge.CleanDraft(getattr(cache, 'text', ''))
    cache.intent.kind = str(getattr(cache.intent.action, 'value', cache.intent.action) or '').lower()
    return state


def MoveMenuIndex(cache: Cache, delta: int) -> None:
    cache.menuq = (int(cache.menuq) + int(delta)) % len(Menu)
    cache.intent.action = MenuAction(cache)


def ResetIntent(cache: Cache) -> None:
    cache.focus = Focus.Menu
    cache.menuq = 0
    cache.stateq = int(cache.intent.q.self or 0)
    cache.statekey = KeyxQ(cache.state, cache.stateq) if getattr(cache, 'state', None) is not None else ''
    cache.targetq = None
    cache.targetkey = ''
    cache.salt = 1
    cache.text = ''
    cache.SyncIntent()


def MoveBoard(cache: Cache, state: Any, arrow: str, current: Action) -> None:
    if current == Action.Purge:
        cache.stateq = Forge.geometry.Move(getattr(cache, 'stateq', 0), arrow)
    else:
        cache.stateq = Forge.MoveTable(state, getattr(cache, 'stateq', 0), arrow, current)
    cache.statekey = KeyxQ(state, cache.stateq)
    cache.targetq = int(cache.stateq)
    cache.targetkey = KeyxQ(state, cache.targetq)
    cache.SyncIntent()


def Sender() -> Optional[Callable[[Any], Any]]:
    live = Core
    if live is None:
        return None
    fn = getattr(live, 'Intent', None)
    if callable(fn):
        return fn
    return None


def Send(value: Any) -> Any:
    payload = dict(value) if isinstance(value, dict) else {
        'kind': str(getattr(value, 'kind', '') or getattr(getattr(value, 'action', ''), 'value', getattr(value, 'action', '')) or '').lower(),
        'pairs': list(getattr(value, 'pairs', []) or []),
        'text': Forge.MessageNorm(getattr(value, 'text', '')),
        'lock': getattr(value, 'lock', None),
        'key': str(getattr(value, 'key', '') or '').strip(),
    }
    fn = Sender()
    if fn is None:
        return None
    try:
        return fn(payload)
    except Exception:
        return None


def TargetCells(state: Any, mine: int, current: Action, target: Optional[int]):
    cells = list(getattr(state, 'cells', []) or [])
    qs = Forge.Targets(state, int(mine), current, target)
    return [cells[int(q)] for q in qs if 0 <= int(q) < len(cells)]


def IntentPairs(state: Any, mine: int, current: Action, target: Optional[int], total: int):
    cells = TargetCells(state, mine, current, target)
    if current == Action.Purge:
        return ()
    if current == Action.Whisper:
        if len(cells) == 1:
            found = Forge.Key(cells[0])
            return () if not found else ((found, int(total)),)
        return Forge.geometry.Split(int(total), cells)
    if current == Action.Defect:
        victimkey = Forge.DefectTargetKey(state, target)
        legs = list(Forge.geometry.Split(int(total), cells))
        if victimkey:
            legs.append((victimkey, 0))
        return tuple(legs)
    return Forge.geometry.Split(int(total), cells)


def IntentLock(state: Any, mine: int):
    cells = list(getattr(state, 'cells', []) or [])
    if 0 <= int(mine) < len(cells):
        return getattr(cells[int(mine)], 'lock', None)
    return None


def Submit(cache: Cache, state: Any, current: Action) -> None:
    mine = SelfQ(state)
    total = int(getattr(cache, 'salt', 0) or 0)
    text = str(getattr(cache, 'text', '') or '')
    RebindFocus(cache, state)
    target = getattr(cache, 'targetq', None)
    try:
        intent = cache.SyncIntent()
        intent.kind = str(getattr(current, 'value', current) or '').lower()
        intent.pairs = IntentPairs(state, mine, current, target, total)
        intent.text = Forge.MessageNorm(text)
        intent.lock = IntentLock(state, mine)
        if current == Action.Purge:
            intent.key = Forge.PurgeTarget(state, target)
            intent.text = Forge.PurgeFlavor()
            intent.pairs = ()
            intent.lock = None
        Send(intent)
    except Exception as exc:
        Say(cache, 'ash', str(exc) or 'intent failed')
    finally:
        ResetIntent(cache)


def MoveMenu(cache: Cache, state: Any, kind: str, value: Optional[str]):
    current = MenuAction(cache)
    if kind == 'Arrow':
        if value == 'C':
            MoveMenuIndex(cache, +1)
        elif value == 'D':
            MoveMenuIndex(cache, -1)
        return (cache, state, False)
    if kind != 'Enter':
        return (cache, state, False)
    if current == Action.Exit:
        return (cache, state, True)
    if current == Action.Purge:
        cache.focus = Focus.TableMove
        mine = SelfQ(state) % geometry.cells
        BindState(cache, state, mine)
        BindTarget(cache, state, None)
        cache.text = ''
        cache.SyncIntent()
        return (cache, state, False)
    if current == Action.Monument:
        cache.banner = not bool(getattr(cache, 'banner', True))
        cache.SyncIntent()
        return (cache, state, False)
    if current == Action.Lore:
        cache.lore = not bool(getattr(cache, 'lore', False))
        if cache.lore:
            cache.lorescroll = 0
        cache.SyncIntent()
        return (cache, state, False)
    if current in (Action.Rally, Action.Wrath):
        cache.focus = Focus.TableLock
        cache.salt = Floor(current, state)
        cache.text = ''
        BindState(cache, state, SelfQ(state) % geometry.cells)
        BindTarget(cache, state, None)
        cache.SyncIntent()
        return (cache, state, False)
    if current == Action.Whisper:
        cache.focus = Focus.TableMove
        cache.salt = 1
        cache.text = ''
        BindState(cache, state, (SelfQ(state) + 1) % geometry.cells)
        cache.SyncIntent()
        return (cache, state, False)
    if current == Action.Defect:
        cache.focus = Focus.TableMove
        cache.salt = Floor(current, state)
        mine = SelfQ(state) % geometry.cells
        start = int(getattr(cache, 'stateq', mine) or mine) % geometry.cells
        if start == mine:
            start = (mine + 1) % geometry.cells
        target = DefectWrap(state, mine, start)
        BindState(cache, state, mine if target is None else target)
        cache.SyncIntent()
        return (cache, state, False)
    return (cache, state, False)


def MoveTable(cache: Cache, state: Any, kind: str, value: Optional[str]):
    current = MenuAction(cache)
    if kind == 'Arrow' and value in ('A', 'B', 'C', 'D'):
        MoveBoard(cache, state, value, current)
        return (cache, state, False)
    if kind != 'Enter':
        return (cache, state, False)
    if current == Action.Whisper:
        mine = SelfQ(state) % geometry.cells
        if int(cache.stateq) == mine:
            cache.stateq = (mine + 1) % geometry.cells
        BindTarget(cache, state, int(cache.stateq))
        cache.focus = Focus.Spine
        cache.SyncIntent()
        return (cache, state, False)
    if current == Action.Purge:
        mine = SelfQ(state) % geometry.cells
        target = int(cache.stateq) % geometry.cells
        if mine != target and not Forge.PurgeViable(state, mine, target):
            ResetIntent(cache)
            return (cache, state, False)
        BindTarget(cache, state, target)
        Submit(cache, state, current)
        return (cache, state, False)
    if current == Action.Defect:
        mine = SelfQ(state) % geometry.cells
        target = int(cache.stateq) % geometry.cells
        if not Forge.DefectViable(state, mine, target):
            ResetIntent(cache)
            return (cache, state, False)
        BindTarget(cache, state, target)
        cache.focus = Focus.Spine
        cache.text = ''
        cache.SyncIntent()
        return (cache, state, False)
    return (cache, state, False)


def LockTable(cache: Cache, state: Any, kind: str, value: Optional[str]):
    if kind == 'Enter':
        cache.focus = Focus.Spine
        cache.text = ''
        cache.SyncIntent()
    return (cache, state, False)


def EditSpine(cache: Cache, state: Any, kind: str, value: Optional[str]):
    current = MenuAction(cache)
    if kind == 'Arrow':
        if current == Action.Defect:
            return (cache, state, False)
        step = Floor(current, state)
        cells = list(getattr(state, 'cells', []) or [])
        mine = SelfQ(state) % geometry.cells
        have = Forge.Amount(cells[mine]) if 0 <= mine < len(cells) else 0
        cap = have if have >= step else step
        if value == 'A':
            cache.salt = min(cap, int(cache.salt) + step)
        elif value == 'B':
            cache.salt = max(step, int(cache.salt) - step)
        cache.SyncIntent()
        return (cache, state, False)
    if kind == 'Backspace':
        if current in (Action.Whisper, Action.Rally, Action.Wrath, Action.Defect):
            cache.text = (cache.text or '')[:-1]
            cache.SyncIntent()
        return (cache, state, False)
    if kind == 'Character' and value and value.isprintable() and current in (Action.Whisper, Action.Rally, Action.Wrath, Action.Defect):
        if len(cache.text) < Forge.MessageMax:
            cache.text += value
            cache.SyncIntent()
        return (cache, state, False)
    if kind != 'Enter':
        return (cache, state, False)
    if current in (Action.Whisper, Action.Rally, Action.Wrath, Action.Defect):
        Submit(cache, state, current)
    else:
        ResetIntent(cache)
    return (cache, state, False)


def DispatchCore(cache: Cache, state: Any, token: Tuple[str, Optional[str]]):
    kind, value = token
    if kind == 'Interrupt':
        return (cache, state, True)
    if bool(getattr(cache, 'lore', False)):
        if kind == 'Arrow':
            offset = int(getattr(cache, 'lorescroll', 0) or 0)
            if value == 'A':
                cache.lorescroll = max(0, offset - 1)
            elif value == 'B':
                cache.lorescroll = offset + 1
            cache.SyncIntent()
            return (cache, state, False)
        if kind == 'Enter' or (kind == 'Character' and value == ' '):
            cache.lore = False
            cache.lorescroll = 0
            cache.SyncIntent()
        return (cache, state, False)
    if kind == 'Character' and value == ' ' and cache.focus in (Focus.TableMove, Focus.TableLock):
        ResetIntent(cache)
        return (cache, state, False)
    if cache.focus == Focus.Menu:
        return MoveMenu(cache, state, kind, value)
    if cache.focus == Focus.TableMove:
        return MoveTable(cache, state, kind, value)
    if cache.focus == Focus.TableLock:
        return LockTable(cache, state, kind, value)
    if cache.focus == Focus.Spine:
        return EditSpine(cache, state, kind, value)
    return (cache, state, False)


def ParseKeys(buffer: str):
    out = []
    i = 0
    while i < len(buffer):
        c = buffer[i]
        if c == '\x03':
            out.append(('Interrupt', None))
            i += 1
            continue
        if c == '\n':
            out.append(('Enter', None))
            i += 1
            continue
        if c in ('\x7f', '\x08'):
            out.append(('Backspace', None))
            i += 1
            continue
        if c != '\x1b':
            out.append(('Character', c))
            i += 1
            continue
        if i + 1 >= len(buffer):
            break
        n1 = buffer[i + 1]
        if n1 == 'O' and i + 2 < len(buffer) and buffer[i + 2] in ('A', 'B', 'C', 'D'):
            out.append(('Arrow', buffer[i + 2]))
            i += 3
            continue
        if n1 == '[':
            j = i + 2
            seq = ''
            while j < len(buffer):
                d = buffer[j]
                if d in ('A', 'B', 'C', 'D'):
                    seq += d
                    out.append(('PortBump', d) if seq.startswith('1;2') or seq.startswith('2') else ('Arrow', d))
                    i = j + 1
                    break
                if d == '~':
                    i = j + 1
                    break
                seq += d
                j += 1
            else:
                break
            continue
        i += 1
    return (out, buffer[i:])


def InitCache(state: Any):
    state = Forge.MakeState(state)
    cache = Cache(feed=[], name='')
    cache.focus = Focus.Title
    cache.state = state
    cache.activerequest = 'waitingstate'
    if Citadel.ash is None:
        Citadel.ash = []
    ApplyAsh(cache, Citadel.ash)
    cache.lorescroll = 0
    cache.intent.q.self = Forge.SelfQ(state)
    Citadel.BindCache(cache)
    Citadel.State = state
    return cache


def BindCache(cache: Any) -> Any:
    return Citadel.BindCache(cache)


def BindCore(coreobj: Any) -> Any:
    global Core
    Core = coreobj
    return coreobj


def State(value: Any = None) -> Any:
    if value is None:
        return Citadel.State
    Citadel.State = value
    return Citadel.State


def Intent(value: Any) -> Any:
    return Citadel.Intent(value)


def Ashfall(value: Any) -> Any:
    return Citadel.Ashfall(value)


def PeerDisplayLabel(key: str) -> str:
    return Citadel.PeerDisplay(key)


InputBuffer = ParseKeys
Dispatch = DispatchCore
