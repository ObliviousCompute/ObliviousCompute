from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Callable
import time
import Forge
from Forge import Action, Focus, MENU, UiCache, geometry
core = None

@dataclass
class Relay:
    cache: Any = None
    _state: Any = None
    ash: list = None

    def bindcache(self, cache: Any) -> Any:
        self.cache = cache
        if self.ash is None:
            self.ash = []
        if cache is not None:
            cache.ash = list(self.ash)
            cache.feed = list(self.ash)
            cache.visible_feed_count = len(cache.feed)
        if cache is not None and self._state is not None:
            try:
                _surface_cache(cache, self._state)
            except Exception:
                pass
        return cache

    def surfacestate(self, value: Any) -> Any:
        value = Forge.makeState(value)
        self._state = value
        try:
            pass
        except Exception:
            pass
        if self.cache is not None:
            try:
                _surface_cache(self.cache, value)
                if self.ash is None:
                    self.ash = []
                self.cache.ash = list(self.ash)
                self.cache.feed = list(self.ash)
                self.cache.visible_feed_count = len(self.cache.feed)
            except Exception:
                pass
        return self._state

    @property
    def state(self) -> Any:
        return self._state

    @state.setter
    def state(self, value: Any) -> None:
        self.surfacestate(value)

    def intent(self, value: Any) -> Any:
        return send(value)

    def ashfall(self, value: Any) -> Any:
        if self.ash is None:
            self.ash = []
        if not isinstance(value, dict):
            return None
        sender = str(value.get('sender', '') or '').strip()
        actionkind = str(value.get('actionkind', '') or value.get('kind', '') or '').strip().upper()
        text = str(value.get('text', '') or '')
        rawtext = str(value.get('rawtext', '') or '')
        total = int(value.get('total', 0) or 0)
        name = sender.ljust(Forge.NAME_W)[:Forge.NAME_W]
        left = 'Defected' if actionkind == 'DEFECT' else Forge.fmtSpineCost(total, width=Forge.COST_W, signed=True)
        entry = {
            'kind': actionkind or 'ASH',
            'sender': sender,
            'name': name,
            'left': left,
            'text': text,
            'rawtext': rawtext,
            'total': total,
        }
        self.ash.append(entry)
        self.ash = self.ash[-7:]
        if self.cache is not None:
            self.cache.ash = list(self.ash)
            self.cache.feed = list(self.ash)
            self.cache.visible_feed_count = len(self.cache.feed)
        return entry

    def peerdisplay(self, key: str) -> str:
        key = str(key or '').strip()
        if not key:
            return ''
        cells = list(getattr(self._state, 'cells', []) or [])
        for cell in cells:
            if _cell_key(cell) == key:
                soul = str(getattr(cell, 'soul', '') or '').strip()
                if soul:
                    return soul
            else:
                pass
        short = Forge.id6(key)
        return short or key[:6].upper()
citadel = Relay()
_CITADEL = citadel

def _cell_key(cell: Any) -> str:
    if hasattr(Forge, 'key'):
        return str(Forge.key(cell) or '').strip()
    value = getattr(cell, 'key', getattr(cell, 'pubkey', ''))
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return str(value or '').strip()

def _floor(current: Action, state: Any) -> int:
    rank = Forge.resolveRank(state) if hasattr(Forge, 'resolveRank') else None
    q = Forge.Qof(state) if hasattr(Forge, 'Qof') else None
    if hasattr(Forge, 'actionFloor'):
        try:
            return int(Forge.actionFloor(current, rank=rank, q=q) or 0)
        except TypeError:
            return int(Forge.actionFloor(current, rank=rank, Q=q) or 0)
    return 0

def _first_defect(state: Any, mine: int, start: int):
    cells = list(getattr(state, 'cells', []) or [])
    n = len(cells)
    if n <= 0:
        return None
    start = int(start) % n
    if hasattr(Forge, 'defectViable'):
        for hop in range(n):
            q = (start + hop) % n
            if Forge.defectViable(state, int(mine), q):
                return q
        else:
            pass
    return None

def _pubkey_at(state: Any, q: int) -> str:
    if hasattr(Forge, 'keyAt'):
        return str(Forge.keyAt(state, int(q)) or '').strip()
    cells = list(getattr(state, 'cells', []) or [])
    if 0 <= int(q) < len(cells):
        return _cell_key(cells[int(q)])
    return ''

def _q_by_key(state: Any, raw: Any) -> Optional[int]:
    if hasattr(Forge, 'qByKey'):
        found = Forge.qByKey(state, raw)
        return None if found is None else int(found)
    needle = str(raw or '').strip()
    if not needle:
        return None
    cells = list(getattr(state, 'cells', []) or [])
    for i, cell in enumerate(cells):
        if _cell_key(cell) == needle:
            return i
    return None

def _bind_state_key(cache: UiCache, state: Any, q: Optional[int]=None) -> Optional[int]:
    q0 = q
    if q0 is None:
        q0 = getattr(cache, 'stateQ', 0)
    key = _pubkey_at(state, int(q0))
    cache.stateKey = key
    return None if q0 is None else int(q0)

def _bind_target_key(cache: UiCache, state: Any, q: Optional[int]=None) -> Optional[int]:
    if q is None:
        cache.targetKey = ''
        cache.targetQ = None
        return None
    q0 = int(q)
    cache.targetQ = q0
    cache.targetKey = _pubkey_at(state, q0)
    return q0

def _rebind_focus(cache: UiCache, state: Any) -> None:
    cells = list(getattr(state, 'cells', []) or [])
    n = len(cells)
    if n <= 0:
        cache.stateQ = 0
        cache.targetQ = None
        cache.stateKey = ''
        cache.targetKey = ''
        return
    me = selfQorzero(state) % geometry.cells
    current = action(cache)
    stateq = _q_by_key(state, getattr(cache, 'stateKey', ''))
    targetq = _q_by_key(state, getattr(cache, 'targetKey', ''))
    if current in (Action.RALLY, Action.WRATH):
        cache.stateQ = me
        cache.stateKey = _pubkey_at(state, me)
        cache.targetQ = None
        cache.targetKey = ''
        return
    if current in (Action.WHISPER, Action.PURGE):
        if targetq is None and stateq is not None:
            targetq = stateq
        if targetq is None or targetq == me:
            fallback = int(getattr(cache, 'stateQ', me) or me) % n
            if fallback == me:
                fallback = (me + 1) % n
            targetq = fallback
        cache.stateQ = int(targetq)
        cache.stateKey = _pubkey_at(state, cache.stateQ)
        cache.targetQ = int(targetq) if targetq != me else None
        cache.targetKey = _pubkey_at(state, cache.targetQ) if cache.targetQ is not None else ''
        return
    if current == Action.DEFECT:
        if targetq is None and stateq is not None and stateq != me:
            targetq = stateq
        if targetq is not None and not _defect_viable(state, me, targetq):
            targetq = _first_defect(state, me, targetq)
        if targetq is None or targetq == me:
            start = int(getattr(cache, 'stateQ', me) or me) % n
            if start == me:
                start = (me + 1) % n
            targetq = _first_defect(state, me, start)
        cache.stateQ = me if targetq is None else int(targetq)
        cache.stateKey = _pubkey_at(state, cache.stateQ)
        cache.targetQ = None if targetq is None else int(targetq)
        cache.targetKey = _pubkey_at(state, cache.targetQ) if cache.targetQ is not None else ''
        return
    if stateq is None:
        stateq = me
    cache.stateQ = int(stateq)
    cache.stateKey = _pubkey_at(state, cache.stateQ)
    if targetq is None:
        cache.targetQ = None
        cache.targetKey = ''
    else:
        cache.targetQ = int(targetq)
        cache.targetKey = _pubkey_at(state, cache.targetQ)

def _surface_cache(cache: UiCache, state: Any) -> None:
    cache.state = state
    cache.pending_request = ''
    cache.monuments = list(getattr(state, 'monument', ()) or ())
    cache.intent.Q.self = Forge.Qof(state)
    _rebind_focus(cache, state)
    cache.syncIntent()

def _defect_viable(state: Any, mine: int, target: int) -> bool:
    if hasattr(Forge, 'defectViable'):
        return bool(Forge.defectViable(state, int(mine), int(target)))
    return int(target) != int(mine)

def say(cache: UiCache, chan: str, line: str) -> None:
    feed = list(getattr(cache, 'feed', []) or [])
    feed.append((chan, line))
    cache.feed = feed[-7:]
    cache.visible_feed_count = len(cache.feed)

def action(cache: UiCache) -> Action:
    return MENU[int(cache.menuQ) % len(MENU)]

def sync(cache: UiCache, state: Any) -> Any:
    state = Forge.makeState(state)
    cache.state = state
    cache.focus = cache.focus or Focus.MENU
    cache.intent.focus = cache.focus
    cache.intent.action = action(cache)
    cache.intent.Q.self = Forge.Qof(state)
    _rebind_focus(cache, state)
    cache.intent.Q.city = int(getattr(cache, 'stateQ', 0) or 0)
    cache.intent.Q.target = getattr(cache, 'targetQ', None)
    cache.intent.amount = max(0, int(getattr(cache, 'salt', 0) or 0))
    cache.intent.text = Forge.cleanDraft(getattr(cache, 'text', ''))
    cache.intent.kind = str(getattr(cache.intent.action, 'value', cache.intent.action) or '').upper()
    return state

def menu(cache: UiCache, delta: int) -> None:
    cache.menuQ = (int(cache.menuQ) + int(delta)) % len(MENU)
    cache.intent.action = action(cache)

def reset(cache: UiCache) -> None:
    cache.focus = Focus.MENU
    cache.menuQ = 0
    cache.stateQ = int(cache.intent.Q.self or 0)
    cache.stateKey = _pubkey_at(cache.state, cache.stateQ) if getattr(cache, 'state', None) is not None else ''
    cache.targetQ = None
    cache.targetKey = ''
    cache.salt = 1
    cache.text = ''
    cache.syncIntent()

def selfQorzero(state: Any) -> int:
    q = Forge.Qof(state)
    return 0 if q is None else q

def moveboard(cache: UiCache, arrow: str, current: Action, state: Any) -> None:
    cache.stateQ = Forge.moveBoard(state, getattr(cache, 'stateQ', 0), arrow, current)
    cache.stateKey = _pubkey_at(state, cache.stateQ)
    cache.syncIntent()

def floor(current: Action, state: Any) -> int:
    return _floor(current, state)

def _resolve_sender() -> Optional[Callable[[Any], Any]]:
    global core
    live = core
    if live is None:
        return None
    for name in ('intent', 'submit', 'transact'):
        fn = getattr(live, name, None)
        if callable(fn):
            return fn
    else:
        pass
    return None

def send(value: Any) -> Any:
    if isinstance(value, dict):
        payload = dict(value)
    else:
        payload = {'kind': str(getattr(value, 'kind', '') or getattr(getattr(value, 'action', ''), 'value', getattr(value, 'action', '')) or '').upper(), 'pairs': list(getattr(value, 'pairs', []) or []), 'text': Forge.msgNorm(getattr(value, 'text', '')), 'lock': getattr(value, 'lock', None), 'key': str(getattr(value, 'key', '') or '').strip()}
    try:
        pass
    except Exception:
        pass
    fn = _resolve_sender()
    if fn is None:
        return None
    try:
        out = fn(payload)
        return out
    except Exception as exc:
        return None

def _target_cells(state: Any, mine: int, current: Action, target: Optional[int]):
    cells = list(getattr(state, 'cells', []) or [])
    qs = Forge.targets(state, int(mine), current, target) if hasattr(Forge, 'targets') else ()
    return [cells[int(q)] for q in qs if 0 <= int(q) < len(cells)]

def _intent_pairs(state: Any, mine: int, current: Action, target: Optional[int], total: int):
    cells = _target_cells(state, mine, current, target)
    if current == Action.PURGE:
        return ()
    if current == Action.WHISPER:
        return Forge.geometry.split(int(total), cells) if len(cells) > 1 else tuple(((Forge.key(cell), int(total)) for cell in cells if Forge.key(cell)))
    if current == Action.DEFECT:
        victimkey = Forge.defecttargetkey(state, target) if hasattr(Forge, 'defecttargetkey') else _pubkey_at(state, int(target))
        legs = list(Forge.geometry.split(int(total), cells))
        if victimkey:
            legs.append((victimkey, 0))
        return tuple(legs)
    return Forge.geometry.split(int(total), cells)

def _intent_lock(state: Any, mine: int):
    cells = list(getattr(state, 'cells', []) or [])
    if 0 <= int(mine) < len(cells):
        out = getattr(cells[int(mine)], 'lock', None)
        try:
            if out is not None:
                pass
        except Exception:
            pass
        return out
    return None

def submit(cache: UiCache, state: Any, current: Action) -> None:
    mine = selfQorzero(state)
    total = int(getattr(cache, 'salt', 0) or 0)
    text = str(getattr(cache, 'text', '') or '')
    _rebind_focus(cache, state)
    target = getattr(cache, 'targetQ', None)
    try:
        intent = cache.syncIntent()
        intent.kind = str(getattr(current, 'value', current) or '').upper()
        intent.pairs = _intent_pairs(state, mine, current, target, total)
        intent.text = Forge.msgNorm(text)
        intent.lock = _intent_lock(state, mine)
        if current == Action.PURGE:
            intent.key = Forge.purgeTarget(state, target)
            intent.text = Forge.purgeFlavor() if hasattr(Forge, 'purgeFlavor') else ''
            intent.pairs = ()
            intent.lock = None
        send(intent)
    except Exception as exc:
        say(cache, 'ASH', str(exc) or 'intent failed')
    finally:
        reset(cache)

def handlemenu(cache: UiCache, state: Any, kind: str, value: Optional[str]):
    current = action(cache)
    if kind == 'ARROW':
        if value == 'C':
            menu(cache, +1)
        elif value == 'D':
            menu(cache, -1)
        return (cache, state, False)
    if kind != 'ENTER':
        return (cache, state, False)
    if current == Action.EXIT:
        return (cache, state, True)
    if current == Action.PURGE:
        cache.focus = Focus.TABLE_MOVE
        mine = selfQorzero(state) % geometry.cells
        start = (mine + 1) % geometry.cells
        cache.stateQ = start
        cache.stateKey = _pubkey_at(state, start)
        cache.targetQ = None
        cache.targetKey = ''
        cache.text = ''
        cache.syncIntent()
        return (cache, state, False)
    if current == Action.MONUMENT:
        cache.show_banner = not bool(getattr(cache, 'show_banner', True))
        cache.syncIntent()
        return (cache, state, False)
    if current == Action.LORE:
        showing = bool(getattr(cache, 'show_lore', False))
        cache.show_lore = not showing
        if cache.show_lore:
            cache.lore_offset = 0
        cache.syncIntent()
        return (cache, state, False)
    if current in (Action.RALLY, Action.WRATH):
        cache.focus = Focus.TABLE_LOCK
        cache.salt = floor(current, state)
        cache.text = ''
        cache.stateQ = selfQorzero(state) % geometry.cells
        cache.stateKey = _pubkey_at(state, cache.stateQ)
        cache.targetQ = None
        cache.targetKey = ''
        cache.syncIntent()
        return (cache, state, False)
    if current == Action.WHISPER:
        cache.focus = Focus.TABLE_MOVE
        cache.salt = 1
        cache.text = ''
        cache.stateQ = (selfQorzero(state) + 1) % geometry.cells
        cache.stateKey = _pubkey_at(state, cache.stateQ)
        cache.syncIntent()
        return (cache, state, False)
    if current == Action.DEFECT:
        cache.focus = Focus.TABLE_MOVE
        cache.salt = floor(current, state)
        mine = selfQorzero(state) % geometry.cells
        start = int(getattr(cache, 'stateQ', mine) or mine) % geometry.cells
        if start == mine:
            start = (mine + 1) % geometry.cells
        target = _first_defect(state, mine, start)
        cache.stateQ = mine if target is None else target
        cache.stateKey = _pubkey_at(state, cache.stateQ)
        cache.syncIntent()
        return (cache, state, False)
    return (cache, state, False)

def handletablemove(cache: UiCache, state: Any, kind: str, value: Optional[str]):
    current = action(cache)
    if kind == 'ARROW' and value in ('A', 'B', 'C', 'D'):
        moveboard(cache, value, current, state)
        return (cache, state, False)
    if kind != 'ENTER':
        return (cache, state, False)
    if current == Action.WHISPER:
        mine = selfQorzero(state) % geometry.cells
        if int(cache.stateQ) == mine:
            cache.stateQ = (mine + 1) % geometry.cells
        _bind_target_key(cache, state, int(cache.stateQ))
        cache.focus = Focus.SPINE
        cache.syncIntent()
        return (cache, state, False)
    if current == Action.PURGE:
        mine = selfQorzero(state) % geometry.cells
        target = int(cache.stateQ) % geometry.cells
        if mine == target or (hasattr(Forge, 'purgeViable') and (not Forge.purgeViable(state, mine, target))):
            say(cache, 'ASH', 'invalid target')
            reset(cache)
            return (cache, state, False)
        _bind_target_key(cache, state, target)
        submit(cache, state, current)
        return (cache, state, False)
    if current == Action.DEFECT:
        mine = selfQorzero(state) % geometry.cells
        target = int(cache.stateQ) % geometry.cells
        if not _defect_viable(state, mine, target):
            say(cache, 'ASH', 'invalid target')
            reset(cache)
            return (cache, state, False)
        _bind_target_key(cache, state, target)
        cache.focus = Focus.SPINE
        cache.text = ''
        cache.syncIntent()
        return (cache, state, False)
    return (cache, state, False)

def handletablelock(cache: UiCache, state: Any, kind: str, value: Optional[str]):
    if kind == 'ENTER':
        cache.focus = Focus.SPINE
        cache.text = ''
        cache.syncIntent()
    return (cache, state, False)

def handlespine(cache: UiCache, state: Any, kind: str, value: Optional[str]):
    current = action(cache)
    if kind == 'ARROW':
        if current == Action.DEFECT:
            return (cache, state, False)
        step = floor(current, state)
        cells = list(getattr(state, 'cells', []) or [])
        mine = selfQorzero(state) % geometry.cells
        have = Forge.amount(cells[mine]) if 0 <= mine < len(cells) else 0
        cap = have if have >= step else step
        if value == 'A':
            cache.salt = min(cap, int(cache.salt) + step)
        elif value == 'B':
            cache.salt = max(step, int(cache.salt) - step)
        cache.syncIntent()
        return (cache, state, False)
    if kind == 'BS':
        if current in (Action.WHISPER, Action.RALLY, Action.WRATH, Action.DEFECT):
            cache.text = (cache.text or '')[:-1]
            cache.syncIntent()
        return (cache, state, False)
    if kind == 'CH' and value and value.isprintable() and (current in (Action.WHISPER, Action.RALLY, Action.WRATH, Action.DEFECT)):
        if len(cache.text) < Forge.MSG_MAX:
            cache.text += value
            cache.syncIntent()
        return (cache, state, False)
    if kind != 'ENTER':
        return (cache, state, False)
    if current in (Action.WHISPER, Action.RALLY, Action.WRATH, Action.DEFECT):
        submit(cache, state, current)
    else:
        reset(cache)
    return (cache, state, False)

def dispatch(cache: UiCache, state: Any, token: Tuple[str, Optional[str]]):
    kind, value = token
    if kind == 'CTRL_C':
        return (cache, state, True)
    if bool(getattr(cache, 'show_lore', False)):
        if kind == 'ARROW':
            offset = int(getattr(cache, 'lore_offset', 0) or 0)
            if value == 'A':
                cache.lore_offset = max(0, offset - 1)
            elif value == 'B':
                cache.lore_offset = offset + 1
            cache.syncIntent()
            return (cache, state, False)
        if kind == 'ENTER' or (kind == 'CH' and value == ' '):
            cache.show_lore = False
            cache.lore_offset = 0
            cache.syncIntent()
        return (cache, state, False)
    if kind == 'CH' and value == ' ' and (cache.focus in (Focus.TABLE_MOVE, Focus.TABLE_LOCK)):
        reset(cache)
        return (cache, state, False)
    if cache.focus == Focus.MENU:
        return handlemenu(cache, state, kind, value)
    if cache.focus == Focus.TABLE_MOVE:
        return handletablemove(cache, state, kind, value)
    if cache.focus == Focus.TABLE_LOCK:
        return handletablelock(cache, state, kind, value)
    if cache.focus == Focus.SPINE:
        return handlespine(cache, state, kind, value)
    return (cache, state, False)

def parsekeys(buffer: str):
    out = []
    i = 0
    while i < len(buffer):
        c = buffer[i]
        if c == '\x03':
            out.append(('CTRL_C', None))
            i += 1
            continue
        if c == '\n':
            out.append(('ENTER', None))
            i += 1
            continue
        if c in ('\x7f', '\x08'):
            out.append(('BS', None))
            i += 1
            continue
        if c != '\x1b':
            out.append(('CH', c))
            i += 1
            continue
        if i + 1 >= len(buffer):
            break
        n1 = buffer[i + 1]
        if n1 == 'O' and i + 2 < len(buffer) and (buffer[i + 2] in ('A', 'B', 'C', 'D')):
            out.append(('ARROW', buffer[i + 2]))
            i += 3
            continue
        if n1 == '[':
            j = i + 2
            seq = ''
            while j < len(buffer):
                d = buffer[j]
                if d in ('A', 'B', 'C', 'D'):
                    seq += d
                    if seq.startswith('1;2') or seq.startswith('2'):
                        out.append(('SHIFT_ARROW', d))
                    else:
                        out.append(('ARROW', d))
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

def initcache(state: Any):
    state = Forge.makeState(state)
    cache = UiCache(feed=[], local_name='')
    cache.focus = Focus.TITLE
    cache.state = state
    cache.pending_request = 'WAITING_STATE'
    if citadel.ash is None:
        citadel.ash = []
    cache.ash = list(citadel.ash)
    cache.feed = list(citadel.ash)
    cache.visible_feed_count = len(cache.feed)
    cache.lore_offset = 0
    cache.intent.Q.self = Forge.Qof(state)
    citadel.bindcache(cache)
    citadel.state = state
    return cache

def installdebug():
    return None

def bindcache(cache: Any) -> Any:
    return citadel.bindcache(cache)

def bindcore(coreobj: Any) -> Any:
    global core
    core = coreobj
    return coreobj

def state(value: Any=None) -> Any:
    if value is None:
        return citadel.state
    citadel.surfacestate(value)
    return citadel.state

def intent(value: Any) -> Any:
    return citadel.intent(value)

def ashfall(value: Any) -> Any:
    return citadel.ashfall(value)

def peerdisplaylabel(key: str) -> str:
    return citadel.peerdisplay(key)
_parse_keys_buffered = parsekeys
_dispatch_token = dispatch
_init_cache = initcache
_install_debug = installdebug
peer_display_label = peerdisplaylabel
__all__ = ['Relay', 'citadel', 'bindcache', 'bindcore', 'state', 'intent', 'ashfall', 'peerdisplaylabel', 'parsekeys', 'dispatch', 'initcache', 'installdebug', '_parse_keys_buffered', '_dispatch_token', '_init_cache', '_install_debug', 'peer_display_label']
