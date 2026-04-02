from __future__ import annotations
import os
import sys
import termios
import time
import tty
from dataclasses import dataclass
from select import select
from typing import Dict, List, Optional
import Citadel
import Forge
import Spire
import Vault
from Forge import Focus, UiCache
FPS: float = 55.0
WAITING_DOT_FRAMES: int = int(FPS)
DEFAULT_MODE = 'Campaign'
DEFAULT_SKELETON = 'Skeleton'
DEFAULT_SECRET = 'Password'
DEFAULT_GATE = '9000'
DEFAULT_GENESIS = '1'
DEFAULT_SOUL = 'Satoshi'
MAX_FIELD_LEN = 8
MODE_OPTIONS: List[str] = ['Campaign', 'Siege', 'Exit']
TITLE_FIELDS: List[str] = ['mode', 'gate', 'skeleton', 'soul', 'secret', 'genesis']
FIELD_DEFAULTS: Dict[str, str] = {'mode': DEFAULT_MODE, 'gate': DEFAULT_GATE, 'skeleton': DEFAULT_SKELETON, 'soul': '', 'secret': DEFAULT_SECRET, 'genesis': DEFAULT_GENESIS}
FIELD_LIMITS: Dict[str, int] = {'mode': 8, 'gate': 6, 'skeleton': MAX_FIELD_LEN, 'soul': MAX_FIELD_LEN, 'secret': MAX_FIELD_LEN, 'genesis': 2}
EDITABLE_FIELDS = {'gate', 'skeleton', 'soul', 'secret', 'genesis'}
STEPPED_FIELDS = {'mode', 'gate', 'genesis'}

@dataclass
class State:
    mode: str
    gate: str
    skeleton: str
    soul: str
    secret: str
    genesis: int

@dataclass
class PlaceholderState:
    cells: tuple = ()
    reserve: int = 0

def empty_state() -> PlaceholderState:
    return PlaceholderState()

class Core:

    def __init__(self) -> None:
        self.spire = Spire
        self.citadel = getattr(Citadel, '_CITADEL', Citadel)
        self.vault = None

    def intent(self, value):
        if self.vault is None:
            return None
        return self.vault.glyph(value)

class Gateway:

    def __init__(self) -> None:
        self.core = Core()
        self.state: Optional[State] = None
        self.listenport = parseports()
        self.runtime = empty_state()
        self.cache = Citadel._init_cache(self.runtime)
        _ensure_state(self.cache)
        self.cache.mode = DEFAULT_MODE
        self.cache.title_step = 0
        self.cache.title_idx = 0
        self.cache.cursor_pos = len(DEFAULT_MODE)
        if self.listenport is not None:
            self.cache.gate = str(self.listenport)
        Citadel.core = self.core

    def buildstate(self) -> State:
        self.state = _build_state(self.cache)
        return self.state

    def bootfromtitle(self) -> object:
        self.buildstate()
        self.cache.gatejam = False
        self.cache.win_screen = False
        self.cache.exit_screen = False
        self.cache.pending_request = None
        try:
            self.core.vault = Vault.Vault(state=self.state, citadel=self.core.citadel)
        except Exception:
            self.core.vault = None
            self.cache.waiting = False
            self.cache.waiting_frame = 0
            self.cache.gatejam = True
            self.cache.gate = str(getattr(self.state, 'gate', self.cache.gate) or self.cache.gate)
            return self.runtime
        Citadel.core = self.core
        self.cache.city_idx = 0
        self.cache.waiting = True
        self.cache.waiting_frame = 0
        self.cache.waiting_started = time.monotonic()
        return empty_state()

    def dispatch(self, state: object, tok):
        kind, val = tok
        if kind == 'CTRL_C':
            return (self.cache, state, True)
        if getattr(self.cache, 'exit_screen', False):
            return (self.cache, state, True)
        if getattr(self.cache, 'win_screen', False):
            if kind in ('ENTER', 'CTRL_C') or kind == 'CH':
                self.cache.win_screen = False
                self.cache.exit_screen = True
            return (self.cache, state, False)
        if getattr(self.cache, 'gatejam', False):
            self.cache.gatejam = False
            self.cache.exit_screen = True
            return (self.cache, state, False)
        if getattr(self.cache, 'waiting', False):
            if kind == 'CH' and val == ' ':
                self.cache.waiting = False
                self.cache.waiting_frame = 0
                self.cache.exit_screen = True
                return (self.cache, state, False)
            return (self.cache, state, False)
        if self.cache.focus == Focus.TITLE:
            return _handle_title_gateway(self, self.cache, state, kind, val)
        cache, state, should_quit = Citadel._dispatch_token(self.cache, state, tok)
        if should_quit:
            self.cache.exit_screen = True
            return (self.cache, state, False)
        return (cache, state, should_quit)

    def currentrenderstate(self, state: object) -> object:
        surfaced = getattr(self.cache, 'state', None)
        if getattr(self.cache, 'waiting', False):
            self.cache.waiting_frame = int(getattr(self.cache, 'waiting_frame', 0) or 0) + 1
            if _state_is_ready(surfaced):
                self.cache.waiting = False
                self.cache.focus = Focus.MENU
                if hasattr(self.cache, 'syncIntent'):
                    self.cache.syncIntent()
        if surfaced is None:
            return state
        if _is_victory(surfaced):
            self.cache.win_screen = True
        return surfaced

    def render(self, state: object) -> str:
        if getattr(self.cache, 'exit_screen', False):
            return renderexit(self.cache)
        if getattr(self.cache, 'win_screen', False):
            return renderwin(self.cache)
        if getattr(self.cache, 'gatejam', False):
            return rendergatejam(self.cache)
        if getattr(self.cache, 'waiting', False):
            return renderwaiting(self.cache)
        return Spire.render(self.cache, state)

def parseports() -> Optional[int]:
    listenport: Optional[int] = None
    try:
        if len(sys.argv) >= 2:
            listenport = int(sys.argv[1])
    except Exception:
        listenport = None
    return listenport

def _safe_int(text: object, fallback: int) -> int:
    try:
        s = str(text or '').strip()
        return int(s) if s else int(fallback)
    except Exception:
        return int(fallback)

def _ensure_state(cache: UiCache) -> None:
    if not hasattr(cache, 'title_step'):
        cache.title_step = 0
    if not hasattr(cache, 'title_idx'):
        cache.title_idx = int(getattr(cache, 'title_step', 0) or 0)
    if not hasattr(cache, 'cursor_pos'):
        cache.cursor_pos = 0
    if not hasattr(cache, 'waiting'):
        cache.waiting = False
    if not hasattr(cache, 'waiting_frame'):
        cache.waiting_frame = 0
    if not hasattr(cache, 'waiting_started'):
        cache.waiting_started = 0.0
    if not hasattr(cache, 'exit_screen'):
        cache.exit_screen = False
    if not hasattr(cache, 'gatejam'):
        cache.gatejam = False
    if not hasattr(cache, 'win_screen'):
        cache.win_screen = False
    for key, default in FIELD_DEFAULTS.items():
        if not hasattr(cache, key):
            setattr(cache, key, default)
    raw_mode = str(getattr(cache, 'mode', DEFAULT_MODE) or DEFAULT_MODE).strip()
    cache.mode = raw_mode if raw_mode in MODE_OPTIONS else DEFAULT_MODE
    cache.local_name = str(getattr(cache, 'soul', '') or '')
    raw_gate = str(getattr(cache, 'gate', FIELD_DEFAULTS['gate']) or '')
    cache.gate = ''.join((ch for ch in raw_gate if ch.isdigit()))[:FIELD_LIMITS['gate']]
    raw_genesis = str(getattr(cache, 'genesis', FIELD_DEFAULTS['genesis']) or '')
    cache.genesis = ''.join((ch for ch in raw_genesis if ch.isdigit()))[:FIELD_LIMITS['genesis']]
    step_raw = getattr(cache, 'title_idx', getattr(cache, 'title_step', 0))
    step = max(0, min(len(TITLE_FIELDS) - 1, int(step_raw or 0)))
    cache.title_step = step
    cache.title_idx = step
    field = TITLE_FIELDS[step]
    value = str(getattr(cache, field, FIELD_DEFAULTS[field]) or '')
    cache.cursor_pos = max(0, min(len(value), int(getattr(cache, 'cursor_pos', len(value)) or 0)))

def _active_field(cache: UiCache) -> str:
    _ensure_state(cache)
    return TITLE_FIELDS[int(getattr(cache, 'title_step', 0) or 0)]

def _field_value(cache: UiCache, field: str) -> str:
    _ensure_state(cache)
    return str(getattr(cache, field, FIELD_DEFAULTS[field]) or '')

def _set_field_value(cache: UiCache, field: str, value: str) -> None:
    limit = int(FIELD_LIMITS.get(field, MAX_FIELD_LEN))
    value = str(value or '')[:limit]
    if field in ('gate', 'genesis'):
        value = ''.join((ch for ch in value if ch.isdigit()))
    if field == 'mode':
        value = str(value or DEFAULT_MODE).strip()
        value = value if value in MODE_OPTIONS else DEFAULT_MODE
    setattr(cache, field, value)
    if field == 'soul':
        cache.local_name = value

def _move_field(cache: UiCache, delta: int) -> None:
    _ensure_state(cache)
    step = int(getattr(cache, 'title_idx', getattr(cache, 'title_step', 0)) or 0) + int(delta)
    step = max(0, min(len(TITLE_FIELDS) - 1, step))
    cache.title_step = step
    cache.title_idx = step
    field = TITLE_FIELDS[step]
    cache.cursor_pos = len(_field_value(cache, field))

def _edit_insert(cache: UiCache, ch: str) -> None:
    field = _active_field(cache)
    if field not in EDITABLE_FIELDS:
        return
    value = _field_value(cache, field)
    limit = int(FIELD_LIMITS.get(field, MAX_FIELD_LEN))
    if len(value) >= limit:
        return
    if field in ('gate', 'genesis') and (not ch.isdigit()):
        return
    pos = int(getattr(cache, 'cursor_pos', len(value)) or 0)
    pos = max(0, min(len(value), pos))
    new_value = value[:pos] + ch + value[pos:]
    _set_field_value(cache, field, new_value)
    cache.cursor_pos = min(len(_field_value(cache, field)), pos + 1)

def _edit_backspace(cache: UiCache) -> None:
    field = _active_field(cache)
    if field not in EDITABLE_FIELDS:
        return
    value = _field_value(cache, field)
    pos = int(getattr(cache, 'cursor_pos', len(value)) or 0)
    if pos <= 0 or not value:
        return
    new_value = value[:pos - 1] + value[pos:]
    _set_field_value(cache, field, new_value)
    cache.cursor_pos = max(0, pos - 1)

def _toggle_mode(cache: UiCache, delta: int) -> None:
    current = str(_field_value(cache, 'mode') or DEFAULT_MODE)
    try:
        idx = MODE_OPTIONS.index(current)
    except ValueError:
        idx = 0
    idx = (idx + int(delta)) % len(MODE_OPTIONS)
    _set_field_value(cache, 'mode', MODE_OPTIONS[idx])
    cache.cursor_pos = len(_field_value(cache, 'mode'))

def _build_state(cache: UiCache) -> State:
    _ensure_state(cache)
    mode = _field_value(cache, 'mode') or DEFAULT_MODE
    raw_gate = _field_value(cache, 'gate')
    gate_num = _safe_int(raw_gate, int(DEFAULT_GATE)) if raw_gate else int(DEFAULT_GATE)
    if gate_num < 1024 or gate_num > 65535:
        gate_num = int(DEFAULT_GATE)
    gate = str(gate_num)
    skeleton = _field_value(cache, 'skeleton') or DEFAULT_SKELETON
    soul = _field_value(cache, 'soul') or DEFAULT_SOUL
    secret = _field_value(cache, 'secret') or DEFAULT_SECRET
    raw_genesis = _field_value(cache, 'genesis')
    genesis = max(1, min(24, _safe_int(raw_genesis or DEFAULT_GENESIS, 1)))
    _set_field_value(cache, 'mode', mode)
    _set_field_value(cache, 'gate', gate)
    _set_field_value(cache, 'skeleton', skeleton)
    _set_field_value(cache, 'soul', soul)
    _set_field_value(cache, 'secret', secret)
    _set_field_value(cache, 'genesis', str(genesis))
    cache.local_name = soul
    return State(mode=mode, gate=gate, skeleton=skeleton, soul=soul, secret=secret, genesis=genesis)

def _state_is_ready(state: object) -> bool:
    cells = getattr(state, 'cells', None)
    if cells is None:
        return False
    try:
        return len(cells) > 0
    except Exception:
        return False

def _is_victory(state: object) -> bool:
    cells = list(getattr(state, 'cells', []) or [])
    if len(cells) < 24:
        return False
    totals = [0, 0, 0, 0]
    for i, cell in enumerate(cells[:24]):
        totals[i // 6] += int(Forge.amount(cell))
    return all((total == 250000 for total in totals))

def _screen_palette(cache: UiCache) -> Dict[str, str]:
    raw_phase = getattr(cache, 'flame_phase', None)
    try:
        phase = int(raw_phase) if raw_phase is not None else 0
    except Exception:
        phase = 0
    phase += int(time.monotonic() * 8.0)
    flame = Spire.FLICKER1 if phase % 2 else Spire.FLICKER2
    return {'ash': Spire.ASH, 'flame': flame, 'flare': flame, 'reset': Spire.RESET}

def renderframe(cache: UiCache, label: str, value: str='', subtitle: str='') -> str:
    pal = _screen_palette(cache)
    label = str(label or '')
    subtitle = str(subtitle or '')
    value = str(value or '')[:8]
    if value:
        left = pal['flame'] + ':' + pal['reset']
        right = pal['flame'] + ':' + pal['reset']
        body = f"{left}{Spire.WHITE}{value}{pal['reset']}{right}"
    else:
        body = ''
    lines: List[str] = [Spire.ASH + Spire.HLINE + Spire.RESET, Spire.centerTerm(''), Spire.centerTerm(''), Spire.centerTerm(Spire.ASH + '.' + Spire.RESET), Spire.centerTerm(Spire.ASH + '.' + Spire.RESET + pal['flame'] + '+' + pal['reset'] + Spire.ASH + '.' + Spire.RESET), Spire.centerTerm(Spire.ASH + '.   .   .   .' + Spire.RESET), Spire.centerTerm(Spire.ASH + pal['flame'] + '+' + pal['reset'] + ' BYZANTIUM ' + pal['reset'] + pal['flame'] + '+' + pal['reset']), Spire.centerTerm(Spire.ASH + '·   · ·   · ·   ·' + Spire.RESET), Spire.centerTerm(Spire.ASH + '·' + Spire.RESET + pal['flare'] + '+' + pal['reset'] + Spire.ASH + '·' + Spire.RESET), Spire.centerTerm(Spire.ASH + '·' + Spire.RESET), Spire.centerTerm(''), Spire.centerTerm(''), Spire.centerTerm(Spire.ASH + label + Spire.RESET)]
    if subtitle:
        lines.append(Spire.centerTerm(''))
        lines.append(Spire.centerTerm(Spire.ASH + subtitle + Spire.RESET))
    else:
        lines.append(Spire.centerTerm(''))
    if body:
        lines.extend([Spire.centerTerm(''), Spire.centerTerm(body), Spire.centerTerm('')])
    else:
        lines.extend([Spire.centerTerm(''), Spire.centerTerm(''), Spire.centerTerm('')])
    return Spire._frame_text_screen(lines)

def rendertitle(cache: UiCache) -> str:
    _ensure_state(cache)
    fields = [('Choose Your Arena', 'mode', 'Campaign'), ('Which Gateway', 'gate', '9000'), ('Skeleton Key', 'skeleton', 'Skeleton'), ('Who Are You', 'soul', ''), ('Tell Me A Secret', 'secret', 'Password'), ('How Many Souls', 'genesis', '1')]
    idx = max(0, min(len(fields) - 1, int(getattr(cache, 'title_idx', 0) or 0)))
    label, key, default = fields[idx]
    value = _field_value(cache, key) or default
    return renderframe(cache, label, value)

def renderwaiting(cache: UiCache) -> str:
    return renderframe(cache, _waiting_line(cache), '')

def rendergatejam(cache: UiCache) -> str:
    return renderframe(cache, 'Open...Sesame!!!', '', 'This Gate Is Jammed...')

def renderexit(cache: UiCache) -> str:
    return renderframe(cache, 'Uhh..Ok', '', 'Maybe Go Touch Some Grass')

def renderwin(cache: UiCache) -> str:
    return renderframe(cache, 'Oh WOW...You Did It?', '')

def _waiting_line(cache: UiCache) -> str:
    frame = int(getattr(cache, 'waiting_frame', 0) or 0)
    phase = frame // max(1, WAITING_DOT_FRAMES) % 4
    dots = '.' * phase
    return f'{dots}Collecting Souls{dots}' if dots else 'Collecting Souls'

def _drain_stdin(fd: int) -> List[str]:
    chunks: List[str] = []
    while True:
        r, _w, _e = select([fd], [], [], 0)
        if not r:
            break
        try:
            b = os.read(fd, 4096)
        except BlockingIOError:
            break
        if not b:
            break
        chunks.append(b.decode('latin1', 'ignore'))
    return chunks

def _handle_title_gateway(gateway: Gateway, cache: UiCache, state: object, kind: str, val: Optional[str]) -> tuple[UiCache, object, bool]:
    _ensure_state(cache)
    field = _active_field(cache)

    def _step_genesis(delta: int) -> None:
        raw = _field_value(cache, 'genesis')
        cur = _safe_int(raw, 1) if raw else 1
        _set_field_value(cache, 'genesis', str(max(1, min(24, cur + int(delta)))))
        cache.cursor_pos = len(_field_value(cache, 'genesis'))

    def _step_gate(delta: int) -> None:
        raw = _field_value(cache, 'gate')
        cur = _safe_int(raw, int(DEFAULT_GATE)) if raw else int(DEFAULT_GATE)
        nxt = cur + int(delta)
        if nxt < 0:
            nxt = 0
        if nxt > 65535:
            nxt = 65535
        _set_field_value(cache, 'gate', str(nxt))
        cache.cursor_pos = len(_field_value(cache, 'gate'))
    if kind == 'ENTER':
        if _active_field(cache) == 'mode' and _field_value(cache, 'mode') == 'Exit':
            cache.exit_screen = True
            return (cache, state, False)
        if _active_field(cache) == 'genesis':
            return (cache, gateway.bootfromtitle(), False)
        _move_field(cache, +1)
        return (cache, state, False)
    if kind in ('LEFT', 'RIGHT'):
        _move_field(cache, -1 if kind == 'LEFT' else +1)
        return (cache, state, False)
    if kind in ('UP', 'DOWN', 'ARROW', 'SHIFT_ARROW'):
        arrow = val
        if kind == 'UP':
            arrow = 'A'
        elif kind == 'DOWN':
            arrow = 'B'
        if kind in ('ARROW', 'SHIFT_ARROW') and arrow in ('C', 'D'):
            _move_field(cache, +1 if arrow == 'C' else -1)
            return (cache, state, False)
        delta = +1 if arrow == 'A' else -1 if arrow == 'B' else 0
        if field == 'mode' and delta:
            _toggle_mode(cache, -delta)
            return (cache, state, False)
        if field == 'genesis' and delta:
            _step_genesis(delta)
            return (cache, state, False)
        if field == 'gate' and delta:
            step = 100 if kind == 'SHIFT_ARROW' else 1
            _step_gate(step if delta > 0 else -step)
            return (cache, state, False)
    if kind == 'BS':
        _edit_backspace(cache)
        return (cache, state, False)
    if kind == 'CH' and val and val.isprintable() and (val != '\t'):
        _edit_insert(cache, val)
        return (cache, state, False)
    return (cache, state, False)

def main() -> None:
    gateway = Gateway()
    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    state = gateway.runtime
    buf = ''
    try:
        while True:
            chunks = _drain_stdin(fd)
            if chunks:
                buf += ''.join(chunks)
                toks, buf = Citadel._parse_keys_buffered(buf)
                for tok in toks:
                    gateway.cache, state, should_quit = gateway.dispatch(state, tok)
                    if should_quit:
                        sys.stdout.write('\x1b[H\x1b[2J\x1b[0m')
                        sys.stdout.flush()
                        return
            try:
                state = gateway.currentrenderstate(state)
            except Exception:
                pass
            try:
                frame = gateway.render(state)
            except Exception:
                frame = ''
            if frame:
                sys.stdout.write('\x1b[H\x1b[2J')
                sys.stdout.write(frame)
                sys.stdout.flush()
            time.sleep(1.0 / FPS)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
__all__ = ['State', 'PlaceholderState', 'Core', 'Gateway', 'empty_state', 'parseports', 'renderframe', 'rendertitle', 'renderwaiting', 'rendergatejam', 'renderexit', 'renderwin', 'main']
if __name__ == '__main__':
    main()
