from __future__ import annotations
import textwrap
import time
from typing import Dict, List, Optional
import Forge
from Forge import Action, Focus, UiCache, MENU, TERM_W, INNER_W, NAME_W, BOARD_COLS, BOARD_ROWS, vislen, clipw, clipTerm, centerTerm, padw, makeState
COL_GAP = 3
BODY_FILL_LINES = 23
PACK_RACE_THRESH_DEFAULT = 25000
HLINE = '=' * TERM_W
LORE_VIEW_LINES = 18
RESET = '\x1b[0m'
ASH = '\x1b[90m'
WHITE = '\x1b[97m'
EMBER = '\x1b[38;5;130m'
FLICKER1 = '\x1b[38;5;208m'
FLICKER2 = '\x1b[38;5;214m'

def _append_paragraph(lines: List[str], text: str, *, width: int=INNER_W, color: Optional[str]=None):
    for block in text.split('\n'):
        wrapped = textwrap.wrap(block, width=width, replace_whitespace=False, drop_whitespace=False)
        if not wrapped:
            lines.append('')
            continue
        for ln in wrapped:
            lines.append(clipTerm((color or '') + ln + (RESET if color else '')))

def flickerPair(phase: int) -> tuple[str, str]:
    return (FLICKER1, FLICKER2) if phase % 2 else (FLICKER2, FLICKER1)

def palette(cache: UiCache) -> Dict[str, str]:
    raw_phase = getattr(cache, 'flame_phase', None)
    try:
        phase = int(raw_phase) if raw_phase is not None else 0
    except Exception:
        phase = 0
    phase += int(time.monotonic() * 8.0)
    flicker1, flicker2 = flickerPair(phase)
    return {'reset': RESET, 'ash': ASH, 'white': WHITE, 'ember': EMBER, 'flicker1': flicker1, 'flicker2': flicker2, 'salt': WHITE}

def _intent(cache: UiCache):
    intent = getattr(cache, 'intent', None)
    if intent is None:
        try:
            intent = cache.syncIntent()
        except Exception:
            intent = None
    return intent

def _focus(cache: UiCache) -> Focus:
    intent = _intent(cache)
    picked = getattr(intent, 'focus', None)
    if isinstance(picked, Focus):
        return picked
    raw = getattr(cache, 'focus', None)
    return raw if isinstance(raw, Focus) else Focus.MENU

def _action(cache: UiCache) -> Action:
    intent = _intent(cache)
    picked = getattr(intent, 'action', None)
    if isinstance(picked, Action):
        return picked
    return MENU[int(getattr(cache, 'menuQ', 0) or 0) % len(MENU)]

def _city(cache: UiCache) -> int:
    intent = _intent(cache)
    q = getattr(intent, 'Q', None)
    raw = getattr(q, 'city', None) if q is not None else None
    if raw is None:
        raw = getattr(cache, 'stateQ', 0)
    return int(raw or 0)

def _q_by_key(state: object, raw: object) -> Optional[int]:
    try:
        found = Forge.qByKey(state, raw)
    except Exception:
        found = None
    return None if found is None else int(found)

def _target(cache: UiCache, state: object) -> Optional[int]:
    intent = _intent(cache)
    q = getattr(intent, 'Q', None)
    raw = getattr(q, 'target', None) if q is not None else None
    if raw is None:
        raw = getattr(cache, 'targetQ', None)
    key = str(getattr(cache, 'targetKey', '') or '').strip()
    if key:
        found = _q_by_key(state, key)
        if found is not None:
            return found
    return None if raw is None else int(raw)

def _amount(cache: UiCache) -> int:
    intent = _intent(cache)
    raw = getattr(intent, 'amount', None)
    if raw is None:
        raw = getattr(cache, 'salt', 1)
    return int(raw or 0)

def _text(cache: UiCache) -> str:
    intent = _intent(cache)
    raw = getattr(intent, 'text', None)
    if raw is None:
        raw = getattr(cache, 'text', '')
    return Forge.cleanDraft(raw)

def _me(state: object) -> int:
    try:
        found = Forge.Qof(state)
    except Exception:
        found = None
    return -1 if found is None else int(found)

def _label(cell: object, *, me_label: str='', is_me: bool=False) -> str:
    soul = str(getattr(cell, 'soul', '') or '').strip()
    key = str(getattr(cell, 'key', '') or '').strip()
    if soul:
        return soul
    if is_me and me_label:
        return me_label
    if key:
        short = ''
        try:
            short = Forge.id6(key)
        except Exception:
            short = key[:NAME_W]
        return short or key[:6].upper()
    return ''

def _selected_name(cache: UiCache, state: object) -> str:
    cells = list(getattr(state, 'cells', []) or [])
    me = _me(state)
    city = _city(cache)
    if me >= 0 and city == me:
        return str(getattr(cache, 'local_name', '') or '')
    if 0 <= city < len(cells):
        soul = str(getattr(cells[city], 'soul', '') or '').strip()
        if soul:
            return soul
        name = str(getattr(cells[city], 'name', '') or '').strip()
        if name:
            return name
    return ''

def _target_name(cache: UiCache, state: object) -> str:
    return str(_selected_name(cache, state) or '')

def _defect_viable(state: object, me: int, target: int) -> bool:
    if target == me:
        return False
    cells = list(getattr(state, 'cells', []) or [])
    if not (0 <= me < len(cells) and 0 <= target < len(cells)):
        return False
    try:
        return Forge.amount(cells[target]) < Forge.amount(cells[me]) and target // BOARD_ROWS != me // BOARD_ROWS
    except Exception:
        return False

def _ctx(cache: UiCache, state: object) -> Dict[str, object]:
    action = _action(cache)
    focus = _focus(cache)
    me = _me(state)
    rank = None if me < 0 else me + 1
    floor = Forge.actionFloor(action, rank=rank, Q=None if me < 0 else me)
    label = Forge.actionSpineLabel(action, rank=rank, Q=None if me < 0 else me)
    return {'action': action, 'focus': focus, 'me': me, 'city': _city(cache), 'target': _target(cache, state), 'floor': floor, 'label': label}

def build_banner(pal: Dict[str, str], title: str='BYZANTIUM') -> List[str]:
    return [pal['flicker2'] + '+    ' + RESET, pal['flicker2'] + '•  •  · ' + pal['ash'] + ')══{≡≡≡≡≡≡≡≡>     ' + pal['flicker2'] + '+  ' + pal['white'] + title + RESET + pal['flicker1'] + '  +     ' + pal['ash'] + '<≡≡≡≡≡≡≡≡}══( ' + pal['flicker1'] + '·  •  •    ' + RESET, pal['flicker1'] + '+    ' + RESET]

def _frame_lines(lines: List[str]) -> str:
    return '\n'.join((' ' + padw(clipw(line, INNER_W), INNER_W) + ' ' for line in lines)) + RESET

def _frame_text_screen(lines: List[str]) -> str:
    while len(lines) < BODY_FILL_LINES:
        lines.append(clipTerm(''))
    lines.append(clipTerm(ASH + HLINE + RESET))
    return _frame_lines(lines)

def _text_screen(*lines: str, body: str='', body_color: Optional[str]=None) -> str:
    out = [clipTerm(line) for line in lines]
    if body:
        _append_paragraph(out, body, color=body_color)
        out.append('')
    return _frame_text_screen(out)

def _title_value(cache: UiCache, name: str, default: str='') -> str:
    value = getattr(cache, name, None)
    if value is None:
        if name == 'soul':
            value = getattr(cache, 'local_name', default)
        elif name == 'genesis':
            value = default or '1'
        elif name == 'gate':
            value = default or '9000'
        elif name == 'skeleton':
            value = default or 'Skeleton'
        elif name == 'secret':
            value = default or 'Password'
        else:
            value = default
    return str(value or '').replace('\n', ' ').replace('\r', ' ')

def _title_active_index(cache: UiCache) -> int:
    for key in ('title_idx', 'title_field_idx', 'title_step', 'soul_step'):
        raw = getattr(cache, key, None)
        if raw is not None:
            try:
                return max(0, min(5, int(raw)))
            except Exception:
                pass
    return 0

def render_title_screen(cache: UiCache) -> str:
    pal = palette(cache)
    active = _title_active_index(cache)
    waiting = bool(getattr(cache, 'waiting', False))
    fields = [('Choose Your Arena', 'mode', 'Campaign'), ('Which Gateway', 'gate', '9000'), ('Skeleton Key', 'skeleton', 'Skeleton'), ('Who Are You', 'soul', ''), ('Tell Me A Secret', 'secret', 'Password'), ('How Many Souls', 'genesis', '1')]
    label, key, default = fields[max(0, min(active, len(fields) - 1))]
    value = _title_value(cache, key, default)[:8]
    if waiting:
        frame = int(getattr(cache, 'waiting_frame', 0) or 0)
        phase = frame // 55 % 4
        dots = '.' * phase
        label = f'{dots}Collecting Souls{dots}' if dots else 'Collecting Souls'
        label = pal['ash'] + label + RESET
        body = ''
    else:
        label = pal['ash'] + label + RESET
        left = pal['flicker1'] + ':' + RESET
        right = pal['flicker2'] + ':' + RESET
        body = f'{left}{WHITE}{value}{RESET}{right}' if value else f'{left}{right}'
    lines: List[str] = [ASH + HLINE + RESET, centerTerm(''), centerTerm(''), centerTerm(ASH + '.' + RESET), centerTerm(ASH + '.' + RESET + pal['flicker1'] + '+' + RESET + ASH + '.' + RESET), centerTerm(ASH + '.   .   .   .' + RESET), centerTerm(ASH + pal['flicker1'] + '+' + RESET + ' BYZANTIUM ' + RESET + pal['flicker1'] + '+' + RESET), centerTerm(ASH + '·   · ·   · ·   ·' + RESET), centerTerm(ASH + '·' + RESET + pal['flicker2'] + '+' + RESET + ASH + '·' + RESET), centerTerm(ASH + '·' + RESET), centerTerm(''), centerTerm(''), centerTerm(label), centerTerm('')]
    if waiting:
        lines.extend([centerTerm(''), centerTerm(''), centerTerm('')])
    else:
        lines.extend([centerTerm(body), centerTerm(''), centerTerm('')])
    return _frame_text_screen(lines)

def lore_lines() -> List[str]:
    return ['   The state persists through equilibrium, not control. Every front is', '   connected, and every imbalance is felt across the whole. There is no', '   isolated failure here, only shifts in pressure that must be carried.', '', '   There is no return to a previous state. No correction, no reconciliation.', '   Only continuation under strain. What settles becomes structure. What', '   moves becomes the next imbalance.', '', '   This is not a fixed system. It is a shared dream held under tension,', '   a pattern that persists through adjustment. At times it feels stable', '   At times it feels like a shared nightmare. Both are the same.', '', '   Positions are not assigned. They are maintained. A general holds only', '   as long as he can bear what gathers beneath him. Captains decide long', '   before the ranks ever change whether he is worth his salt.', '', '   There are always those who reach beyond what they can hold. Pressure', '   builds where it should not, and weight is taken before it is earned.', '   This is not an exception. It is expected.', '', '   Equivocation is not rare. A man may speak with two tongues, and both', '   will be heard. The state does not decide which was true. It', '   absorbs the burden and maintains continuity.', '', '   A man is only as good as his word. In Byzantium, every word is a', '   promise and every promise is honored, whether intended or not.', '', '   All debt must settle,', '   Consensus was never trust.'] + [''] * 3

def render_lore_screen(cache: UiCache) -> str:
    pal = palette(cache)
    lore = lore_lines()
    visible = LORE_VIEW_LINES
    total = len(lore)
    max_offset = max(0, total - visible)
    raw = getattr(cache, 'lore_offset', 0)
    try:
        offset = int(raw or 0)
    except Exception:
        offset = 0
    offset = max(0, min(offset, max_offset))
    try:
        cache.lore_offset = offset
    except Exception:
        pass
    window = lore[offset:offset + visible]
    lines: List[str] = [ASH + HLINE + RESET, centerTerm(ASH + '' + RESET), centerTerm(pal['flicker1'] + '·  •  •  ' + WHITE + Action.LORE.value + RESET + pal['flicker2'] + '  •  •  ·'), centerTerm(ASH + '' + RESET)]
    for ln in window:
        lines.append(clipTerm(ASH + ln + RESET if ln else ''))
    lines.append(clipTerm(''))
    return _frame_text_screen(lines)

def render_menu(cache: UiCache, pal: Dict[str, str]) -> List[str]:
    leader = (getattr(cache, 'local_name', '') or '').ljust(NAME_W)[:NAME_W]
    chunks = [(pal['flicker2'] if i == cache.menuQ else pal['ash']) + act.value + RESET for i, act in enumerate(MENU)]
    return [clipTerm(pal['ash'] + HLINE + RESET), clipTerm(f"{pal['flicker1']}{leader}{RESET} {pal['flicker1']}>>>{RESET} " + ' • '.join(chunks)), clipTerm(pal['ash'] + HLINE + RESET)]

def format_monument_line(line: object) -> str:
    raw = str(line or '')
    if not raw.strip():
        return ''
    head, score, body = Forge.parseMonument(raw, name=NAME_W)
    name = str(head or '').ljust(NAME_W)[:NAME_W]
    kind, tail = _ash_split_text(str(body or ''))
    kind = str(kind or '').strip().upper()
    if kind == 'DEFECT':
        score_field = 'Defected'.rjust(Forge.COST_W)[:Forge.COST_W]
    elif score is None:
        score_field = ''.rjust(Forge.COST_W)
    else:
        try:
            score_field = Forge.fmtSpineCost(int(str(score).replace(',', '')), width=Forge.COST_W, signed=True)
        except Exception:
            score_field = str(score).rjust(Forge.COST_W)[:Forge.COST_W]
    return f'{name} {score_field}:{tail}' if score is not None or tail else ''

def render_banner(cache: UiCache, pal: Dict[str, str], monuments: List[str], debug_lines: Optional[List[str]]) -> List[str]:
    if debug_lines is not None:
        payload = [pal['ash'] + ln + RESET for ln in debug_lines]
    elif getattr(cache, 'show_banner', True):
        payload = [centerTerm(line) for line in build_banner(pal, title='BYZANTIUM')]
    else:
        cleaned = [str(m).strip() for m in monuments if str(m).strip()]
        if cleaned:
            payload = [pal['ash'] + format_monument_line(line) + RESET for line in (cleaned + ['', '', ''])[:3]]
        else:
            payload = ['', (lambda line: line[2:])(centerTerm(pal['ash'] + 'Tabula Rasa' + RESET)), '']
    return [clipTerm(line) for line in payload]

def render_board(cache: UiCache, state: object, pal: Dict[str, str]) -> List[str]:
    cells = list(getattr(state, 'cells', []) or [])
    ctx = _ctx(cache, state)
    action = ctx['action']
    focus = ctx['focus']
    me = int(ctx['me'])
    city = int(ctx['city'])
    target = ctx['target']
    waiting_room = False

    def fmt_cell(idx: int) -> str:
        c = cells[idx]
        if waiting_room:
            return f"{pal['ash']}{''.ljust(NAME_W)}{RESET} {''.rjust(7)}"
        base_name = _label(c, me_label=str(getattr(cache, 'local_name', '') or ''), is_me=me >= 0 and idx == me)
        name = str(base_name).ljust(NAME_W)[:NAME_W]
        raw = f'{int(Forge.amount(c)):,}'
        purge = getattr(c, 'purge', None)
        lockbit = int(getattr(purge, 'lockbit', 0) or 0)
        chainbit = int(getattr(purge, 'chainbit', 0) or 0)
        if lockbit == 0:
            salt = raw.rjust(8)
        else:
            plus_col = pal['white'] if chainbit == 1 else pal['ash']
            num = ('+' + raw).rjust(8)
            plus_at = num.find('+')
            salt = num[:plus_at] + plus_col + '+' + RESET + num[plus_at + 1:]
        if action == Action.WRATH and focus in (Focus.TABLE_LOCK, Focus.SPINE):
            name_col = pal['flicker1']
        elif focus == Focus.TABLE_MOVE and idx == city or (action in (Action.WHISPER, Action.PURGE) and focus == Focus.SPINE and (idx == (target if target is not None else -1))):
            name_col = pal['flicker1']
        elif action == Action.RALLY and focus in (Focus.TABLE_LOCK, Focus.SPINE):
            name_col = pal['flicker1'] if me >= 0 and idx // BOARD_ROWS == me // BOARD_ROWS else pal['ash']
        elif action == Action.DEFECT and focus == Focus.TABLE_MOVE:
            name_col = pal['flicker1'] if idx == me else pal['ember'] if me >= 0 and _defect_viable(state, me, idx) else pal['ash']
        elif action == Action.DEFECT and focus == Focus.SPINE:
            if idx == me:
                name_col = pal['flicker1']
            elif idx == (target if target is not None else -1):
                name_col = pal['flicker1']
            else:
                name_col = pal['ash']
        elif me >= 0 and idx == me:
            name_col = pal['flicker1'] if idx % BOARD_ROWS == 0 else pal.get('salt', '')
        elif idx % BOARD_ROWS == 0:
            name_col = pal['ember']
        else:
            name_col = pal['ash']
        return f'{name_col}{name}{RESET} {salt}'
    out = [clipTerm(pal['ash'] + HLINE + RESET)]
    for r in range(BOARD_ROWS):
        parts = [fmt_cell(col * BOARD_ROWS + r) for col in range(BOARD_COLS) if col * BOARD_ROWS + r < len(cells)]
        out.append(clipTerm((' ' * COL_GAP).join(parts)))
    out.append(clipTerm(HLINE))
    return out

def _arm_or_compose_line(label: str, cost: int, draft: str, *, armed: bool, pal: Dict[str, str]) -> str:
    ash = pal['ash']
    flame = pal['flicker1']
    reset = pal['reset']
    name_fixed = str(label or '').ljust(NAME_W)[:NAME_W]
    cost_s = Forge.fmtSpineCost(cost)
    if armed:
        return f'{ash}{name_fixed}{reset} {ash}{cost_s}{reset}{ash}:{reset}'
    return f'{flame}{name_fixed}{reset} {cost_s}{ash}:{reset}{draft}'

def _default_desc_line(action: Action, anchor_col: int, *, state: object) -> str:
    desc = Forge.actionDesc(action)
    if not desc:
        return Forge.actionPreview(action)
    rank = Forge.resolveRank(state)
    floor = Forge.actionFloor(action, rank=rank)
    name_fixed = action.value.ljust(NAME_W)[:NAME_W]
    cost_raw = f'{floor:+,}'
    cost_field_w = max(len(cost_raw), max(0, anchor_col - (NAME_W + 1)))
    return f'{name_fixed} {cost_raw.rjust(cost_field_w)}:{desc}'

def build_spine_lines(cache: UiCache, state: object, *, pal: Optional[Dict[str, str]]=None, anchor_col: int=0) -> List[str]:
    if pal is None:
        pal = palette(cache)
    ash = pal['ash']
    white = pal['white']
    reset = pal['reset']
    ctx = _ctx(cache, state)
    action = ctx['action']
    focus = ctx['focus']
    label = ctx['label']
    floor = int(ctx['floor'])
    if focus == Focus.MENU:
        return [ash + centerTerm(Forge.actionPreview(action)) + reset, white + HLINE + reset]
    if action == Action.WHISPER and focus in (Focus.TABLE_MOVE, Focus.SPINE):
        line = _arm_or_compose_line(_selected_name(cache, state) if focus == Focus.TABLE_MOVE else _target_name(cache, state), _amount(cache), _text(cache), armed=focus == Focus.TABLE_MOVE, pal=pal)
        return [line, white + HLINE + reset]
    if action in (Action.WRATH, Action.RALLY) and focus in (Focus.TABLE_LOCK, Focus.SPINE):
        line = _arm_or_compose_line(str(label or Forge.actionSpineLabel(action, rank=Forge.resolveRank(state)) or action.value), _amount(cache), _text(cache), armed=focus == Focus.TABLE_LOCK, pal=pal)
        return [line, white + HLINE + reset]
    if action == Action.DEFECT and focus in (Focus.TABLE_MOVE, Focus.SPINE):
        line = _arm_or_compose_line('DEFECT', floor, _text(cache), armed=focus == Focus.TABLE_MOVE, pal=pal)
        return [line, white + HLINE + reset]
    if action == Action.PURGE and focus == Focus.TABLE_MOVE:
        name = _selected_name(cache, state) or 'PURGE'
        line = ash + centerTerm(str(name).strip() or 'PURGE') + reset
        return [line, white + HLINE + reset]
    return [ash + _default_desc_line(action, anchor_col, state=state) + reset, white + HLINE + reset]

def _ash_amount_total(raw: str) -> int:
    digits = ''.join((ch for ch in str(raw or '') if ch.isdigit()))
    try:
        return int(digits or '0')
    except Exception:
        return 0

def _ash_split_text(raw: str) -> tuple[str, str]:
    text = str(raw or '')
    head, sep, tail = text.partition('|')
    kind = head.strip().upper() if sep else ''
    return kind, tail if sep else text

def _ash_payload_color(pal: Dict[str, str], mine: bool) -> str:
    return pal['ash'] if mine else pal['white']

def _ash_is_broadcast(kind: str) -> bool:
    return kind in ('WRATH', 'DEFECT')

def _ash_tag_color(pal: Dict[str, str], mine: bool, kind: str, amount_total: int=0) -> str:
    if mine:
        return pal['ash']
    if kind == 'WRATH':
        return pal['flicker1']
    if kind == 'RALLY':
        return pal['ember']
    if kind == 'DEFECT':
        return pal['flicker1'] if amount_total in (2000, 10000) else pal['ember']
    return pal['white']

def _ash_tag_text(raw: str, kind: str) -> str:
    return 'Defected' if kind == 'DEFECT' else raw

def _render_ash_entry(cache: UiCache, pal: Dict[str, str], chan: str, line: object) -> str:
    s = str(line or '')
    sender = s[:NAME_W].strip()
    mine = str(getattr(cache, 'local_name', '') or '').strip()
    is_mine = bool(sender) and sender == mine
    cut = min(len(s), NAME_W + 1)
    colon = s.find(':', cut)
    if colon < 0:
        colon = len(s)
    name = s[:NAME_W]
    gap = s[NAME_W:cut]
    amount = s[cut:colon]
    rawtext = s[colon + 1:] if colon < len(s) else ''
    kind_text, text = _ash_split_text(rawtext)
    kind_chan = str(chan or '').strip().upper()
    kind = kind_text or kind_chan
    payload = _ash_payload_color(pal, is_mine)
    amount_total = _ash_amount_total(amount)
    tag = _ash_tag_color(pal, is_mine, kind, amount_total)
    ash_tag = _ash_tag_text(amount, kind)
    if _ash_is_broadcast(kind) and not is_mine:
        payload = pal['white']
    return clipTerm(pal['ash'] + name + RESET + gap + tag + ash_tag + RESET + pal['ash'] + ':' + RESET + payload + text + RESET)



def _render_ash_placeholder(pal: Dict[str, str]) -> str:
    return clipTerm(centerTerm(pal['ash'] + 'all that remains is ash...' + RESET))

def render_ash(cache: UiCache, state: object, pal: Dict[str, str], anchor_col: int) -> List[str]:
    lines = list(build_spine_lines(cache, state, pal=pal, anchor_col=anchor_col))
    ash = list(getattr(cache, 'ash', []) or [])
    if ash:
        feed = ash
        vis = len(feed)
    else:
        feed = list(getattr(cache, 'feed', []) or [])
        vis = min(int(getattr(cache, 'visible_feed_count', 0)), len(feed))

    start = max(0, len(feed) - vis)
    rendered = [_render_ash_entry(cache, pal, chan, line) for chan, line in reversed(feed[start:])]
    max_rows = 7

    if len(rendered) <= 3:
        placeholder_row = 3 + len(rendered)
        for row in range(max_rows):
            if row < len(rendered):
                lines.append(rendered[row])
            elif row == placeholder_row:
                lines.append(_render_ash_placeholder(pal))
            else:
                lines.append(clipTerm(''))
        return lines

    for row in range(max_rows):
        lines.append(rendered[row] if row < len(rendered) else clipTerm(''))
    return lines

def _team_totals_bottom_bar(state: object, *, width: int, pal: Dict[str, str], thresh: int=PACK_RACE_THRESH_DEFAULT) -> str:
    totals = {tid: 0 for tid in range(1, BOARD_COLS + 1)}
    for i, c in enumerate(list(getattr(state, 'cells', []) or [])):
        tid = i // BOARD_ROWS + 1
        if tid in totals:
            totals[tid] += int(Forge.amount(c))
    pairs = sorted(((totals[i], i) for i in totals))
    hot_ember: set[int] = set()
    lo = best_lo = 0
    best_hi = -1
    for j in range(1, len(pairs)):
        if pairs[j][0] - pairs[j - 1][0] > thresh:
            if j - lo > best_hi - best_lo + 1:
                best_lo, best_hi = (lo, j - 1)
            lo = j
    if len(pairs) - lo > best_hi - best_lo + 1:
        best_lo, best_hi = (lo, len(pairs) - 1)
    if best_hi - best_lo + 1 >= 3:
        hot_ember.update((pairs[k][1] for k in range(best_lo, best_hi + 1)))

    def fmt(team_id: int, val: int) -> str:
        s = f'{val:,}'
        if val == 0:
            return f"{pal['ash']}{s}{RESET}{pal['ash']}"
        tied = sum((1 for v, _ in pairs if v == val)) >= 2
        if tied:
            return f"{pal['flicker1']}{s}{RESET}{pal['ash']}"
        if team_id in hot_ember:
            return f"{pal['ember']}{s}{RESET}{pal['ash']}"
        return f"{pal['ash']}{s}{RESET}{pal['ash']}"
    center = f" {RESET}•{pal['ash']} ".join((fmt(tid, totals[tid]) for tid in range(1, BOARD_COLS + 1)))
    pad_total = width - vislen(center) - 2
    if pad_total < 2:
        center = center[:max(0, width - 4)]
        pad_total = width - vislen(center) - 2
    left = max(0, pad_total // 2 - 2)
    right = pad_total - left
    return '=' * left + ' ' + center + ' ' + '=' * right

def render_totals(cache: UiCache, state: object, pal: Dict[str, str]) -> List[str]:
    waiting_room = str(getattr(cache, 'pending_request', '') or '').strip().upper().startswith('WAIT')
    bar = '=' * TERM_W if waiting_room else _team_totals_bottom_bar(state, width=TERM_W, pal=pal)
    return [clipTerm(pal['ash'] + bar + RESET)]

def render_screen(cache: UiCache, state: object) -> str:
    state = makeState(state)
    if not hasattr(cache, 'ash'):
        try:
            cache.ash = []
        except Exception:
            pass
    focus = _focus(cache)
    cells = list(getattr(state, 'cells', []) or [])
    if focus == Focus.TITLE or not cells:
        return render_title_screen(cache)
    if getattr(cache, 'show_lore', False):
        return render_lore_screen(cache)
    raw_mons = getattr(cache, 'monuments', None) or []
    monuments = [str(m[1]) if isinstance(m, tuple) and len(m) >= 2 else str(m) for m in raw_mons] if isinstance(raw_mons, list) else []
    pal = palette(cache)
    anchor_col = Forge.monumentAnchorCol(monuments, getattr(cache, 'local_name', '') or '')
    lines: List[str] = []
    lines.extend(render_menu(cache, pal))
    lines.extend(render_banner(cache, pal, monuments, None))
    lines.extend(render_board(cache, state, pal))
    lines.extend(render_ash(cache, state, pal, anchor_col))
    while len(lines) < BODY_FILL_LINES:
        lines.append(clipTerm(''))
    lines.extend(render_totals(cache, state, pal))
    return _frame_lines(lines)

def render(cache: UiCache, state: object, ctx: Optional[object]=None) -> str:
    return render_screen(cache, state)
__all__ = ['palette', 'flickerPair', 'build_banner', 'build_spine_lines', 'render_title_screen', 'lore_lines', 'render_lore_screen', 'render_menu', 'render_banner', 'render_board', 'render_ash', 'render_totals', 'render_debug', 'render_screen', 'render']
