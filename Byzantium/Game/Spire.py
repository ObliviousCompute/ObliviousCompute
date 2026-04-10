from __future__ import annotations

from typing import Dict, List, Optional

import Forge
from Forge import Action, Focus, Cache, Menu, NameWidth, Columns, Rows, VisLen, ClipTerm, CenterTerm, MakeState


Colonnade = 3
FinishLine = 25000
LoreCount = 18


def StateIntent(cache: Cache):
    intent = getattr(cache, 'intent', None)
    if intent is None:
        try:
            return cache.SyncIntent()
        except Exception:
            return None
    return intent


def StateFocus(cache: Cache) -> Focus:
    intent = StateIntent(cache)
    focus = getattr(intent, 'focus', None) if intent is not None else None
    if isinstance(focus, Focus):
        return focus
    raw = getattr(cache, 'focus', None)
    return raw if isinstance(raw, Focus) else Focus.Menu


def StateAction(cache: Cache) -> Action:
    intent = StateIntent(cache)
    action = getattr(intent, 'action', None) if intent is not None else None
    if isinstance(action, Action):
        return action
    return Menu[int(getattr(cache, 'menuq', 0) or 0) % len(Menu)]


def ActiveState(cache: Cache) -> int:
    intent = StateIntent(cache)
    q = getattr(intent, 'q', None) if intent is not None else None
    city = getattr(q, 'city', None) if q is not None else None
    if city is None:
        city = getattr(cache, 'stateq', 0)
    return int(city or 0)


def StateTarget(cache: Cache, state: object) -> Optional[int]:
    intent = StateIntent(cache)
    q = getattr(intent, 'q', None) if intent is not None else None
    target = getattr(q, 'target', None) if q is not None else None
    if target is None:
        target = getattr(cache, 'targetq', None)
    targetkey = str(getattr(cache, 'targetkey', '') or '').strip()
    if targetkey:
        found = Forge.QxKey(state, targetkey)
        if found is not None:
            return int(found)
    return None if target is None else int(target)


def StateAmount(cache: Cache) -> int:
    intent = StateIntent(cache)
    amount = getattr(intent, 'amount', None) if intent is not None else None
    if amount is None:
        amount = getattr(cache, 'salt', 1)
    return int(amount or 0)


def StateText(cache: Cache) -> str:
    intent = StateIntent(cache)
    text = getattr(intent, 'text', None) if intent is not None else None
    if text is None:
        text = getattr(cache, 'text', '')
    return Forge.CleanDraft(text)


def StateSelf(state: object) -> int:
    found = Forge.SelfQ(state)
    return -1 if found is None else int(found)


def Label(cell: object, *, melabel: str = '', isstateself: bool = False) -> str:
    soul = str(getattr(cell, 'soul', getattr(cell, 'Soul', '')) or '').strip()
    if soul:
        return soul
    if isstateself and melabel:
        return melabel
    return ''


def SelectedName(cache: Cache, state: object) -> str:
    cells = list(getattr(state, 'cells', []) or [])
    me = StateSelf(state)
    city = ActiveState(cache)
    if me >= 0 and city == me:
        return str(getattr(cache, 'name', '') or '')
    if 0 <= city < len(cells):
        soul = str(getattr(cells[city], 'soul', getattr(cells[city], 'Soul', '')) or '').strip()
        if soul:
            return soul
    return ''


def Possessive(name: str) -> str:
    text = str(name or '').strip()
    if not text:
        return ''
    if text.lower().endswith('s'):
        return text + "'"
    return text + "'s"


def PurgePreview(cache: Cache, state: object) -> str:
    me = StateSelf(state)
    city = ActiveState(cache)
    if me >= 0 and city == me:
        return 'Purge All Locksets'
    name = SelectedName(cache, state) or 'Target'
    return f'Purge {Possessive(name)} Lockset'


def StateView(cache: Cache, state: object) -> Dict[str, object]:
    action = StateAction(cache)
    focus = StateFocus(cache)
    me = StateSelf(state)
    rank = None if me < 0 else me + 1
    return {
        'Action': action,
        'Focus': focus,
        'Self': me,
        'ActiveState': ActiveState(cache),
        'Target': StateTarget(cache, state),
        'Floor': Forge.ActionFloor(action, rank=rank, city=None if me < 0 else me),
        'Label': Forge.ActionSpineLabel(action, rank=rank, city=None if me < 0 else me),
    }


def BuildBanner(lux: Dict[str, str], title: str = 'BYZANTIUM') -> List[str]:
    reset = lux['Reset']
    return [
        lux['Flicker1'] + '+    ' + reset,
        lux['Flicker2'] + '•  •  · ' + lux['Ash'] + ')══{≡≡≡≡≡≡≡≡>     ' + lux['Flicker2'] + '+  ' + lux['Salt'] + title + reset + lux['Flicker3'] + '  +     ' + lux['Ash'] + '<≡≡≡≡≡≡≡≡}══( ' + lux['Flicker3'] + '·  •  •    ' + reset,
        lux['Flicker4'] + '+    ' + reset,
    ]


def LoreLines() -> List[str]:
    return [
        '   There is no return to a previous state. No correction, no reconciliation.',
        '   Only continuation under strain. What settles becomes structure. What',
        '   moves becomes the next imbalance.',
        '',
        '   This is not a fixed system. It is a shared dream held under tension,',
        '   a pattern that persists through adjustment. At times it feels stable,',
        '   at times it feels like a shared nightmare. Both are the same.',
        '',
        '   Positions are not assigned. They are maintained. A general holds only',
        '   as long as he can bear what gathers beneath him. Captains decide long',
        '   before the ranks ever change whether he is worth his salt.',
        '',
        '   In Byzantium equivocation is rare, but if a man speaks with two tongues,',
        '   both will be heard. State does not decide what was true. It absorbs',
        '   the burden and maintains continuity.',
        '', 
        '   The dream resolves when no front exceeds another.', 
        '   All debts must be settled.',
        '',
        '   Thus,',
        '', 
        '   We are souls in roles that shape the whole',
        '   We may push, but we may never pull',
        '   When we take beyond our share, the chain is culled',
        ''
    ] + [''] * 3


def RenderLore(cache: Cache) -> str:
    lux = Forge.Crucible(cache)
    ash = lux['Ash']
    salt = lux['Salt']
    reset = lux['Reset']
    lore = LoreLines()
    visible = LoreCount
    maxoffset = max(0, len(lore) - visible)
    try:
        offset = int(getattr(cache, 'lorescroll', 0) or 0)
    except Exception:
        offset = 0
    offset = max(0, min(offset, maxoffset))
    try:
        cache.lorescroll = offset
    except Exception:
        pass
    window = lore[offset:offset + visible]
    lines: List[str] = [
        ash + Forge.Hline + reset,
        CenterTerm(''),
        CenterTerm(lux['Flicker1'] + '·  •  •  ' + salt + Action.Lore.value.upper() + reset + lux['Flicker2'] + '  •  •  ·'),
        CenterTerm(''),
    ]
    for line in window:
        lines.append(ClipTerm(ash + line + reset if line else ''))
    lines.append(ClipTerm(''))
    return Forge.FrameTextScreen(lines)


def RenderMenu(cache: Cache) -> List[str]:
    lux = Forge.Crucible(cache)
    reset = lux['Reset']
    leader = (getattr(cache, 'name', '') or '').ljust(NameWidth)[:NameWidth]
    chunks = [(lux['Flicker2'] if i == cache.menuq else lux['Ash']) + act.value.upper() + reset for i, act in enumerate(Menu)]
    return [
        ClipTerm(lux['Ash'] + Forge.Hline + reset),
        ClipTerm(f"{lux['Flicker1']}{leader}{reset} {lux['Flicker3']}>>>{reset} " + ' • '.join(chunks)),
        ClipTerm(lux['Ash'] + Forge.Hline + reset),
    ]


def AshSplit(raw: str) -> tuple[str, str]:
    text = str(raw or '')
    body, sep, kind = text.rpartition('|')
    if sep:
        return kind.strip().lower(), body
    return '', text


def BuildMonument(line: object) -> str:
    raw = str(line or '')
    if not raw.strip():
        return ''
    head, score, body = Forge.ParseMonument(raw, name=NameWidth)
    name = str(head or '').ljust(NameWidth)[:NameWidth]
    kind, tail = AshSplit(str(body or ''))
    kind = str(kind or '').strip().lower()
    if kind == 'defect':
        scorefield = 'Defected'.rjust(Forge.SaltWidth)[:Forge.SaltWidth]
    elif score is None:
        scorefield = ''.rjust(Forge.SaltWidth)
    else:
        try:
            scorefield = Forge.SpineCost(int(str(score).replace(',', '')), width=Forge.SaltWidth, signed=True)
        except Exception:
            scorefield = str(score).rjust(Forge.SaltWidth)[:Forge.SaltWidth]
    return f'{name} {scorefield}:{tail}' if score is not None or tail else ''


def RenderBanner(cache: Cache, monuments: List[str]) -> List[str]:
    lux = Forge.Crucible(cache)
    reset = lux['Reset']
    if getattr(cache, 'banner', True):
        payload = [CenterTerm(line) for line in BuildBanner(lux, title='BYZANTIUM')]
    else:
        cleaned = [str(m).strip() for m in monuments if str(m).strip()]
        if cleaned:
            payload = [lux['Ash'] + BuildMonument(line) + reset for line in (cleaned + ['', '', ''])[:3]]
        else:
            payload = ['', CenterTerm(lux['Ash'] + 'Tabula Rasa' + reset)[2:], '']
    return [ClipTerm(line) for line in payload]


def BoardNameColor(lux: Dict[str, str], action: Action, focus: Focus, me: int, idx: int, city: int, target: Optional[int], state: object) -> str:
    if action == Action.Wrath and focus in (Focus.TableLock, Focus.Spine):
        return lux['Flicker1']
    if action == Action.Purge and focus == Focus.TableMove and idx == city:
        return lux['Flicker6']
    if (focus == Focus.TableMove and idx == city) or (action in (Action.Whisper, Action.Purge) and focus == Focus.Spine and idx == (target if target is not None else -1)):
        return lux['Flicker2']
    if action == Action.Rally and focus in (Focus.TableLock, Focus.Spine):
        return lux['Flicker3'] if me >= 0 and idx // Rows == me // Rows else lux['Ash']
    if action == Action.Defect and focus == Focus.TableMove:
        if idx == me:
            return lux['Flicker4']
        return lux['Ember'] if me >= 0 and Forge.DefectViable(state, me, idx) else lux['Ash']
    if action == Action.Defect and focus == Focus.Spine:
        return lux['Flicker1'] if idx == me or idx == (target if target is not None else -1) else lux['Ash']
    if me >= 0 and idx == me:
        return lux['Flicker2'] if idx % Rows == 0 else lux['Flicker5']
    if idx % Rows == 0:
        return lux['Ember']
    return lux['Ash']


def RenderBoard(cache: Cache, state: object) -> List[str]:
    lux = Forge.Crucible(cache)
    reset = lux['Reset']
    cells = list(getattr(state, 'cells', []) or [])
    view = StateView(cache, state)
    action = view['Action']
    focus = view['Focus']
    me = int(view['Self'])
    city = int(view['ActiveState'])
    target = view['Target']

    def FormatCell(idx: int) -> str:
        cell = cells[idx]
        name = Label(cell, melabel=str(getattr(cache, 'name', '') or ''), isstateself=me >= 0 and idx == me).ljust(NameWidth)[:NameWidth]
        raw = f'{int(Forge.Amount(cell)):,}'
        purge = getattr(cell, 'purge', None)
        lockbit = int(getattr(purge, 'lockbit', 0) or 0)
        chainbit = int(getattr(purge, 'chainbit', 0) or 0)
        if lockbit == 0:
            salt = raw.rjust(8)
        else:
            pluscol = lux['Salt'] if chainbit == 1 else lux['Ash']
            number = ('+' + raw).rjust(8)
            plusat = number.find('+')
            salt = number[:plusat] + pluscol + '+' + reset + number[plusat + 1:]
        namecol = BoardNameColor(lux, action, focus, me, idx, city, target, state)
        return f'{namecol}{name}{reset} {salt}'

    lines = [ClipTerm(lux['Ash'] + Forge.Hline + reset)]
    for row in range(Rows):
        parts = [FormatCell(col * Rows + row) for col in range(Columns) if col * Rows + row < len(cells)]
        lines.append(ClipTerm((' ' * Colonnade).join(parts)))
    lines.append(ClipTerm(lux['Salt'] + Forge.Hline + reset))
    return lines


def Compose(label: str, cost: int, draft: str, *, armed: bool, lux: Dict[str, str]) -> str:
    ash = lux['Ash']
    flame = lux['Flicker1']
    reset = lux['Reset']
    namefixed = str(label or '').ljust(NameWidth)[:NameWidth]
    costtext = Forge.SpineCost(cost)
    if armed:
        return f'{ash}{namefixed}{reset} {ash}{costtext}{reset}{ash}:{reset}'
    return f'{flame}{namefixed}{reset} {costtext}{ash}:{reset}{draft}'


def Description(action: Action, anchorcol: int, *, state: object) -> str:
    desc = Forge.ActionDesc(action)
    if not desc:
        return Forge.ActionPreview(action)
    rank = Forge.ResolveRank(state)
    floor = Forge.ActionFloor(action, rank=rank)
    namefixed = action.value.upper().ljust(NameWidth)[:NameWidth]
    costraw = f'{floor:+,}'
    costwidth = max(len(costraw), max(0, anchorcol - (NameWidth + 1)))
    return f'{namefixed} {costraw.rjust(costwidth)}:{desc}'


def BuildSpine(cache: Cache, state: object, *, lux: Optional[Dict[str, str]] = None, anchorcol: int = 0) -> List[str]:
    if lux is None:
        lux = Forge.Crucible(cache)
    ash = lux['Ash']
    white = lux['Salt']
    reset = lux['Reset']
    view = StateView(cache, state)
    action = view['Action']
    focus = view['Focus']
    label = view['Label']
    floor = int(view['Floor'])
    if focus == Focus.Menu:
        return [ash + CenterTerm(Forge.ActionPreview(action)) + reset, white + Forge.Hline + reset]
    if action == Action.Whisper and focus in (Focus.TableMove, Focus.Spine):
        line = Compose(SelectedName(cache, state), StateAmount(cache), StateText(cache), armed=focus == Focus.TableMove, lux=lux)
        return [line, white + Forge.Hline + reset]
    if action in (Action.Wrath, Action.Rally) and focus in (Focus.TableLock, Focus.Spine):
        spinelabel = str(label or Forge.ActionSpineLabel(action, rank=Forge.ResolveRank(state)) or action.value.upper())
        line = Compose(spinelabel, StateAmount(cache), StateText(cache), armed=focus == Focus.TableLock, lux=lux)
        return [line, white + Forge.Hline + reset]
    if action == Action.Defect and focus in (Focus.TableMove, Focus.Spine):
        line = Compose('DEFECT', floor, StateText(cache), armed=focus == Focus.TableMove, lux=lux)
        return [line, white + Forge.Hline + reset]
    if action == Action.Purge and focus == Focus.TableMove:
        return [ash + CenterTerm(PurgePreview(cache, state)) + reset, white + Forge.Hline + reset]
    return [ash + Description(action, anchorcol, state=state) + reset, white + Forge.Hline + reset]


def AshTotal(raw: str) -> int:
    digits = ''.join(ch for ch in str(raw or '') if ch.isdigit())
    try:
        return int(digits or '0')
    except Exception:
        return 0


def AshColor(lux: Dict[str, str], mine: bool) -> str:
    return lux['Ash'] if mine else lux['Salt']


def AshBroadcast(kind: str) -> bool:
    return kind in ('wrath', 'defect')


def AshTag(lux: Dict[str, str], mine: bool, kind: str, amounttotal: int = 0) -> str:
    if mine:
        return lux['Ash']
    if kind == 'wrath':
        return lux['Flicker4']
    if kind == 'rally':
        return lux['Flicker6']
    if kind == 'defect':
        return lux['Flicker2'] if amounttotal in (2000, 10000) else lux['Flicker6']
    return lux['Salt']


def AshText(raw: str, kind: str) -> str:
    return 'Defected' if kind == 'defect' else raw


def RenderAshEntry(cache: Cache, lux: Dict[str, str], entry: object) -> str:
    reset = lux['Reset']
    mine = str(getattr(cache, 'name', '') or '').strip()

    if isinstance(entry, dict):
        kind = str(entry.get('kind', '') or '').strip().lower()
        sender = str(entry.get('sender', '') or '').strip()
        name = str(entry.get('name', '') or sender).ljust(NameWidth)[:NameWidth]
        amounttext = str(entry.get('left', '') or '')
        text = str(entry.get('text', '') or '')
        amounttotal = int(entry.get('total', 0) or 0)
    else:
        chan = ''
        line = ''
        if isinstance(entry, tuple) and len(entry) >= 2:
            chan, line = entry[0], entry[1]
        rawline = str(line or '')
        sender = rawline[:NameWidth].strip()
        name = rawline[:NameWidth].ljust(NameWidth)[:NameWidth]
        cut = min(len(rawline), NameWidth + 1)
        colon = rawline.find(':', cut)
        if colon < 0:
            colon = len(rawline)
        amounttext = rawline[cut:colon].strip()
        rawtext = rawline[colon + 1:] if colon < len(rawline) else ''
        kindtext, body = AshSplit(rawtext)
        kind = kindtext or str(chan or '').strip().lower()
        text = body if kindtext else rawtext
        amounttotal = AshTotal(amounttext)

    ismine = bool(sender) and sender == mine
    payload = AshColor(lux, ismine)
    tag = AshTag(lux, ismine, kind, amounttotal)
    left = AshText(amounttext, kind)
    if AshBroadcast(kind) and not ismine:
        payload = lux['Salt']

    nameslot = str(name).ljust(NameWidth)[:NameWidth]
    leftslot = str(left).rjust(Forge.SaltWidth)[:Forge.SaltWidth]
    bodyslot = str(text or '').replace('\r', ' ').replace('\n', ' ')[:Forge.MessageMax]

    return (
        lux['Ash'] + nameslot + reset +
        lux['Ash'] + ' ' + reset +
        tag + leftslot + reset +
        lux['Ash'] + ':' + reset +
        payload + bodyslot + reset
    )


def AllIsAsh(lux: Dict[str, str]) -> str:
    reset = lux['Reset']
    return ClipTerm(CenterTerm(lux['Ash'] + 'all that remains is ash...' + reset))


def RenderAshfall(cache: Cache, state: object, anchorcol: int) -> List[str]:
    lux = Forge.Crucible(cache)
    lines = list(BuildSpine(cache, state, lux=lux, anchorcol=anchorcol))
    ash = list(getattr(cache, 'ash', []) or [])
    if ash:
        feed = ash
        visible = len(feed)
    else:
        feed = list(getattr(cache, 'feed', []) or [])
        visible = min(int(getattr(cache, 'feedcount', 0) or 0), len(feed))

    start = max(0, len(feed) - visible)
    rendered = [RenderAshEntry(cache, lux, entry) for entry in reversed(feed[start:])]
    maxrows = 7

    if len(rendered) <= 3:
        placeholderrow = 3 + len(rendered)
        for row in range(maxrows):
            if row < len(rendered):
                lines.append(rendered[row])
            elif row == placeholderrow:
                lines.append(AllIsAsh(lux))
            else:
                lines.append(ClipTerm(''))
        return lines

    for row in range(maxrows):
        lines.append(rendered[row] if row < len(rendered) else ClipTerm(''))
    return lines


def BottomBar(state: object, *, width: int, lux: Dict[str, str], thresh: int = FinishLine) -> str:
    reset = lux['Reset']
    totals = {teamid: 0 for teamid in range(1, Columns + 1)}
    for index, cell in enumerate(list(getattr(state, 'cells', []) or [])):
        teamid = index // Rows + 1
        if teamid in totals:
            totals[teamid] += int(Forge.Amount(cell))
    pairs = sorted((totals[i], i) for i in totals)
    hotember: set[int] = set()
    lo = bestlo = 0
    besthi = -1
    for j in range(1, len(pairs)):
        if pairs[j][0] - pairs[j - 1][0] > thresh:
            if j - lo > besthi - bestlo + 1:
                bestlo, besthi = (lo, j - 1)
            lo = j
    if len(pairs) - lo > besthi - bestlo + 1:
        bestlo, besthi = (lo, len(pairs) - 1)
    if besthi - bestlo + 1 >= 3:
        hotember.update(pairs[k][1] for k in range(bestlo, besthi + 1))

    def Format(teamid: int, value: int) -> str:
        raw = f'{value:,}'
        if value == 0:
            return f"{lux['Ash']}{raw}{reset}{lux['Ash']}"
        tied = sum(1 for score, _ in pairs if score == value) >= 2
        if tied:
            return f"{lux['Flicker3']}{raw}{reset}{lux['Ash']}"
        if teamid in hotember:
            return f"{lux['Ember']}{raw}{reset}{lux['Ash']}"
        return f"{lux['Ash']}{raw}{reset}{lux['Ash']}"

    center = f" {reset}•{lux['Ash']} ".join(Format(teamid, totals[teamid]) for teamid in range(1, Columns + 1))
    padtotal = width - VisLen(center) - 2
    if padtotal < 2:
        center = center[:max(0, width - 4)]
        padtotal = width - VisLen(center) - 2
    left = max(0, padtotal // 2 - 2)
    right = padtotal - left
    return '=' * left + ' ' + center + ' ' + '=' * right


def RenderTotals(cache: Cache, state: object) -> List[str]:
    lux = Forge.Crucible(cache)
    reset = lux['Reset']
    waiting = str(getattr(cache, 'activerequest', '') or '').strip().upper().startswith('WAIT')
    bar = '=' * Forge.TerminalWidth if waiting else BottomBar(state, width=Forge.TerminalWidth, lux=lux)
    return [ClipTerm(lux['Ash'] + bar + reset)]


def RenderScreen(cache: Cache, state: object) -> str:
    state = MakeState(state)
    if getattr(cache, 'lore', False):
        return RenderLore(cache)
    rawmonuments = getattr(cache, 'monuments', None) or []
    monuments = [str(item[1]) if isinstance(item, tuple) and len(item) >= 2 else str(item) for item in rawmonuments] if isinstance(rawmonuments, list) else []
    anchorcol = Forge.MonumentAnchorCol(monuments, getattr(cache, 'name', '') or '')
    lines: List[str] = []
    lines.extend(RenderMenu(cache))
    lines.extend(RenderBanner(cache, monuments))
    lines.extend(RenderBoard(cache, state))
    lines.extend(RenderAshfall(cache, state, anchorcol))
    while len(lines) < Forge.BodyFillLines:
        lines.append(ClipTerm(''))
    lines.extend(RenderTotals(cache, state))
    return Forge.FrameLines(lines)


def Render(cache: Cache, state: object, ctx: Optional[object] = None) -> str:
    return RenderScreen(cache, state)
