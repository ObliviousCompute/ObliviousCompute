from __future__ import annotations
import curses
import re
import unicodedata
from collections import deque
from dataclasses import dataclass

from .Socket import Circuit, Stages, Actors
from .Vertex import Words, WordOf, Role

Faint, Lit, Green = 1, 2, 3
Glyph = {'000': '⚜️', '111': '⚜️', '100': '🎗️', '011': '🎗️',
         '110': '🪝', '001': '🪝', '010': '🔱', '101': '🔱'}

def Paint(w, y, x, value, attr=0):
    height, width = w.getmaxyx()
    if not (0 <= y < height and 0 <= x < width-1):
        return
    try:
        w.addnstr(y, x, str(value), width-x-1, attr)
    except (curses.error, UnicodeError):
        pass

def Ink(code, bold=False):
    return (curses.color_pair(code) if curses.has_colors() else (curses.A_DIM if code == Faint else 0)) | (curses.A_BOLD if bold else 0)
def Battery(w, y, x, ready):
    Paint(w, y, x, '🔋' if ready else '🪫', Ink(Lit if ready else Faint, ready))
def TitleLine(w, y, cx, ready):
    Battery(w, y, cx-6, ready)
    Paint(w, y, cx-4, 'NineVolt', Ink(Lit, True))
    Battery(w, y, cx+5, ready)
def WordLine(w, y, cx, word, ready=False, activeBit=None):
    Battery(w, y, cx-6, ready)
    if word is not None:
        Paint(w, y, cx-3, Glyph[word], Ink(Lit, True))
        for idx, bit in enumerate(word):
            dim = activeBit is not None and idx != activeBit
            Paint(w, y, cx-1+idx, bit, Ink(Faint if dim else Lit, not dim))
        Paint(w, y, cx+2, Glyph[word], Ink(Lit, True))
    Battery(w, y, cx+5, ready)

def SocketLine(w, y, cx, name, spec, selected, editing, col, charge, output):
    Left, Mid, Right = cx-5, cx, cx+5
    labelAttr = Ink(Lit, True) if selected and not editing else Ink(Faint)
    socketLabel = f'Socket {name}'
    labelX = Left-len(socketLabel)-3
    Paint(w, y, labelX, socketLabel, labelAttr)
    Paint(w, y, labelX+len(socketLabel)+1, '→', Ink(Faint))
    Paint(w, y+1, Right+2, '←', Ink(Faint))
    Paint(w, y+1, Right+4, f'{name} Cohort', labelAttr)
    for index, x, token, observer in (
        (0, Left, spec.leftCharge, spec.left),
        (1, Mid, f'( {charge if charge is not None else " "} )', spec.receiver),
        (2, Right, spec.rightCharge, spec.right),
    ):
        colActive = editing and col == index
        attr = Ink(Lit, True) if (colActive or (selected and not editing)) else Ink(Faint)
        if index == 1:
            tokenAttr = attr
            Paint(w, y, x-2, token, tokenAttr)
        else:
            Paint(w, y, x, token, attr)
        Paint(w, y+1, x, observer, attr)
    Paint(w, y+2, cx, '↓', Ink(Faint))
    WordLine(w, y+3, cx, output, ready=output is not None and not editing)
    Paint(w, y+4, cx, '↓', Ink(Faint))

def SetupColors():
    if curses.has_colors():
        curses.start_color()
        try: curses.use_default_colors()
        except curses.error: pass
        grey = 8 if curses.COLORS >= 16 else curses.COLOR_WHITE
        curses.init_pair(Faint, grey, -1)
        curses.init_pair(Lit, curses.COLOR_WHITE, -1)
        curses.init_pair(Green, curses.COLOR_GREEN, -1)

class Log:
    RoleNames = {'terminal': 'Apex', 'weak': 'Loop', 'linchpin': 'Hook', 'lateral': 'Gate'}
    GlyphCells = {'⚜️': 2, '🎗️': 2, '🪝': 2, '🔱': 2}

    @classmethod
    def RoleLabel(cls, word: str) -> str:
        sign = Glyph[word]
        Space = ' ' if sign in ('⚜️', '🎗️') else ''
        return f'{sign}{Space}{cls.RoleNames[Role(word)]}{sign}'

    @staticmethod
    def VertexLabel(word: str) -> str:
        sign = Glyph[word]
        return f'{sign}{word}{sign}'

    Margin = 1
    SocketWidth = 3
    ActionWidth = len('configured')
    LabelWidth = 8
    SocketX = Margin
    ActionX = SocketX + SocketWidth + 1
    LabelX = ActionX + ActionWidth + 1
    InfoX = LabelX + LabelWidth + 1
    Bar = '─' * 76
    HeavyBar = '=' * 76
    Volt = '⚡Volt⚡'
    Spark = '✨'
    GlyphPairPattern = re.compile('(' + '|'.join(re.escape(g) for g in GlyphCells) +
                                  r')([01]{3}|Apex|Loop|Hook|Gate)\1')

    @dataclass(frozen=True)
    class Row:
        stage: str
        actor: str
        kind: str
        detail: str = ''
        info: str = ''

        def __str__(self) -> str:
            if not self.stage:
                return ' ' * (Log.InfoX - Log.Margin) + self.info
            return (f'{self.stage}:{self.actor}'.ljust(Log.SocketWidth) + ' ' +
                    self.kind.ljust(Log.ActionWidth) + ' ' +
                    Log.LabelField(self.detail) + ' ' + self.info).rstrip()

    def __init__(self):
        self.Scroll = 0
        self.Reset()

    def Reset(self):
        self.Lines = deque(maxlen=1500)
        self.Recorded = {stage: [] for stage in Stages}
        self.Headers = {}
        self.Results = {}
        self.GenesisLines = []
        self.GenesisSeen = set()
        self.FinalLine = ''
        self.Scroll = 0

    def Invalidate(self):
        self.Lines.append('EDIT: output cleared; awaiting next complete Genesis')

    def Begin(self, board):
        self.Reset()
        self.Lines.extend((self.HeavyBar, 'GENESIS ' +
                          ' | '.join(f'{s}:{board[s].Serialize(s)[2:]}' for s in Stages)))

    def GenesisEvent(self, event):
        if event['kind'] == 'genesis':
            self.GenesisSeen.add(event['actor'])
            self.Lines.append(f"GENESIS:{Actors[event['actor']]} adopted complete board")
        else:
            self.Lines.append(f"BOOT:{Actors[event['actor']]} {event['kind']}")

    def GenesisReady(self):
        return len(self.GenesisSeen) == 3

    def GenesisComplete(self):
        self.Lines.append(self.HeavyBar)
        self.GenesisLines = list(self.Lines)

    def OpenStage(self, stage, cfg, word):
        self.Headers[stage] = (f'{stage}: {cfg.leftCharge}{cfg.left}({cfg.receiver})'
                               f'{cfg.right}{cfg.rightCharge} Receiver ({cfg.receiver})'
                               f' Helix ({cfg.left} | {cfg.right}) Input {self.VertexLabel(word)}')

    def RecordEvents(self, events):
        for event in events:
            stage = event.get('stage')
            if stage in self.Recorded:
                self.Recorded[stage].append(event)

    def Resolved(self, stage, cfg, word):
        participants = [cfg.receiver]
        for event in self.Recorded[stage]:
            if event['kind'] in ('append', 'candidate'):
                actor = Actors[event['actor']]
                if actor != participants[-1]:
                    participants.append(actor)
        receivers = [Actors[e['actor']] for e in self.Recorded[stage] if e['kind'] == 'observed']
        if receivers:
            participants.append(receivers[-1])
        target = Stages[Stages.index(stage) + 1] if stage != Stages[-1] else 'Ω'
        self.Results[stage] = (f'{stage}:{cfg.Serialize(stage)[2:]} '
                               f'{"→".join(participants)} ✓ {target}:{self.VertexLabel(word)}')

    def Complete(self, word):
        self.FinalLine = f'Ω:{self.VertexLabel(word)} FINAL OUTPUT'

    @staticmethod
    def DisplayWidth(value: str) -> int:
        return sum(0 if unicodedata.combining(char) or char in '\ufe0e\ufe0f' else
                   2 if unicodedata.east_asian_width(char) in 'WF' or
                   (index + 1 < len(value) and value[index + 1] == '\ufe0f') else 1
                   for index, char in enumerate(value))

    @classmethod
    def PairedGlyphSegments(cls, detail):
        for glyph, cells in cls.GlyphCells.items():
            if detail.startswith(glyph) and detail.endswith(glyph) and len(detail) > 2 * len(glyph):
                text = detail[len(glyph):-len(glyph)]
                Core = cls.DisplayWidth(text)
                return (((0, glyph), (cells, text), (cells + Core, glyph)), 2 * cells + Core)
        return None

    @classmethod
    def LabelSegments(cls, detail):
        if not detail:
            return ()
        pair = cls.PairedGlyphSegments(detail)
        if pair is not None:
            segments, total = pair
            if total <= cls.LabelWidth:
                start = (cls.LabelWidth - total + 1) // 2
                return tuple((start + offset, token) for offset, token in segments)
        cells = cls.DisplayWidth(detail)
        return ((max(0, (cls.LabelWidth - cells + 1) // 2), detail),)

    @classmethod
    def LabelField(cls, detail):
        result, position = '', 0
        for start, token in cls.LabelSegments(detail):
            result += ' ' * max(0, start - position) + token
            position = start + cls.DisplayWidth(token)
        return result + ' ' * max(0, cls.LabelWidth - position)

    @classmethod
    def Describe(cls, event):
        kind = event.get('kind', '')
        reason = event.get('reason') or ''
        charge = event.get('charge', '')
        payload = event.get('payload') or ''
        if kind == 'configured': return event.get('q', '')
        if kind in ('ready', 'genesis', 'input'): return ''
        if kind == 'candidate':
            if event.get('origin') in Words:
                Role = cls.RoleLabel(event['origin'])

                Space = '  ' if Glyph[event['origin']] in ('⚜️', '🎗️') else ''
                return f'{Role}{Space} completes {cls.Volt}'
            return f'Observer completes {cls.Volt}'
        if kind == 'observed': return f'Received {cls.Volt}'
        if kind == 'confirmed': return f'Confirmed {cls.Volt}'
        if kind == 'transist': return 'Transist opens next socket'
        if kind == 'override': return 'Opposing evidence overrides'
        if kind == 'send': return f'Broadcast {cls.Volt}' if WordOf(payload) else cls.Spark
        if kind == 'receive': return '' if WordOf(payload) else cls.Spark
        if kind == 'initial': return cls.Spark
        if kind == 'lock':
            return f'{"Positive" if charge == "+" else "Negative" if charge == "-" else ""} Apex locks'
        if kind == 'reject': return reason
        if kind == 'conflict': return 'Conflicting completed Volt'
        if kind == 'open' or reason: return reason
        return ''

    @classmethod
    def EventRow(cls, event):
        stage, kind = event['stage'], event['kind']
        payload = event.get('payload') or ''
        info = cls.Describe(event)
        if kind == 'configured':
            detail = event['q'] if event['q'] in Words else 'Helix'
            info = detail
        elif kind in ('initial', 'lock'):
            detail = cls.VertexLabel(event['q']) if event['q'] in Words else event.get('charge', '')
        elif kind == 'input':
            detail = cls.VertexLabel(event['word'])
        else:
            detail = payload or event.get('after') or event.get('word') or event.get('role') or ''
            if detail in Words:
                detail = cls.VertexLabel(detail)
            elif WordOf(detail):
                detail = cls.VertexLabel(WordOf(detail))
        return cls.Row(stage, Actors[event['actor']], 'Lock' if kind in ('jump', 'lock') else kind, detail, info)

    def Refresh(self):
        lines = list(self.GenesisLines)
        for stage in Stages:
            if stage not in self.Headers:
                break
            lines.append(self.Headers[stage])
            events = self.Recorded[stage]
            configured = {e['actor']: self.EventRow(e) for e in events if e['kind'] == 'configured'}
            lines.extend(configured[index] for index in sorted(configured))
            if len(configured) == 3:
                lines.append(self.Bar)
            for event in events:
                if event['kind'] == 'configured':
                    continue
                lines.append(self.EventRow(event))
                if event['kind'] == 'observed' or (
                        event['kind'] == 'send' and WordOf(event.get('payload') or '')):
                    lines.append(self.Row(stage, Actors[event['actor']], 'Quiet...'))
            if stage in self.Results:
                lines.extend(('', 'RESOLVED ' + self.Results[stage], self.HeavyBar))
        if self.FinalLine:
            lines.append(self.FinalLine)
        self.Lines = deque(lines, maxlen=1500)

    @classmethod
    def FitColumns(cls, value, limit):
        for index in range(1, len(value) + 1):
            if cls.DisplayWidth(value[:index]) > limit:
                return value[:index-1]
        return value

    @classmethod
    def Wrap(cls, value, limit):
        if isinstance(value, cls.Row):
            capacity = max(8, limit + 1 - cls.InfoX)
            info = value.info
            if cls.DisplayWidth(info) <= capacity:
                return [value]
            pieces = []
            while cls.DisplayWidth(info) > capacity:
                cutoff = cls.FitColumns(info, capacity)
                split = cutoff.rfind(' ')
                if split < 1:
                    split = len(cutoff)
                pieces.append(info[:split].rstrip())
                info = info[split:].lstrip()
            pieces.append(info)
            return [cls.Row(value.stage, value.actor, value.kind, value.detail, pieces[0])] + [
                cls.Row('', '', '', '', part) for part in pieces[1:]]
        result = []
        while cls.DisplayWidth(value) > limit:
            cutoff = cls.FitColumns(value, limit)
            split = cutoff.rfind(' ')
            if split < 8:
                split = len(cutoff)
            result.append(value[:split].rstrip())
            value = '    ' + value[split:].lstrip()
        result.append(value)
        return result

    @classmethod
    def PaintInlineLabels(cls, w, y, x, value, attr):
        end = 0
        for match in cls.GlyphPairPattern.finditer(value):
            prefix = value[end:match.start()]
            if prefix:
                Paint(w, y, x, prefix, attr)
                x += cls.DisplayWidth(prefix)
            pair = cls.PairedGlyphSegments(match.group())
            assert pair is not None
            segments, CellCount = pair
            for offset, token in segments:
                Paint(w, y, x + offset, token, attr)
            x += CellCount
            end = match.end()
        suffix = value[end:]
        if suffix:
            Paint(w, y, x, suffix, attr)
            x += cls.DisplayWidth(suffix)
        return x

    @classmethod
    def PaintEntry(cls, w, y, line, width):
        if isinstance(line, cls.Row):
            if line.stage:
                Paint(w, y, cls.SocketX, f'{line.stage}:{line.actor}', Ink(Faint))
                Paint(w, y, cls.ActionX, line.kind, Ink(Faint))
            for offset, token in cls.LabelSegments(line.detail):
                Paint(w, y, cls.LabelX + offset, token, Ink(Faint))
            if line.info:
                Paint(w, y, cls.InfoX, line.info, Ink(Faint))
            return
        if line in (cls.Bar, cls.HeavyBar):
            Paint(w, y, cls.Margin, line[0] * (width-2), Ink(Faint))
            return
        label = cls.FitColumns(line, width-4)
        x = cls.Margin
        parts = label.split('✓')
        for index, part in enumerate(parts):
            x = cls.PaintInlineLabels(w, y, x, part,
                                      Ink(Lit, True) if label.startswith('RESOLVED') else Ink(Faint))
            if index < len(parts)-1:
                Paint(w, y, x, '✓', Ink(Green, True))
                x += 1

    def Draw(self, w, height, width):
        entries = [part for entry in self.Lines for part in self.Wrap(entry, width-4)]
        if not entries:
            Paint(w, 3, 2, 'No execution yet.', Ink(Faint))
        end = max(0, len(entries)-self.Scroll)
        start = max(0, end-(height-4))
        for index, line in enumerate(entries[start:end]):
            self.PaintEntry(w, index+3, line, width)
        return len(entries)

    def ScrollBy(self, change, count):
        self.Scroll = max(0, min(count, self.Scroll + change))

def Screen(w):
    SetupColors()
    try: curses.curs_set(0)
    except curses.error: pass
    w.keypad(True)
    w.timeout(30)
    c = Circuit(Log=Log())
    selection = 0
    editing = False
    column = 0
    logMode = False
    try:
        while True:
            c.Poll()
            w.erase()
            h, width = w.getmaxyx()
            cx = width // 2
            if h < 24 or width < 80:
                Paint(w, 1, 1, 'NineVolt requires at least an 80x24 terminal.', Ink(Lit))
                Paint(w, 2, 1, 'Resize terminal, or (Q)UIT.', Ink(Faint))
                w.refresh()
                key = w.getch()
                if key in (3, ord('q'), ord('Q')): break
                continue
            if logMode:
                TitleLine(w, 1, cx, c.mode in ('running', 'complete') and not c.dirty)
                count = c.Log.Draw(w, h, width)
                w.refresh()
                key = w.getch()
                if key in (-1, curses.KEY_RESIZE): continue
                if key == 3: break
                if key == curses.KEY_UP: c.Log.ScrollBy(1, count)
                elif key == curses.KEY_DOWN: c.Log.ScrollBy(-1, count)
                else: logMode = False
                continue
            isLive = c.mode in ('running', 'complete') and not editing and not c.dirty
            TitleLine(w, 1, cx, isLive)
            Paint(w, 3, cx-len('INPUT')//2, 'INPUT',
                  Ink(Lit, True) if selection == 0 else Ink(Faint))
            WordLine(w, 5, cx, c.inputWord, ready=isLive,
                      activeBit=column if (editing and selection == 0) else None)
            Paint(w, 6, cx, '↓', Ink(Faint))
            for i, stage in enumerate(Stages):
                SocketLine(w, 7 + 5*i, cx, stage, c.board[stage],
                            selected=selection == i+1,
                            editing=editing and selection == i+1,
                            col=column, charge=c.injected[stage],
                            output=c.output[stage])
            Paint(w, 22, cx, 'Ω', Ink(Lit, True) if c.mode == 'complete' else Ink(Faint))
            footer = '(L)OG (Q)UIT'
            Paint(w, 22, width-len(footer)-2, footer, Ink(Faint))
            elapsed = c.Elapsed()
            if elapsed is not None:
                Paint(w, 22, 2, f'{elapsed:.3f} s', Ink(Faint))
            w.refresh()
            key = w.getch()
            if key in (3, ord('q'), ord('Q')): break
            if key in (ord('l'), ord('L')):
                logMode = True
                c.Log.Scroll = 0
                continue
            if key in (10, 13, curses.KEY_ENTER):
                if not editing:
                    editing = True
                    column = 0 if selection == 0 else 1
                    c.Invalidate()
                else:
                    editing = False
                    if selection == 0:
                        c.Begin()
                continue
            if editing:
                if key == curses.KEY_LEFT: column = max(0, column-1)
                elif key == curses.KEY_RIGHT: column = min(2, column+1)
                elif key in (curses.KEY_UP, curses.KEY_DOWN):
                    delta = 1 if key == curses.KEY_UP else -1
                    if selection == 0:
                        bits = list(c.inputWord)
                        bits[column] = '0' if bits[column] == '1' else '1'
                        c.inputWord = ''.join(bits)
                    else:
                        sock = c.board[Stages[selection-1]]
                        if column == 1: sock.CycleReceiver(delta)
                        else: sock.FlipHelix(column, delta)
                    c.Invalidate()
            elif key == curses.KEY_UP:
                selection = (selection-1)%4
            elif key == curses.KEY_DOWN:
                selection = (selection+1)%4
    finally:
        c.Close()

def Run():
    try:
        curses.wrapper(Screen)
    except KeyboardInterrupt:
        pass
