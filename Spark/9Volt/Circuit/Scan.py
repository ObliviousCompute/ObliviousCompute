from __future__ import annotations
import curses
import re

from .Socket import (Circuit, Stages, DisplayWidth, Bar, HeavyBar, TraceRow,
                     LabelSegments, PairedGlyphSegments, TRACE_SOCKET_X, TRACE_ACTION_X,
                     TRACE_LABEL_X, TRACE_INFO_X)
from .Vertex import Glyph, LogGlyphCells

Faint, Lit, Green = 1, 2, 3

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
    Paint(w, y, cx-4, 'Nine-Volt', Ink(Lit, True))
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
    left_x, mid_x, right_x = cx-5, cx, cx+5
    labelAttr = Ink(Lit, True) if selected and not editing else Ink(Faint)
    socketLabel = f'Socket {name}'
    labelX = left_x-len(socketLabel)-3
    Paint(w, y, labelX, socketLabel, labelAttr)
    Paint(w, y, labelX+len(socketLabel)+1, '→', Ink(Faint))
    Paint(w, y+1, right_x+2, '←', Ink(Faint))
    Paint(w, y+1, right_x+4, 'Cohort', labelAttr)
    for index, x, token, observer in (
        (0, left_x, spec.leftCharge, spec.left),
        (1, mid_x, f'( {charge if charge is not None else " "} )', spec.receiver),
        (2, right_x, spec.rightCharge, spec.right),
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

def CellWidth(value):
    return DisplayWidth(value)
def FitColumns(value, limit):
    for i in range(1, len(value) + 1):
        if CellWidth(value[:i]) > limit:
            return value[:i-1]
    return value

def WrapTrace(value, limit):
    if isinstance(value, TraceRow):
        capacity = max(8, limit + 1 - TRACE_INFO_X)
        info = value.info
        if CellWidth(info) <= capacity:
            return [value]
        pieces = []
        while CellWidth(info) > capacity:
            cutoff = FitColumns(info, capacity)
            split = cutoff.rfind(' ')
            if split < 1:
                split = len(cutoff)
            pieces.append(info[:split].rstrip())
            info = info[split:].lstrip()
        pieces.append(info)
        return [TraceRow(value.stage, value.actor, value.kind, value.detail, pieces[0])] + [
            TraceRow('', '', '', '', part) for part in pieces[1:]]
    result = []
    while CellWidth(value) > limit:
        cutoff = FitColumns(value, limit)
        split = cutoff.rfind(' ')
        if split < 8:
            split = len(cutoff)
        result.append(value[:split].rstrip())
        value = '    ' + value[split:].lstrip()
    result.append(value)
    return result

GlyphPairPattern = re.compile(
    '(' + '|'.join(re.escape(g) for g in LogGlyphCells) +
    r')([01]{3}|Apex|Loop|Hook|Gate)\1'
)

def PaintInlineLabels(w, y, x, value, attr):
    end = 0
    for match in GlyphPairPattern.finditer(value):
        prefix = value[end:match.start()]
        if prefix:
            Paint(w, y, x, prefix, attr)
            x += CellWidth(prefix)
        pair = PairedGlyphSegments(match.group())
        assert pair is not None
        segments, cell_count = pair
        for offset, token in segments:
            Paint(w, y, x + offset, token, attr)
        x += cell_count
        end = match.end()
    suffix = value[end:]
    if suffix:
        Paint(w, y, x, suffix, attr)
        x += CellWidth(suffix)
    return x
def PaintTraceEntry(w, y, line, width):
    if isinstance(line, TraceRow):
        if line.stage:
            Paint(w, y, TRACE_SOCKET_X, f'{line.stage}:{line.actor}', Ink(Faint))
            Paint(w, y, TRACE_ACTION_X, line.kind, Ink(Faint))
        for offset, token in LabelSegments(line.detail):
            Paint(w, y, TRACE_LABEL_X + offset, token, Ink(Faint))
        if line.info:
            Paint(w, y, TRACE_INFO_X, line.info, Ink(Faint))
        return
    if line in (Bar, HeavyBar):
        Paint(w, y, 1, line[0] * (width-2), Ink(Faint))
        return
    label = FitColumns(line, width-4)
    x = 1
    parts = label.split('✓')
    for i, part in enumerate(parts):
        x = PaintInlineLabels(w, y, x, part,
                              Ink(Lit, True) if label.startswith('RESOLVED') else Ink(Faint))
        if i < len(parts)-1:
            Paint(w, y, x, '✓', Ink(Green, True))
            x += 1

def Screen(w):
    SetupColors()
    try: curses.curs_set(0)
    except curses.error: pass
    w.keypad(True)
    w.timeout(30)
    c = Circuit()
    selection = 0
    editing = False
    column = 0
    logMode = False
    logScroll = 0
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
                entries = [part for entry in c.log for part in WrapTrace(entry, width-4)]
                if not entries:
                    Paint(w, 3, 2, 'No execution yet.', Ink(Faint))
                end = max(0, len(entries)-logScroll)
                start = max(0, end-(h-4))
                for i, line in enumerate(entries[start:end]):
                    PaintTraceEntry(w, i+3, line, width)
                w.refresh()
                key = w.getch()
                if key in (-1, curses.KEY_RESIZE): continue
                if key == 3: break
                if key == curses.KEY_UP: logScroll = min(len(entries), logScroll+1)
                elif key == curses.KEY_DOWN: logScroll = max(0, logScroll-1)
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
                logScroll = 0
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
