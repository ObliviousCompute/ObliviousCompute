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
from Forge import Cache, Focus

InitCache = Citadel.InitCache
DispatchToken = Citadel.Dispatch
InputBuffer = Citadel.InputBuffer

Fps: float = 55.0
WaitingDots: int = int(Fps)
DefaultMode = 'Campaign'
DefaultSkeleton = 'Skeleton'
DefaultSecret = 'Password'
DefaultGate = '9000'
DefaultGenesis = '1'
DefaultSoul = 'Satoshi'
MaxFields = 8
Options: List[str] = ['Campaign', 'Siege', 'Exit']
TitleFields: List[str] = ['mode', 'gate', 'skeleton', 'soul', 'secret', 'genesis']
FieldDefaults: Dict[str, str] = {
    'mode': DefaultMode,
    'gate': DefaultGate,
    'skeleton': DefaultSkeleton,
    'soul': '',
    'secret': DefaultSecret,
    'genesis': DefaultGenesis,
}
FieldLimits: Dict[str, int] = {
    'mode': 8,
    'gate': 6,
    'skeleton': MaxFields,
    'soul': MaxFields,
    'secret': MaxFields,
    'genesis': 2,
}
EditFields = {'gate', 'skeleton', 'soul', 'secret', 'genesis'}


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


def EmptyState() -> PlaceholderState:
    return PlaceholderState()


class Core:
    def __init__(self) -> None:
        self.Spire = Spire
        self.Citadel = Citadel.Citadel
        self.Vault = None

    def Intent(self, value):
        if self.Vault is None:
            return None
        return self.Vault.Intent(value)


class Gateway:
    def __init__(self) -> None:
        self.Core = Core()
        self.State: Optional[State] = None
        self.ListenPort = ParsePorts()
        self.Runtime = EmptyState()
        self.Cache = InitCache(self.Runtime)
        EnsureState(self.Cache)
        self.Cache.mode = DefaultMode
        self.Cache.titlestep = 0
        self.Cache.titleselect = 0
        self.Cache.cursorposition = len(DefaultMode)
        if self.ListenPort is not None:
            self.Cache.gate = str(self.ListenPort)
        Citadel.BindCore(self.Core)

    def BuildState(self) -> State:
        self.State = BuildState(self.Cache)
        return self.State

    def Boot(self) -> object:
        self.BuildState()
        self.Cache.gatejam = False
        self.Cache.win = False
        self.Cache.exit = False
        self.Cache.activerequest = None
        self.Cache.waitingframe = 0
        try:
            self.Core.Vault = Vault.Vault(state=self.State, citadel=self.Core.Citadel)
        except Exception:
            self.Core.Vault = None
            self.Cache.waiting = False
            self.Cache.gatejam = True
            self.Cache.gate = str(getattr(self.State, 'gate', self.Cache.gate) or self.Cache.gate)
            return self.Runtime
        Citadel.BindCore(self.Core)
        self.Cache.waiting = True
        self.Cache.waitingstarted = time.monotonic()
        return EmptyState()

    def Dispatch(self, statevalue: object, token):
        kind, value = token
        if kind == 'Interrupt':
            return (self.Cache, statevalue, True)
        if getattr(self.Cache, 'exit', False):
            return (self.Cache, statevalue, True)
        if getattr(self.Cache, 'win', False) or getattr(self.Cache, 'gatejam', False) or getattr(self.Cache, 'waiting', False):
            self.Cache.win = False
            self.Cache.gatejam = False
            self.Cache.waiting = False
            self.Cache.waitingframe = 0
            self.Cache.exit = True
            return (self.Cache, statevalue, False)
        if self.Cache.focus == Focus.Title:
            return HandlePortcullis(self, self.Cache, statevalue, kind, value)
        cachevalue, statevalue, shouldquit = DispatchToken(self.Cache, statevalue, token)
        if shouldquit:
            self.Cache.exit = True
            return (self.Cache, statevalue, False)
        return (cachevalue, statevalue, shouldquit)

    def RenderState(self, statevalue: object) -> object:
        self.Cache.waitingframe = int(getattr(self.Cache, 'waitingframe', 0) or 0) + 1
        surfaced = getattr(self.Cache, 'state', None)
        if getattr(self.Cache, 'waiting', False):
            if StateReady(surfaced):
                self.Cache.waiting = False
                self.Cache.focus = Focus.Menu
                if hasattr(self.Cache, 'SyncIntent'):
                    self.Cache.SyncIntent()
        if surfaced is None:
            return statevalue
        if Victory(surfaced):
            self.Cache.win = True
        return surfaced

    def Render(self, statevalue: object) -> str:
        if getattr(self.Cache, 'exit', False):
            return PortcullisExit(self.Cache)
        if getattr(self.Cache, 'win', False):
            return PortcullisVictory(self.Cache)
        if getattr(self.Cache, 'gatejam', False):
            return PortcullisJammed(self.Cache)
        if getattr(self.Cache, 'waiting', False):
            return PortcullisHold(self.Cache)
        if getattr(self.Cache, 'focus', None) == Focus.Title:
            return PortcullisEngaged(self.Cache)
        return Spire.Render(self.Cache, statevalue)


def ParsePorts() -> Optional[int]:
    listenport: Optional[int] = None
    try:
        if len(sys.argv) >= 2:
            listenport = int(sys.argv[1])
    except Exception:
        listenport = None
    return listenport


def SafeInt(text: object, fallback: int) -> int:
    try:
        raw = str(text or '').strip()
        return int(raw) if raw else int(fallback)
    except Exception:
        return int(fallback)


def EnsureState(cache: Cache) -> None:
    if not hasattr(cache, 'titlestep'):
        cache.titlestep = 0
    if not hasattr(cache, 'titleselect'):
        cache.titleselect = int(getattr(cache, 'titlestep', 0) or 0)
    if not hasattr(cache, 'cursorposition'):
        cache.cursorposition = 0
    if not hasattr(cache, 'waiting'):
        cache.waiting = False
    if not hasattr(cache, 'waitingframe'):
        cache.waitingframe = 0
    if not hasattr(cache, 'waitingstarted'):
        cache.waitingstarted = 0.0
    if not hasattr(cache, 'exit'):
        cache.exit = False
    if not hasattr(cache, 'gatejam'):
        cache.gatejam = False
    if not hasattr(cache, 'win'):
        cache.win = False
    for key, default in FieldDefaults.items():
        if not hasattr(cache, key):
            setattr(cache, key, default)
    rawmode = str(getattr(cache, 'mode', DefaultMode) or DefaultMode).strip()
    cache.mode = rawmode if rawmode in Options else DefaultMode
    cache.name = str(getattr(cache, 'soul', '') or '')
    rawgate = str(getattr(cache, 'gate', FieldDefaults['gate']) or '')
    cache.gate = ''.join((character for character in rawgate if character.isdigit()))[:FieldLimits['gate']]
    rawgenesis = str(getattr(cache, 'genesis', FieldDefaults['genesis']) or '')
    cache.genesis = ''.join((character for character in rawgenesis if character.isdigit()))[:FieldLimits['genesis']]
    stepraw = getattr(cache, 'titleselect', getattr(cache, 'titlestep', 0))
    step = max(0, min(len(TitleFields) - 1, int(stepraw or 0)))
    cache.titlestep = step
    cache.titleselect = step
    field = TitleFields[step]
    value = str(getattr(cache, field, FieldDefaults[field]) or '')
    cursorvalue = int(getattr(cache, 'cursorposition', len(value)) or 0)
    cache.cursorposition = max(0, min(len(value), cursorvalue))


def PortcullisField(cache: Cache) -> str:
    EnsureState(cache)
    return TitleFields[int(getattr(cache, 'titlestep', 0) or 0)]


def FieldValue(cache: Cache, field: str) -> str:
    EnsureState(cache)
    return str(getattr(cache, field, FieldDefaults[field]) or '')


def SetFieldValue(cache: Cache, field: str, value: str) -> None:
    limit = int(FieldLimits.get(field, MaxFields))
    value = str(value or '')[:limit]
    if field in ('gate', 'genesis'):
        value = ''.join((character for character in value if character.isdigit()))
    if field == 'mode':
        value = str(value or DefaultMode).strip()
        value = value if value in Options else DefaultMode
    setattr(cache, field, value)
    if field == 'soul':
        cache.name = value


def ShiftField(cache: Cache, delta: int) -> None:
    EnsureState(cache)
    step = int(getattr(cache, 'titleselect', getattr(cache, 'titlestep', 0)) or 0) + int(delta)
    step = max(0, min(len(TitleFields) - 1, step))
    cache.titlestep = step
    cache.titleselect = step
    field = TitleFields[step]
    cache.cursorposition = len(FieldValue(cache, field))


def Insert(cache: Cache, character: str) -> None:
    field = PortcullisField(cache)
    if field not in EditFields:
        return
    value = FieldValue(cache, field)
    limit = int(FieldLimits.get(field, MaxFields))
    if len(value) >= limit:
        return
    if field in ('gate', 'genesis') and (not character.isdigit()):
        return
    position = int(getattr(cache, 'cursorposition', len(value)) or 0)
    position = max(0, min(len(value), position))
    newvalue = value[:position] + character + value[position:]
    SetFieldValue(cache, field, newvalue)
    cache.cursorposition = min(len(FieldValue(cache, field)), position + 1)


def Backspace(cache: Cache) -> None:
    field = PortcullisField(cache)
    if field not in EditFields:
        return
    value = FieldValue(cache, field)
    position = int(getattr(cache, 'cursorposition', len(value)) or 0)
    if position <= 0 or not value:
        return
    newvalue = value[:position - 1] + value[position:]
    SetFieldValue(cache, field, newvalue)
    cache.cursorposition = max(0, position - 1)


def ToggleMode(cache: Cache, delta: int) -> None:
    current = str(FieldValue(cache, 'mode') or DefaultMode)
    try:
        index = Options.index(current)
    except ValueError:
        index = 0
    index = (index + int(delta)) % len(Options)
    SetFieldValue(cache, 'mode', Options[index])
    cache.cursorposition = len(FieldValue(cache, 'mode'))


def BuildState(cache: Cache) -> State:
    EnsureState(cache)
    mode = FieldValue(cache, 'mode') or DefaultMode
    rawgate = FieldValue(cache, 'gate')
    gatenumber = SafeInt(rawgate, int(DefaultGate)) if rawgate else int(DefaultGate)
    if gatenumber < 1024 or gatenumber > 65535:
        gatenumber = int(DefaultGate)
    gate = str(gatenumber)
    skeleton = FieldValue(cache, 'skeleton') or DefaultSkeleton
    soul = FieldValue(cache, 'soul') or DefaultSoul
    secret = FieldValue(cache, 'secret') or DefaultSecret
    rawgenesis = FieldValue(cache, 'genesis')
    genesis = max(1, min(24, SafeInt(rawgenesis or DefaultGenesis, 1)))
    SetFieldValue(cache, 'mode', mode)
    SetFieldValue(cache, 'gate', gate)
    SetFieldValue(cache, 'skeleton', skeleton)
    SetFieldValue(cache, 'soul', soul)
    SetFieldValue(cache, 'secret', secret)
    SetFieldValue(cache, 'genesis', str(genesis))
    cache.name = soul
    return State(mode=mode, gate=gate, skeleton=skeleton, soul=soul, secret=secret, genesis=genesis)


def StateReady(statevalue: object) -> bool:
    cells = getattr(statevalue, 'cells', None)
    return bool(cells)


def Victory(statevalue: object) -> bool:
    cells = list(getattr(statevalue, 'cells', []) or [])
    if len(cells) < 24:
        return False
    totals = [0, 0, 0, 0]
    for index, cell in enumerate(cells[:24]):
        totals[index // 6] += int(Forge.Amount(cell))
    return all((total == 250000 for total in totals))


def BuildPortcullis(cache: Cache, label: str, value: str = '', subtitle: str = '', *, titletext: str = 'BYZANTIUM', showfield: bool = False) -> List[str]:
    lux = Forge.Crucible(cache)
    lines: List[str] = [lux['Ash'] + Forge.Hline + lux['Reset']]
    lines.extend(Forge.MastLines(lux, title=titletext))
    lines.append(Forge.CenterTerm(''))
    lines.append(Forge.CenterTerm(''))
    lines.append(Forge.CenterTerm(lux['Ash'] + str(label or '') + lux['Reset']))
    if subtitle:
        lines.append(Forge.CenterTerm(''))
        lines.append(Forge.CenterTerm(lux['Ash'] + str(subtitle or '') + lux['Reset']))
    else:
        lines.append(Forge.CenterTerm(''))
    if showfield or str(value or ''):
        lines.extend([Forge.CenterTerm(''), Forge.CenterTerm(Forge.ValueField(lux, value)), Forge.CenterTerm('')])
    else:
        lines.extend([Forge.CenterTerm(''), Forge.CenterTerm(''), Forge.CenterTerm('')])
    return lines


def EnterPortcullis(cache: Cache, label: str, value: str = '', subtitle: str = '', *, titletext: str = 'BYZANTIUM', showfield: bool = False) -> str:
    return Forge.FrameTextScreen(BuildPortcullis(cache, label, value=value, subtitle=subtitle, titletext=titletext, showfield=showfield))


def Left(text: str, width: int) -> str:
    return str(text or '').rjust(int(width or 0))


def Right(text: str, width: int) -> str:
    return str(text or '').ljust(int(width or 0))


def Ellipsis(cache: Cache, width: int = 3) -> str:
    frame = int(getattr(cache, 'waitingframe', 0) or 0)
    phase = (frame // max(1, WaitingDots)) % (int(width) + 1)
    return '.' * phase


def EllipsisLeft(cache: Cache, width: int = 3) -> str:
    return Left(Ellipsis(cache, width), width)


def EllipsisRight(cache: Cache, width: int = 3) -> str:
    return Right(Ellipsis(cache, width), width)


def Excitement(cache: Cache, width: int = 3) -> str:
    frame = int(getattr(cache, 'waitingframe', 0) or 0)
    phase = (frame // max(1, WaitingDots)) % (int(width) + 1)
    return Right('!' * phase, width)


def PortcullisEngaged(cache: Cache) -> str:
    EnsureState(cache)
    fields = [
        ('Choose Your Arena', 'mode', 'Campaign'),
        ('Which Gateway', 'gate', '9000'),
        ('Skeleton Key', 'skeleton', 'Skeleton'),
        ('Who Are You', 'soul', ''),
        ('Tell Me A Secret', 'secret', 'Password'),
        ('How Many Souls', 'genesis', '1'),
    ]
    index = max(0, min(len(fields) - 1, int(getattr(cache, 'titleselect', 0) or 0)))
    label, key, default = fields[index]
    rawvalue = FieldValue(cache, key)
    value = rawvalue or default
    if key in ('soul', 'skeleton', 'secret', 'gate') and not str(rawvalue or ''):
        return EnterPortcullis(cache, label, value='', showfield=True)
    return EnterPortcullis(cache, label, value=value)


def PortcullisHold(cache: Cache) -> str:
    return EnterPortcullis(cache, CollectingSouls(cache))


def PortcullisJammed(cache: Cache) -> str:
    return EnterPortcullis(cache, 'Open..Sesame!!', subtitle=f'{EllipsisLeft(cache)}This Gate Is Jammed{EllipsisRight(cache)}')


def PortcullisExit(cache: Cache) -> str:
    return EnterPortcullis(cache, f"{EllipsisLeft(cache)}I'll Hold The State{EllipsisRight(cache)}", subtitle=f"Maybe Go Touch Some Grass Now")


def PortcullisVictory(cache: Cache) -> str:
    return EnterPortcullis(cache, f'Wow{Excitement(cache)}', subtitle='Did You Do It??')


def CollectingSouls(cache: Cache) -> str:
    return f'{EllipsisLeft(cache)}Collecting Souls{EllipsisRight(cache)}'


def DrainStdin(filedescriptor: int) -> List[str]:
    chunks: List[str] = []
    while True:
        ready, write, error = select([filedescriptor], [], [], 0)
        if not ready:
            break
        try:
            block = os.read(filedescriptor, 4096)
        except BlockingIOError:
            break
        if not block:
            break
        chunks.append(block.decode('latin1', 'ignore'))
    return chunks


def HandlePortcullis(gatewayvalue: Gateway, cache: Cache, statevalue: object, kind: str, value: Optional[str]):
    EnsureState(cache)
    field = PortcullisField(cache)

    def StepGenesis(delta: int) -> None:
        raw = FieldValue(cache, 'genesis')
        current = SafeInt(raw, 1) if raw else 1
        SetFieldValue(cache, 'genesis', str(max(1, min(24, current + int(delta)))))
        cache.cursorposition = len(FieldValue(cache, 'genesis'))

    def StepGate(delta: int) -> None:
        raw = FieldValue(cache, 'gate')
        current = SafeInt(raw, int(DefaultGate)) if raw else int(DefaultGate)
        nextvalue = current + int(delta)
        if nextvalue < 0:
            nextvalue = 0
        if nextvalue > 65535:
            nextvalue = 65535
        SetFieldValue(cache, 'gate', str(nextvalue))
        cache.cursorposition = len(FieldValue(cache, 'gate'))

    if kind == 'Enter':
        if PortcullisField(cache) == 'mode' and FieldValue(cache, 'mode') == 'Exit':
            cache.exit = True
            return (cache, statevalue, False)
        if PortcullisField(cache) == 'genesis':
            return (cache, gatewayvalue.Boot(), False)
        ShiftField(cache, +1)
        return (cache, statevalue, False)
    if kind in ('Left', 'Right'):
        ShiftField(cache, -1 if kind == 'Left' else +1)
        return (cache, statevalue, False)
    if kind in ('Up', 'Down', 'Arrow', 'PortBump'):
        arrow = value
        if kind == 'Up':
            arrow = 'A'
        elif kind == 'Down':
            arrow = 'B'
        if kind in ('Arrow', 'PortBump') and arrow in ('C', 'D'):
            ShiftField(cache, +1 if arrow == 'C' else -1)
            return (cache, statevalue, False)
        delta = +1 if arrow == 'A' else -1 if arrow == 'B' else 0
        if field == 'mode' and delta:
            ToggleMode(cache, -delta)
            return (cache, statevalue, False)
        if field == 'genesis' and delta:
            StepGenesis(delta)
            return (cache, statevalue, False)
        if field == 'gate' and delta:
            step = 100 if kind == 'PortBump' else 1
            StepGate(step if delta > 0 else -step)
            return (cache, statevalue, False)
    if kind == 'Backspace':
        Backspace(cache)
        return (cache, statevalue, False)
    if kind == 'Character' and value and value.isprintable() and (value != '\t'):
        Insert(cache, value)
        return (cache, statevalue, False)
    return (cache, statevalue, False)


def main() -> None:
    gatewayvalue = Gateway()
    filedescriptor = sys.stdin.fileno()
    basicterminal = termios.tcgetattr(filedescriptor)
    tty.setcbreak(filedescriptor)
    statevalue = gatewayvalue.Runtime
    buffer = ''
    try:
        while True:
            chunks = DrainStdin(filedescriptor)
            if chunks:
                buffer += ''.join(chunks)
                tokens, buffer = InputBuffer(buffer)
                for token in tokens:
                    gatewayvalue.Cache, statevalue, shouldquit = gatewayvalue.Dispatch(statevalue, token)
                    if shouldquit:
                        sys.stdout.write('\x1b[H\x1b[2J\x1b[0m')
                        sys.stdout.flush()
                        return
            try:
                statevalue = gatewayvalue.RenderState(statevalue)
            except Exception as exceptionvalue:
                frame = EnterPortcullis(getattr(gatewayvalue, 'Cache', Cache(feed=[], name='')), 'RenderState', subtitle=str(exceptionvalue)[:60])
                sys.stdout.write('\x1b[H\x1b[2J')
                sys.stdout.write(frame)
                sys.stdout.flush()
                time.sleep(1.0 / Fps)
                continue
            try:
                frame = gatewayvalue.Render(statevalue)
            except Exception as exceptionvalue:
                frame = EnterPortcullis(getattr(gatewayvalue, 'Cache', Cache(feed=[], name='')), 'Gateway Render', subtitle=str(exceptionvalue)[:60])
            if frame:
                sys.stdout.write('\x1b[H\x1b[2J')
                sys.stdout.write(frame)
                sys.stdout.flush()
            time.sleep(1.0 / Fps)
    finally:
        termios.tcsetattr(filedescriptor, termios.TCSADRAIN, basicterminal)


if __name__ == '__main__':
    main()
