from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import time
import unicodedata

from .Observer import Session
from .Vertex import Words, WordOf, RoleLabel, VertexLabel, LogGlyphCells

VOLT = '⚡VOLT⚡'
SPARK = '✨'

RuleText = {
    'matching bare charge': 'Matching {own} Charge walks',
    'opposing charge, unchanged': 'Opposing {other} Charge leaves state unchanged',
    'opposing lateral, capture': '{Other} lateral locks {own} Gate',
    'opposing bare charge': '{Other} Charge locks {own} Gate',
    'matching lateral, append': 'Matching {own} lateral walks',
    'Helix admits lateral': 'Helix admits {lateral} lateral',
    'walk admits lateral': 'Helix admits {lateral} lateral',
    'capture holds Helix charge': SPARK,
    'two matching laterals': 'Matching {own} laterals form pair',
    'terminal admits other lateral': '{Own} Apex admits {other} lateral',
    'terminal pair needs third': 'Matching {own} primes await lateral',
    'ordinary rail reached terminal': 'Matching {own} primes await lateral',
    'unknown shaped projection': 'Unrecognized projection',
    'terminal pings lateral': 'Captured Gate reprojects',
}

def DescriptionForRule(reason: str, charge: str, projection: str) -> str:
    template = RuleText.get(reason)
    if template is None:
        return reason
    own = 'positive' if charge == '+' else 'negative' if charge == '-' else ''
    other = 'negative' if charge == '+' else 'positive' if charge == '-' else ''
    middle = projection[1] if len(projection) == 3 and projection[1] in '+-' else charge
    lateral = 'positive' if middle == '+' else 'negative' if middle == '-' else own
    return template.format(own=own, other=other, Own=own.capitalize(),
                           Other=other.capitalize(), lateral=lateral)

def Describe(event: dict) -> str:
    kind = event.get('kind', '')
    reason = event.get('reason') or ''
    charge = event.get('charge', '')
    payload = event.get('payload') or ''
    if kind == 'configured': return event.get('q', '')
    if kind in ('ready', 'genesis', 'input'): return ''
    if kind == 'candidate':
        phase = event.get('phase', '')
        who = ('Captured Gate' if phase == 'captured' else
               ('Positive' if charge == '+' else 'Negative' if charge == '-' else '') + ' Apex'
               if phase == 'terminal' else 'Walking Helix')
        return f'{who} completes {VOLT}'
    if kind == 'observed': return f'Received {VOLT}'
    if kind == 'send': return f'Broadcast {VOLT}' if WordOf(payload) else SPARK
    if kind == 'receive': return '' if WordOf(payload) else SPARK
    if kind == 'initial': return SPARK
    if kind == 'lock':
        return f'{"Positive" if charge == "+" else "Negative" if charge == "-" else ""} Apex locks'
    if kind == 'reject': return reason
    if kind == 'conflict': return 'Conflicting completed VOLT'
    if kind in ('append', 'jump', 'walk', 'open') or reason:
        return DescriptionForRule(reason, charge, event.get('projection', ''))
    return ''

Actors = 'ABC'
Stages = 'ΔΘΣ'
Bar = '─' * 76
HeavyBar = '=' * 76
TRACE_LEFT = 1
TRACE_SOCKET_WIDTH = 3
TRACE_ACTION_WIDTH = len('configured')
TRACE_LABEL_WIDTH = 8
TRACE_SOCKET_X = TRACE_LEFT
TRACE_ACTION_X = TRACE_SOCKET_X + TRACE_SOCKET_WIDTH + 1
TRACE_LABEL_X = TRACE_ACTION_X + TRACE_ACTION_WIDTH + 1
TRACE_INFO_X = TRACE_LABEL_X + TRACE_LABEL_WIDTH + 1

def PairedGlyphSegments(detail: str) -> tuple[tuple[tuple[int, str], ...], int] | None:
    for glyph, cells in LogGlyphCells.items():
        if detail.startswith(glyph) and detail.endswith(glyph) and len(detail) > 2 * len(glyph):
            core = detail[len(glyph):-len(glyph)]
            core_width = DisplayWidth(core)
            return (((0, glyph), (cells, core), (cells + core_width, glyph)),
                    2 * cells + core_width)
    return None
def LabelSegments(detail: str) -> tuple[tuple[int, str], ...]:
    if not detail:
        return ()
    pair = PairedGlyphSegments(detail)
    if pair is not None:
        segments, total = pair
        if total <= TRACE_LABEL_WIDTH:
            left = (TRACE_LABEL_WIDTH - total + 1) // 2
            return tuple((left + offset, token) for offset, token in segments)
    cells = DisplayWidth(detail)
    return ((max(0, (TRACE_LABEL_WIDTH - cells + 1) // 2), detail),)
def LabelField(detail: str) -> str:
    result, position = '', 0
    for start, token in LabelSegments(detail):
        result += ' ' * max(0, start - position) + token
        position = start + DisplayWidth(token)
    return result + ' ' * max(0, TRACE_LABEL_WIDTH - position)
def DisplayWidth(value: str) -> int:
    return sum(0 if unicodedata.combining(char) or char in '\ufe0e\ufe0f' else
               2 if unicodedata.east_asian_width(char) in 'WF' or
               (index + 1 < len(value) and value[index + 1] == '\ufe0f') else 1
               for index, char in enumerate(value))

@dataclass(frozen=True)
class TraceRow:
    stage: str
    actor: str
    kind: str
    detail: str = ''
    info: str = ''

    def __str__(self) -> str:
        if not self.stage:
            return ' ' * (TRACE_INFO_X - TRACE_LEFT) + self.info
        socket = f'{self.stage}:{self.actor}'
        return (socket.ljust(TRACE_SOCKET_WIDTH) + ' ' +
                self.kind.ljust(TRACE_ACTION_WIDTH) + ' ' +
                LabelField(self.detail) + ' ' + self.info).rstrip()

@dataclass
class SocketSpec:
    left: str
    receiver: str
    right: str
    leftCharge: str = '+'
    rightCharge: str = '-'

    def __post_init__(self):
        if sorted((self.left, self.receiver, self.right)) != list(Actors):
            raise ValueError('each socket needs all three observers exactly once')
        if self.leftCharge not in '+-' or self.rightCharge not in '+-':
            raise ValueError('invalid Helix polarity')

    def Serialize(self, stage: str) -> str:
        return f'{stage}/{self.leftCharge}{self.left}({self.receiver}){self.right}{self.rightCharge}'
    def Loads(self) -> str:
        return self.leftCharge + self.rightCharge
    def CycleReceiver(self, delta: int) -> None:
        oldIndex = Actors.index(self.receiver)
        direction = 1 if self.left == Actors[(oldIndex + 1) % 3] else -1
        newIndex = (oldIndex + delta) % 3
        self.receiver = Actors[newIndex]
        self.left = Actors[(newIndex + direction) % 3]
        self.right = Actors[(newIndex - direction) % 3]
    def FlipHelix(self, side: int, direction: int) -> None:
        if side not in (0, 2):
            raise ValueError(side)
        charge = self.leftCharge if side == 0 else self.rightCharge
        pos = (0 if charge == '+' else 1) + (0 if (self.left < self.right) else 2)
        pos = (pos + direction) % 4
        left, right = sorted((self.left, self.right))
        if pos >= 2:
            left, right = right, left
        self.left, self.right = left, right
        chosenCharge = '+' if pos % 2 == 0 else '-'
        if side == 0:
            self.leftCharge = chosenCharge
        else:
            self.rightCharge = chosenCharge

def DefaultBoard() -> dict[str, SocketSpec]:
    return {'Δ': SocketSpec('A', 'C', 'B', '+', '-'),
            'Θ': SocketSpec('B', 'A', 'C', '-', '+'),
            'Σ': SocketSpec('C', 'B', 'A', '+', '-')}

class Circuit:
    def __init__(self, board=None, inputWord='101'):
        self.board = board if board is not None else DefaultBoard()
        self.inputWord = inputWord
        self.session: Session | None = None
        self.stageIndex = -1
        self.mode = 'idle'
        self.output = dict.fromkeys(Stages)
        self.injected = dict.fromkeys(Stages)
        self.stageEvents: list[dict] = []
        self.log = deque(maxlen=1500)
        self.candidates = set()
        self.completions = set()
        self.runStartedAt = None
        self.runFinishedAt = None
        self.confirmed = set()
        self.configuredEntries = {}
        self.stagedLines = []
        self.dirty = False
    def Close(self):
        if self.session:
            self.session.Stop()
            self.session = None
    def Invalidate(self):
        self.output = dict.fromkeys(Stages)
        self.injected = dict.fromkeys(Stages)
        self.dirty = True
        self.mode = 'idle'
        self.stageIndex = -1
        self.runStartedAt = None
        self.runFinishedAt = None
        self.log.append('EDIT: output cleared; awaiting next complete Genesis')

    def Begin(self):
        self.output = dict.fromkeys(Stages)
        self.injected = dict.fromkeys(Stages)
        self.log.clear()
        self.log.append(HeavyBar)
        self.dirty = False
        self.mode = 'genesis'
        self.stageIndex = -1
        self.runStartedAt = time.monotonic()
        self.runFinishedAt = None
        self.log.append('GENESIS ' + ' | '.join(f'{s}:{self.board[s].Serialize(s)[2:]}' for s in Stages))
        if self.session is None:
            self.session = Session(self.inputWord, self.board[Stages[0]].Loads())
            self.session.Start()
        else:
            self.session.Poll()
        self.genesisSeen = set()
        self.session.PublishGenesis(self.board)

    def NextStage(self, word: str):
        self.stageIndex += 1
        if self.stageIndex == 3:
            self.mode = 'complete'
            self.runFinishedAt = time.monotonic()
            self.log.append(f'Ω:{VertexLabel(word)} FINAL OUTPUT')
            return
        stage = Stages[self.stageIndex]
        cfg = self.board[stage]
        self.candidates = set()
        self.completions = set()
        self.stageEvents = []
        self.confirmed = set()
        self.configuredEntries = {}
        self.stagedLines = []
        self.mode = 'running'
        self.log.append(f'{stage}: {cfg.leftCharge}{cfg.left}({cfg.receiver}){cfg.right}{cfg.rightCharge} Receiver ({cfg.receiver}) Helix ({cfg.left} | {cfg.right}) Input {VertexLabel(word)}')
        assert self.session is not None
        self.session.InjectWord(stage, word)

    def Ingest(self, eventData: dict):
        stage = Stages[self.stageIndex] if 0 <= self.stageIndex < 3 else '?'
        if eventData.get('stage') != stage:
            return
        self.stageEvents.append(eventData)
        actor = Actors[eventData['actor']]
        kind = eventData['kind']
        info = Describe(eventData)
        payload = eventData.get('payload') or ''
        if kind == 'configured':
            detail = RoleLabel(eventData['q'])
            info = eventData['q']
        elif kind in ('initial', 'lock'):
            detail = VertexLabel(eventData['q'])
        elif kind == 'input':
            detail = VertexLabel(eventData['word'])
        else:
            detail = (payload or eventData.get('after') or
                      eventData.get('word') or eventData.get('role') or '')
            if detail in Words:
                detail = VertexLabel(detail)
            elif WordOf(detail):
                detail = VertexLabel(WordOf(detail))
        line = TraceRow(stage, actor, 'Lock' if kind in ('jump', 'lock') else kind, detail, info)
        if kind == 'configured':
            self.configuredEntries[eventData['actor']] = line
            if len(self.configuredEntries) == 3:
                self.log.extend(self.configuredEntries[index] for index in range(3))
                self.log.append(Bar)
                self.log.extend(self.stagedLines)
                self.stagedLines.clear()
        else:
            destination = self.stagedLines if len(self.configuredEntries) < 3 else self.log
            destination.append(line)
            if kind == 'observed':
                destination.append(TraceRow(stage, actor, 'Quiet...'))
        if kind == 'initial' and eventData['actor'] == Actors.index(self.board[stage].receiver):
            self.injected[stage] = eventData['charge']
        if kind in ('candidate', 'observed'):
            if kind == 'candidate':
                self.candidates.add(eventData['word'])
            self.completions.add(eventData['word'])
            self.confirmed.add(eventData['actor'])
        if kind == 'conflict':
            self.completions.update((eventData['local'], eventData['other']))

    def CompleteStage(self, word: str):
        stage = Stages[self.stageIndex]
        cfg = self.board[stage]
        participants = [cfg.receiver]
        for e in self.stageEvents:
            if e['kind'] in ('append', 'candidate'):
                actor = Actors[e['actor']]
                if actor != participants[-1]:
                    participants.append(actor)
        finalReceivers = [Actors[e['actor']] for e in self.stageEvents if e['kind'] == 'observed']
        if finalReceivers:
            participants.append(finalReceivers[-1])
        path = '→'.join(participants)
        target = Stages[self.stageIndex+1] if self.stageIndex+1 < len(Stages) else 'Ω'
        compact = f'{stage}:{cfg.Serialize(stage)[2:]} {path} ✓ {target}:{VertexLabel(word)}'
        self.log.append('')
        self.log.append('RESOLVED ' + compact + f' [Cohort {len(self.confirmed)}/3]')
        self.log.append(HeavyBar)
        self.output[stage] = word
        self.NextStage(word)

    def Poll(self):
        if not self.session or self.mode not in ('genesis', 'running'):
            return
        newEvents = self.session.Poll()
        if self.mode == 'genesis':
            for e in newEvents:
                if e['kind'] == 'genesis':
                    self.genesisSeen.add(e['actor'])
                    self.log.append(f"GENESIS:{Actors[e['actor']]} adopted complete board")
                else:
                    self.log.append(f"BOOT:{Actors[e['actor']]} {e['kind']}")
            if len(self.genesisSeen) == 3:
                self.log.append(HeavyBar)
                self.NextStage(self.inputWord)
            return
        for eventData in newEvents:
            self.Ingest(eventData)
        if len(self.completions) > 1:
            self.mode = 'conflict'
            self.log.append(f'CONFLICT at {Stages[self.stageIndex]}: {", ".join(sorted(self.completions))}')
            return
        if self.candidates and len(self.confirmed) == 3:
            self.CompleteStage(next(iter(self.candidates)))
    def Elapsed(self) -> float | None:
        if self.runStartedAt is None:
            return None
        return (self.runFinishedAt if self.runFinishedAt is not None
                else time.monotonic()) - self.runStartedAt
