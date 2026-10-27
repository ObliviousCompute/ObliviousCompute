from __future__ import annotations
from dataclasses import dataclass
import time

from .Observer import Session

Actors = 'ABC'
Stages = 'ΔΘΣ'

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
    def __init__(self, board=None, inputWord='101', Log=None):
        if Log is None:
            from .Scan import Log as SpectatorLog
            Log = SpectatorLog()
        self.Log = Log
        self.board = board if board is not None else DefaultBoard()
        self.inputWord = inputWord
        self.session: Session | None = None
        self.stageIndex = -1
        self.mode = 'idle'
        self.output = dict.fromkeys(Stages)
        self.injected = dict.fromkeys(Stages)
        self.runStartedAt = None
        self.runFinishedAt = None
        self.deferred = []
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
        self.runStartedAt = self.runFinishedAt = None
        self.Log.Invalidate()

    def Begin(self):
        self.output = dict.fromkeys(Stages)
        self.injected = dict.fromkeys(Stages)
        self.dirty = False
        self.mode = 'genesis'
        self.deferred.clear()
        self.stageIndex = -1
        self.runStartedAt = time.monotonic()
        self.runFinishedAt = None
        self.Log.Begin(self.board)
        if self.session is None:
            self.session = Session(self.inputWord, self.board[Stages[0]].Loads())
            self.session.Start()
        else:
            self.session.Poll()
        self.session.PublishGenesis(self.board)

    def NextStage(self, word: str):
        self.stageIndex += 1
        if self.stageIndex == len(Stages):
            self.mode = 'complete'
            self.runFinishedAt = time.monotonic()
            self.Log.Complete(word)
            return
        stage = Stages[self.stageIndex]
        self.mode = 'running'
        self.Log.OpenStage(stage, self.board[stage], word)
        if self.stageIndex == 0:
            assert self.session is not None
            self.session.InjectWord(stage, word)

    def Poll(self):
        if not self.session or self.mode not in ('genesis', 'running', 'complete'):
            return
        events = self.session.Poll()
        self.Log.RecordEvents(events)
        if self.mode == 'genesis':
            for event in events:
                self.Log.GenesisEvent(event)
            if self.Log.GenesisReady():
                self.Log.GenesisComplete()
                self.NextStage(self.inputWord)
        elif self.mode == 'running':
            self.deferred.extend(events)
            waiting = []
            for event in self.deferred:
                if self.mode == 'complete':
                    break
                stage = Stages[self.stageIndex]
                if event['stage'] == stage:
                    if (event['kind'] == 'initial' and
                            event['actor'] == Actors.index(self.board[stage].receiver)):
                        self.injected[stage] = event['charge']
                    if event['kind'] == 'transist':
                        word = event['word']
                        self.output[stage] = word
                        self.Log.Resolved(stage, self.board[stage], word)
                        self.NextStage(word)
                elif event['stage'] in Stages[self.stageIndex + 1:]:
                    waiting.append(event)
            self.deferred = waiting
        self.Log.Refresh()

    def Elapsed(self) -> float | None:
        if self.runStartedAt is None:
            return None
        return (self.runFinishedAt if self.runFinishedAt is not None
                else time.monotonic()) - self.runStartedAt
