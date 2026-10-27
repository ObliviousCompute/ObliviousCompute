from __future__ import annotations
import multiprocessing as mp
import queue
import re
import signal
import socket
import sys
from dataclasses import dataclass
from typing import Any

from .Vertex import Words, Role, Polarity, WordOf, Projection, Pair, Prime as PrimeShape, Vertex, Tessellate

def RunObserver(index: int, pipe, events) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    sock.settimeout(.009)
    pipe.send(sock.getsockname()[1])
    peers = pipe.recv()

    stage = ''
    genesis = {}
    isReceiver = False
    prime = None
    charge = ''
    q = None
    phase = ''
    projection = ''
    pair = None
    primeSeen = 0
    voltSeen = 0

    def Event(kind: str, **kw) -> None:
        events.put(dict(actor=index, kind=kind, q=q, charge=charge,
                        phase=phase, projection=projection, stage=stage, **kw))

    def Emit(payload: str, reason: str) -> None:
        for port in peers:
            sock.sendto(f'{stage}/{payload}'.encode(), ('127.0.0.1', port))
        Event('send', payload=payload, reason=reason)

    def Trigger(kind: str) -> bool:
        nonlocal primeSeen, voltSeen
        if kind == 'prime':
            primeSeen = min(2, primeSeen + 1)
            return primeSeen == 2
        voltSeen = min(2, voltSeen + 1)
        return voltSeen == 2

    def Prime(newStage: str, word: str) -> None:
        nonlocal stage, isReceiver, prime, charge, q, phase, projection
        nonlocal pair, primeSeen, voltSeen
        assert genesis[newStage][0] == 'receiver'
        stage = newStage
        isReceiver = True
        prime = PrimeShape(word)
        charge = Polarity(word)
        q = word
        phase = 'prime'
        projection = prime
        pair = None
        primeSeen = voltSeen = 0
        Event('configured', receiver=True, role=Role(word))
        Event('initial', role=Role(word))
        Emit(prime, 'Prime exposure')

    def Helix(newStage: str, incomingPrime: str) -> None:
        nonlocal stage, isReceiver, prime, charge, q, phase, projection
        nonlocal pair, primeSeen, voltSeen
        assert genesis[newStage][0] == 'helix'
        stage = newStage
        isReceiver = False
        prime = None
        charge = genesis[newStage][1]
        q = None
        phase = 'prime'
        projection = incomingPrime
        pair = None
        primeSeen = voltSeen = 0
        Event('configured', receiver=False, role='helix')
        Event('initial', role='helix')
        Trigger('prime')
        Emit(incomingPrime, 'Prime relay')

    # ================= LINCHPIN ================= #
    def Transist(word: str) -> None:
        current = stage
        step = Stages.index(current)
        Event('transist', word=word, next=Stages[step + 1] if step + 1 < 3 else 'Ω')
        if step + 1 < 3:
            Prime(Stages[step + 1], word)
    # ============================================ #

    def Completed(evidence: str) -> None:
        nonlocal q, projection, phase
        assert isReceiver and prime is not None
        resolved = Tessellate(prime, evidence)
        if projection == resolved:
            return
        before = q
        q = WordOf(resolved)
        phase = 'complete'
        projection = resolved
        Event('candidate', word=q, origin=Vertex(prime), before=before,
              after=projection, reason='Tessellate Prime and Helix evidence')
        Trigger('volt')
        Emit(resolved, 'completed Volt broadcast')

    def SeeVolt(word: str) -> None:
        nonlocal q, projection, phase
        resolved = Projection(word)
        if WordOf(projection) is None:
            before = q
            q = word
            phase = 'complete'
            projection = resolved
            Trigger('volt')
            Event('observed', word=word, before=before)
            Emit(resolved, 'received Volt echo')
            return
        if projection != resolved:
            Event('conflict', local=q, other=word, reason='distinct completed Volt')
            return
        if voltSeen < 2 and Trigger('volt'):
            step = Stages.index(stage)
            nextReceiver = (step + 1 < 3 and genesis[Stages[step + 1]][0] == 'receiver')
            if nextReceiver or (step == 2 and isReceiver):
                Transist(word)
            else:
                Event('confirmed', word=word)

    Event('ready', role='observer')
    while True:
        if pipe.poll() and pipe.recv() == 'stop':
            break
        try:
            data = sock.recvfrom(128)[0]
        except socket.timeout:
            continue
        try:
            payload = data.decode('utf-8')
        except UnicodeError:
            Event('reject', reason='invalid UTF-8 projection')
            continue
        if payload.startswith('G/'):
            parsed = {}
            for section in payload[2:].split('|'):
                match = re.fullmatch(r'([ΔΘΣ])/([+-])([ABC])\(([ABC])\)([ABC])([+-])', section)
                if not match:
                    break
                step, lc, left, receiver, right, rc = match.groups()
                if set((left, receiver, right)) != set('ABC'):
                    break
                who = 'ABC'[index]
                parsed[step] = ('receiver', '') if who == receiver else (
                    ('helix', lc) if who == left else ('helix', rc))
            if set(parsed) == set(Stages):
                genesis = parsed
                stage = ''
                Event('genesis', board=payload, sockets=Stages)
            else:
                Event('reject', payload=payload, reason='invalid complete Genesis')
            continue
        incomingStage, sep, content = payload.partition('/')
        if not sep or incomingStage not in genesis:
            continue
        if incomingStage != stage:
            expected = 'Δ' if not stage else (
                Stages[Stages.index(stage) + 1] if stage in Stages[:2] else '')
            if incomingStage != expected or content not in PrimeShapes:
                continue
            if genesis[incomingStage][0] == 'receiver':
                if not stage:
                    Prime(incomingStage, Vertex(content))
            else:
                Helix(incomingStage, content)
            continue
        Event('receive', payload=content)
        if content in PrimeShapes:
            if not isReceiver and primeSeen < 2 and Trigger('prime'):
                phase = 'helix'
                projection = charge
                Emit(charge, 'bare Helix charge')
            continue
        if len(content) == 1 and content in '+-' and not isReceiver and pair is None:
            pair = Pair(charge, content)
            Emit(pair, 'Helixes compose double-charge evidence')
            continue
        if content in ('++', '--', '+-', '-+') and isReceiver and phase == 'prime':
            Completed('+-' if content in ('+-', '-+') else content)
            continue
        word = WordOf(content)
        if word is not None:
            SeeVolt(word)
    sock.close()
    pipe.close()

Stages = 'ΔΘΣ'
PrimeShapes = frozenset(PrimeShape(q) for q in Words)

@dataclass
class Session:
    word: str
    loads: str

    def __post_init__(self):
        if self.word not in Words:
            raise ValueError('input must be 000..111')
        if len(self.loads) != 2 or any(c not in '+-' for c in self.loads):
            raise ValueError('helix orientations must be ++,+-,-+,--')
        self.ctx = mp.get_context('fork' if sys.platform.startswith('linux') else 'spawn')
        self.events = self.ctx.Queue()
        self.pipes = []
        self.procs = []
        self.running = False
        self.board = None

    def Start(self) -> None:
        if self.running:
            return
        for index in range(3):
            parent, child = self.ctx.Pipe()
            p = self.ctx.Process(target=RunObserver, args=(index, child, self.events), daemon=True)
            p.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(p)
        self.ports = [pipe.recv() for pipe in self.pipes]
        for i, pipe in enumerate(self.pipes):
            pipe.send([port for j, port in enumerate(self.ports) if j != i])
        self.running = True

    def PublishGenesis(self, board: dict) -> None:
        if not self.running:
            raise RuntimeError('session not started')
        self.board = board
        payload = 'G/' + '|'.join(board[stage].Serialize(stage) for stage in sorted(board))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as injector:
            for port in self.ports:
                injector.sendto(payload.encode('utf-8'), ('127.0.0.1', port))

    def InjectWord(self, stage: str, word: str) -> None:
        if not self.running or self.board is None:
            raise RuntimeError('Genesis must configure the socket first')
        receiver = self.board[stage].receiver
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as injector:
            for i, port in enumerate(self.ports):
                if 'ABC'[i] == receiver:
                    injector.sendto(f'{stage}/{PrimeShape(word)}'.encode(), ('127.0.0.1', port))

    def Poll(self) -> list[dict[str, Any]]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def Stop(self) -> None:
        if not self.running:
            return
        for pipe in self.pipes:
            try:
                pipe.send('stop')
            except (OSError, BrokenPipeError):
                pass
        for p in self.procs:
            p.join(.07)
            if p.is_alive():
                p.terminate()
                p.join(.07)
        for pipe in self.pipes:
            pipe.close()
        self.events.cancel_join_thread()
        self.events.close()
        self.running = False
