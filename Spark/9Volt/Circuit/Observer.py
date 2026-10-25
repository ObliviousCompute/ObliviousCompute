from __future__ import annotations
import multiprocessing as mp
import queue
import re
import signal
import socket
import sys
import time
from dataclasses import dataclass
from typing import Any

from .Vertex import Words, Role, Polarity, WordOf, Projection
from .Geometry import Pair, Prime, Vertex, Tessellate


def RunObserver(index: int, pipe, events) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    sock.settimeout(.009)
    pipe.send(sock.getsockname()[1])
    peers = pipe.recv()

    stage = ''
    genesis = {}
    receiver = False
    prime = None
    charge = ''
    q = None
    projection = ''
    pair = None
    confirmed = False
    conflict = False
    challenged = None
    canOverride = False
    pending = None
    lastPulse = 0.0
    echoes = 0

    def Event(kind, **details):
        events.put(dict(actor=index, kind=kind, stage=stage, q=q, charge=charge,
                        phase='complete' if WordOf(projection) else 'prime' if receiver else 'helix',
                        projection=projection, **details))

    def Emit(value, reason):
        for port in peers:
            sock.sendto(f'{stage}/{value}'.encode(), ('127.0.0.1', port))
        Event('send', payload=value, reason=reason)

    def Complete(word, kind):
        nonlocal q, projection, confirmed, challenged, echoes, canOverride
        before = q
        q = word
        projection = Projection(word)
        confirmed = False
        challenged = None
        canOverride = False
        echoes = 0
        if kind == 'candidate':
            Event(kind, word=word, origin=Vertex(prime), before=before,
                  after=projection, reason='Tessellate Prime and Helix evidence')
        else:
            Event(kind, word=word, before=before)
        Emit(projection, 'completed Volt broadcast' if receiver else 'received Volt echo')

    def Confirm(word):
        nonlocal confirmed, echoes
        if not confirmed and not conflict:
            confirmed = True
            echoes = 4
            Event('confirmed', word=word)

    def HearVolt(word):
        nonlocal pending, confirmed, challenged
        if receiver and WordOf(projection) is None:
            return
        if not receiver and pair is None:
            pending = word
            return
        if not receiver and (WordOf(projection) is None or canOverride):
            Complete(word, 'observed')
        elif WordOf(projection) == word:
            Confirm(word)
        elif challenged != word:
            confirmed = False
            challenged = word
            Event('conflict', local=q, other=word,
                  reason='conflicting completed Volt without opposing evidence')
            Emit(projection, 'conflicting Volt reply')

    def HearPair(evidence):
        nonlocal pair, conflict, confirmed, canOverride, pending
        evidence = Pair(*evidence)
        if pair == evidence:
            return
        if receiver:
            if pair is None or evidence == '+-':
                if pair is not None:
                    Event('override', before=pair, after=evidence,
                          reason='opposing evidence trumps')
                pair = evidence
                conflict = False
                # ================= LINCHPIN ================= #
                Complete(WordOf(Tessellate(prime, evidence)), 'candidate')
                # ============================================ #
            elif pair != '+-':
                conflict = True
                confirmed = False
                Event('conflict', local=q, other=WordOf(Tessellate(prime, evidence)),
                      reason='incompatible equal-rank Helix evidence')
        elif evidence == '+-' and pair != '+-':
            pair = evidence
            canOverride = True
            confirmed = False
            Event('override', before=projection, after=evidence,
                  reason='opposing evidence trumps')
            if challenged is not None:
                HearVolt(challenged)

    def Adopt(newStage, incomingPrime, role, helix):
        nonlocal stage, receiver, prime, charge, q, projection, pair
        nonlocal confirmed, conflict, challenged, canOverride, pending, lastPulse, echoes
        stage = newStage
        receiver = role == 'receiver'
        prime = incomingPrime if receiver else None
        charge = Polarity(Vertex(prime)) if receiver else helix
        q = Vertex(prime) if receiver else None
        projection = prime if receiver else charge
        pair = None
        confirmed = conflict = canOverride = False
        challenged = pending = None
        echoes = 0
        lastPulse = time.monotonic()
        Event('configured', receiver=receiver, role=Role(q) if receiver else 'helix')
        Event('initial', role=Role(q) if receiver else 'helix')
        Emit(projection, 'Prime exposure' if receiver else 'bare Helix charge')

    def Retry():
        nonlocal echoes, lastPulse
        if not stage or time.monotonic() - lastPulse < .035:
            return
        lastPulse = time.monotonic()
        if confirmed:
            if not echoes:
                return
            echoes -= 1
        if not receiver:
            Emit(charge, 'bare Helix exposure')
            if pair:
                Emit(pair, 'paired Helix evidence')
        Emit(projection, 'completed Volt echo' if WordOf(projection) else 'projection retry')

    Event('ready', role='observer')
    while True:
        if pipe.poll() and pipe.recv() == 'stop':
            break
        try:
            data, _ = sock.recvfrom(128)
        except socket.timeout:
            Retry()
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
                step, lc, left, middle, right, rc = match.groups()
                if set((left, middle, right)) != set('ABC'):
                    break
                who = 'ABC'[index]
                parsed[step] = ('receiver', '') if who == middle else (
                    ('helix', lc) if who == left else ('helix', rc))
            if len(parsed) == 3:
                genesis = parsed
                Event('genesis', board=payload, sockets=''.join(sorted(parsed)))
            else:
                Event('reject', payload=payload, reason='invalid Genesis')
            continue
        prefix, separator, content = payload.partition('/')
        if not separator or prefix not in genesis:
            continue
        if prefix != stage:
            role, helix = genesis[prefix]
            if role == 'receiver' and content in PrimeShapes:
                Adopt(prefix, content, role, helix)
            elif role == 'helix' and content == '?':
                Adopt(prefix, None, role, helix)
            continue
        Event('receive', payload=content)
        if len(content) == 1 and content in '+-' and not receiver:
            evidence = Pair(charge, content)
            if pair is None:
                pair = evidence
                Emit(pair, 'Helixes compose double-charge evidence')
                if pending is not None:
                    word, pending = pending, None
                    HearVolt(word)
            elif pair != evidence:
                Event('conflict', local=q, other=None,
                      reason='incompatible bare Helix evidence')
        elif content in ('++', '--', '+-', '-+'):
            HearPair(content)
        elif (word := WordOf(content)) is not None:
            HearVolt(word)
        elif content not in (prime, charge, '?', '+', '-') and content not in PrimeShapes:
            Event('open', payload=content, reason='unrecognized packet shape')
    sock.close()
    pipe.close()

PrimeShapes = frozenset(Prime(q) for q in Words)


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
                shape = Prime(word) if 'ABC'[i] == receiver else '?'
                injector.sendto(f'{stage}/{shape}'.encode('utf-8'), ('127.0.0.1', port))

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
