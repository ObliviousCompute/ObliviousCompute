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

from .Vertex import (Words, Poles, Ordinary, Terminal, Lateral, Role,
                       Polarity, Jump, WordOf, Admission)
Valid = set('+-_')
def ValidProjection(payload: str) -> bool:
    return ((len(payload) == 1 and payload in '+-') or
            (len(payload) == 3 and set(payload) <= Valid and payload != '___'))

def RunObserver(index: int, word: str, load: str, pipe, events) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1', 0))
    sock.settimeout(.009)
    pipe.send(sock.getsockname()[1])
    peers = pipe.recv()
    isReceiver = index == 0
    wireStage = ""
    genesis = {}
    q = word if isReceiver else Poles[load]
    charge = Polarity(word) if isReceiver else load
    projection = charge
    phase = ('terminal' if q in Terminal else 'prime_lateral' if q in Lateral
             else 'walking') if isReceiver else 'helix'
    def Event(kind: str, **kw) -> None:
        events.put(dict(actor=index, kind=kind, q=q, charge=charge,
                        phase=phase, projection=projection, stage=wireStage, **kw))
    def Emit(payload: str, reason: str) -> None:
        assert ValidProjection(payload), payload
        wire = f'{wireStage}/{payload}' if wireStage else payload
        for port in peers:
            sock.sendto(wire.encode('utf-8'), ('127.0.0.1', port))
        Event('send', payload=payload, reason=reason)
    def Reproject(reason: str, *, new: bool = False) -> None:
        if WordOf(projection):
            if new:
                Emit(projection, reason)
            return
        Emit(projection, reason)
        if phase == 'terminal' and projection == charge + '_' + charge:
            Emit(charge, 'Apex Charge')
    def SetProjection(newProjection: str, reason: str) -> None:
        nonlocal projection
        assert ValidProjection(newProjection), newProjection
        if newProjection == projection:
            return
        before = projection
        projection = newProjection
        w = WordOf(projection)
        if w:
            Event('candidate', word=w, before=before, after=projection, reason=reason)
        else:
            Event('append', before=before, after=projection, reason=reason)
        Reproject(reason, new=bool(w))
    def LockTerminal() -> None:
        nonlocal phase
        phase = 'terminal'
        Event('lock', role='terminal')
        SetProjection(charge + '_' + charge, 'ordinary rail reached terminal')
    def Capture(reason: str) -> None:
        nonlocal q, phase
        old = q
        q = Jump(q)
        phase = 'captured'
        Event('jump', before=old, after=q, reason=reason,
              doorway='invert cube then one shell; original charge retained')
    def Walk(reason: str) -> None:
        nonlocal q, phase
        lane = Ordinary[charge]
        if q not in lane or lane.index(q) == len(lane)-1:
            return
        old = q
        q = lane[lane.index(q)+1]
        phase = 'walking'
        Event('walk', before=old, after=q, reason=reason)
        if q in Terminal:
            LockTerminal()
        else:
            Reproject('one-shell ordinary walk')
    def Start() -> None:
        nonlocal projection
        projection = (charge + '_' + charge if phase == 'terminal' else
                      '_' + charge + '_' if phase == 'prime_lateral' else charge)
        Event('initial', role=Role(q))
        Reproject('incoming exposure')
    def Adopt(stage: str, incomingWord: str, assignedPosition: str, assignedCharge: str) -> None:
        nonlocal wireStage, isReceiver, word, load, q, charge, projection, phase
        wireStage = stage
        isReceiver = assignedPosition == 'receiver'
        word = incomingWord
        load = assignedCharge
        q = word if isReceiver else Poles[load]
        charge = Polarity(word) if isReceiver else load
        projection = charge
        phase = (('terminal' if q in Terminal else
                  'prime_lateral' if q in Lateral else 'walking')
                 if isReceiver else 'helix')
        Event('configured', receiver=isReceiver, role=Role(q))
    Event('ready', role=Role(q))
    while True:
        if pipe.poll():
            command = pipe.recv()
            if command == 'stop':
                break
        try:
            data, _addr = sock.recvfrom(128)
        except socket.timeout:
            continue
        try:
            payload = data.decode('utf-8')
        except UnicodeError:
            Event('reject', reason='invalid UTF-8 projection')
            continue
        if payload.startswith('G/'):
            parts = payload[2:].split('|')
            parsed = {}
            ok = len(parts) >= 1
            for section in parts:
                match = re.fullmatch(r'([ΔΘΣ])/([+-])([ABC])\(([ABC])\)([ABC])([+-])', section)
                if not match:
                    ok = False
                    break
                stage, lc, left, receiverName, right, rc = match.groups()
                if set((left, receiverName, right)) != set('ABC'):
                    ok = False
                    break
                who = 'ABC'[index]
                if who == receiverName:
                    parsed[stage] = ('receiver', '')
                elif who == left:
                    parsed[stage] = ('helix', lc)
                elif who == right:
                    parsed[stage] = ('helix', rc)
            if ok and len(parsed) == len(parts):
                genesis = parsed
                wireStage = ''
                Event('genesis', board=payload, sockets=''.join(sorted(parsed)))
            else:
                Event('reject', payload=payload, reason='invalid complete Genesis')
            continue
        if genesis or wireStage:
            prefix, sep, content = payload.partition('/')
            if not sep or (prefix not in genesis if genesis else prefix != wireStage):
                continue
            if genesis and content in Words and prefix != wireStage:
                position, helixCharge = genesis[prefix]
                Adopt(prefix, content, position, helixCharge)
            if prefix != wireStage:
                continue
            if content in Words:
                if isReceiver:
                    Event('input', payload=payload, word=content)
                    if content == word:
                        Start()
                continue
            payload = content
        if not ValidProjection(payload):
            Event('reject', payload=payload, reason='invalid projection')
            continue
        time.sleep(.00035)
        Event('receive', payload=payload)
        done = WordOf(projection)
        if done:
            if WordOf(payload) and WordOf(payload) != done:
                Event('conflict', other=WordOf(payload), local=done)
            continue
        if WordOf(payload):
            projection = payload
            Event('observed', word=WordOf(payload))
            continue
        # ================= LINCHPIN ================= #
        decisions = Admission(phase, charge, projection, isReceiver, payload)
        # ================= LINCHPIN ================= #
        for instruction, value, reason in decisions:
            if instruction == 'walk':
                Walk(reason)
            elif instruction == 'capture':
                Capture(reason)
            elif instruction == 'set':
                SetProjection(value, reason)
            elif instruction == 'reproject':
                Reproject(reason)
            elif instruction == 'open':
                Event('open', payload=value, reason=reason)
    sock.close()
    pipe.close()

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

    def Start(self) -> None:
        if self.running:
            return
        for index in range(3):
            parent, child = self.ctx.Pipe()
            p = self.ctx.Process(target=RunObserver,
                                 args=(index,self.word,self.loads[index-1] if index else '',
                                       child,self.events), daemon=True)
            p.start()
            child.close()
            self.pipes.append(parent)
            self.procs.append(p)
        ports = [pipe.recv() for pipe in self.pipes]
        self.ports = ports
        for i, pipe in enumerate(self.pipes):
            pipe.send([port for j,port in enumerate(ports) if j != i])
        self.running = True
    def PublishGenesis(self, board: dict) -> None:
        if not self.running:
            raise RuntimeError('session not started')
        payload = 'G/' + '|'.join(board[stage].Serialize(stage) for stage in sorted(board))
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as injector:
            for port in self.ports:
                injector.sendto(payload.encode('utf-8'), ('127.0.0.1', port))
    def InjectWord(self, stage: str, word: str) -> None:
        if not self.running:
            raise RuntimeError('session not started')
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as injector:
            for port in self.ports:
                injector.sendto(f'{stage}/{word}'.encode('utf-8'), ('127.0.0.1', port))
    def Poll(self) -> list[dict[str,Any]]:
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
            try: pipe.send('stop')
            except (OSError,BrokenPipeError): pass
        for p in self.procs:
            p.join(.07)
            if p.is_alive():
                p.terminate()
                p.join(.07)
        for pipe in self.pipes: pipe.close()
        self.events.cancel_join_thread()
        self.events.close()
        self.running = False
