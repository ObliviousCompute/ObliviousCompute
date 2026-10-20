from __future__ import annotations
from itertools import product

Words = tuple(''.join(p) for p in product('01', repeat=3))
Poles = {'+': '000', '-': '111'}
Ordinary = {'+': ('000','100','110','111'), '-': ('111','011','001','000')}
Terminal = {'000', '111'}
Lateral = {'010', '101'}
Linchpin = {'001', '110'}
Weak = {'011', '100'}

RoleNames = {'terminal': 'Apex', 'weak': 'Loop', 'linchpin': 'Hook', 'lateral': 'Gate'}
Glyph = {'000': '⚜️', '111': '⚜️', '100': '🎗️', '011': '🎗️',
         '110': '🪝', '001': '🪝', '010': '🔱', '101': '🔱'}
LogGlyphCells = {'⚜️': 2, '🎗️': 2, '🪝': 2, '🔱': 2}
def RoleLabel(word: str) -> str:
    sign = Glyph[word]
    return f'{sign}{RoleNames[Role(word)]}{sign}'
def VertexLabel(word: str) -> str:
    sign = Glyph[word]
    return f'{sign}{word}{sign}'

def Polarity(word: str) -> str:
    if word not in Words:
        raise ValueError('Trinity must be exactly three binary digits')
    return '+' if word[0] == '1' else '-'

def Role(word: str) -> str:
    if word in Terminal: return 'terminal'
    if word in Lateral: return 'lateral'
    if word in Linchpin: return 'linchpin'
    if word in Weak: return 'weak'
    raise ValueError(word)

def Jump(word: str) -> str:
    return ''.join((('0' if word[0]=='1' else '1'), word[1],
                    ('0' if word[2]=='1' else '1')))
def LateralPair(charge: str) -> str:
    return '++_' if charge == '+' else '_--'

def LateralAppend(middle: str, contribution: str) -> str:
    return contribution + middle + '_' if middle == '+' else '_' + middle + contribution

def WordOf(projection: str) -> str | None:
    if len(projection) != 3 or any(c not in '+-' for c in projection): return None
    return ''.join('1' if c == '+' else '0' for c in projection)

def Admission(phase: str, charge: str, projection: str,
              receiver: bool, payload: str) -> tuple[tuple[str, str, str], ...]:
    def Change(value, reason): return (('set', value, reason),)
    def Move(reason): return (('walk', '', reason),)
    def Capture(reason): return (('capture', '', reason),)
    def Replay(reason): return (('reproject', '', reason),)
    if len(payload) == 1:
        if phase == 'helix':
            if payload == charge:
                return Move('matching bare charge')
            return Capture('opposing bare charge') + Change('_'+charge+'_', 'capture holds Helix charge')
        if phase == 'walking':
            return (Move('matching bare charge') if payload == charge else
                    Replay('opposing charge, unchanged'))
        return ()
    if payload[0] == '_' and payload[2] == '_':
        middle = payload[1]
        if phase == 'helix':
            action = (Capture('opposing lateral, capture') if middle != charge
                      else Move('matching lateral, append'))
            return action + Change(LateralAppend(middle, charge), 'Helix admits lateral')
        if phase == 'walking' and projection == charge and not receiver:
            return Change(LateralAppend(middle, charge), 'walk admits lateral')
        if phase == 'captured' and middle == charge and projection == '_'+charge+'_':
            return Change(LateralPair(charge), 'two matching laterals')
        if phase == 'terminal' and middle != charge and projection == charge+'_'+charge:
            proposed = '_'+middle+charge if charge == '+' else charge+middle+'_'
            return Change(proposed, 'terminal admits other lateral')
        return ()
    if payload in ('++_', '_--'):
        mirror = '+' if payload == '++_' else '-'
        if phase == 'walking' and projection == charge and charge != mirror:
            complete = '++'+charge if mirror == '+' else charge+'--'
            return Change(complete, 'transitory after two captures')
        if phase == 'terminal' and receiver and charge != mirror:
            complete = mirror*2+charge if charge == '+' else charge+mirror*2
            return Change(complete, 'terminal + two laterals')
        if phase in ('walking', 'helix') and projection == charge:
            complete = '++'+charge if mirror == '+' else charge+'--'
            return Change(complete, 'third after two laterals')
        if phase == 'terminal' and projection == charge+'_'+charge and charge == mirror:
            return ()
        if phase == 'captured' and projection == payload:
            return ()
        if phase == 'walking' and projection == payload and not receiver:
            full = payload[:2]+charge if mirror == '+' else charge+payload[1:]
            return Change(full, '2 matching appenders, primed')
        return ()
    if payload[0] == '_' and payload[1] in '+-' and payload[2] in '+-':
        if (phase == 'terminal' and not receiver and charge == '+' and
                payload == '_++' and projection == payload):
            return Change('+++', 'positive terminal appenders')
        if phase == 'terminal' and receiver and charge == '+' and payload == '_-+':
            return Change('+-+', 'terminal admits Helix/lateral')
        if phase == 'walking' and not receiver and projection != payload:
            return Change(charge+payload[1:], 'Helix fills positive prime')
        if phase == 'captured' and projection == payload and not receiver:
            return Change(charge+payload[1:], 'Helix fills positive prime')
        if phase == 'walking' and receiver and projection == charge:
            return Change(charge+payload[1:], 'transitory fills positive prime')
        return ()
    if payload[0] in '+-' and payload[1] in '+-' and payload[2] == '_':
        if (phase == 'terminal' and not receiver and charge == '-' and
                payload == '--_' and projection == payload):
            return Change('---', 'negative terminal appenders')
        if phase == 'terminal' and receiver and charge == '-' and payload == '-+_':
            return Change('-+-', 'terminal admits Helix/lateral')
        if (phase == 'walking' and not receiver and projection != payload and
                projection.startswith(charge)):
            return Change(payload[:2]+charge, 'Helix fills negative prime')
        if phase == 'captured' and projection == payload and not receiver:
            return Change(payload[:2]+charge, 'Helix fills negative prime')
        if phase == 'walking' and projection == charge and receiver:
            return Change(payload[:2]+charge, 'transitory fills negative prime')
        return ()
    if payload[0] in '+-' and payload[1] == '_' and payload[2] == payload[0]:
        if phase == 'terminal' and projection == payload and not receiver:
            shape = '_'+charge*2 if charge == '+' else charge*2+'_'
            return Change(shape, 'terminal pair needs third')
        if phase == 'captured':
            return Replay('terminal pings lateral')
        return ()
    return (('open', payload, 'unknown shaped projection'),)
