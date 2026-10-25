from itertools import product

Words = tuple(''.join(bits) for bits in product('01', repeat=3))
Terminal = {'000', '111'}
Lateral = {'010', '101'}
Linchpin = {'001', '110'}
Weak = {'011', '100'}
RoleNames = {'terminal': 'Apex', 'weak': 'Loop', 'linchpin': 'Hook', 'lateral': 'Gate'}
Glyph = {'000': '⚜️', '111': '⚜️', '100': '🎗️', '011': '🎗️',
         '110': '🪝', '001': '🪝', '010': '🔱', '101': '🔱'}
LogGlyphCells = {'⚜️': 2, '🎗️': 2, '🪝': 2, '🔱': 2}

def Role(word: str) -> str:
    if word in Terminal: return 'terminal'
    if word in Lateral: return 'lateral'
    if word in Linchpin: return 'linchpin'
    if word in Weak: return 'weak'
    raise ValueError(word)

def Polarity(word: str) -> str:
    if word not in Words:
        raise ValueError('Trinity must be exactly three binary digits')
    return '+' if word in ('011', '101', '110', '111') else '-'

def RoleLabel(word: str) -> str:
    sign = Glyph[word]
    return f'{sign}{RoleNames[Role(word)]}{sign}'

def VertexLabel(word: str) -> str:
    sign = Glyph[word]
    return f'{sign}{word}{sign}'

def WordOf(projection: str) -> str | None:
    if len(projection) != 3 or any(c not in '+-' for c in projection):
        return None
    return ''.join('1' if c == '+' else '0' for c in projection)

def Projection(word: str) -> str:
    if word not in Words:
        raise ValueError(word)
    return ''.join('+' if c == '1' else '-' for c in word)
