from itertools import product

Words = tuple(''.join(bits) for bits in product('01', repeat=3))
Terminal = {'000', '111'}
Lateral = {'010', '101'}
Linchpin = {'001', '110'}
Weak = {'011', '100'}

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

def WordOf(projection: str) -> str | None:
    if len(projection) != 3 or any(c not in '+-' for c in projection):
        return None
    return ''.join('1' if c == '+' else '0' for c in projection)

def Projection(word: str) -> str:
    if word not in Words:
        raise ValueError(word)
    return ''.join('+' if c == '1' else '-' for c in word)

PAIRS = ('++', '--', '+-')
SHAPES = {'terminal': '{0}_{0}', 'weak': '{0}__',
          'linchpin': '__{0}', 'lateral': '_{0}_'}

def Pair(a: str, b: str) -> str:
    if a not in ('+', '-') or b not in ('+', '-'):
        raise ValueError('Helix charge must be positive or negative')
    return '+-' if a != b else a + b

def Prime(word: str) -> str:
    if word not in Words:
        raise ValueError('Prime source must be a three-bit Volt')
    return SHAPES[Role(word)].format(Polarity(word))

PRIMES = {Prime(word): word for word in Words}
assert len(PRIMES) == 8, 'Every cube vertex must have a distinct Prime'

def Vertex(prime: str) -> str:
    try:
        return PRIMES[prime]
    except KeyError:
        raise ValueError('Not an admissible Prime shape') from None

def Mask(pair: str) -> str:
    if pair not in PAIRS:
        raise ValueError('Evidence must be ++, -- or unordered opposing +-')
    a, b = (int(c == '+') for c in pair)
    return f'{a & b}{a ^ b}{int(not (a | b))}'

def Tessellate(prime: str, evidence: str) -> str:
    origin = Vertex(prime)
    mask = Mask(evidence)
    Volt = ''.join(str(int(x) ^ int(y)) for x, y in zip(origin, mask))
    assert sum(x != y for x, y in zip(origin, Volt)) == 1
    return Projection(Volt)

def Admit(origin: str, evidence: str) -> str:
    return WordOf(Tessellate(Prime(origin), evidence))

def Template(origin: str) -> str:
    return Prime(origin)
