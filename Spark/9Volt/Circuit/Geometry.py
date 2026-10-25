from .Vertex import Words, Role, Polarity, Projection

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
    next_word = ''.join(str(int(x) ^ int(y)) for x, y in zip(origin, mask))
    assert sum(x != y for x, y in zip(origin, next_word)) == 1
    return Projection(next_word)


def Admit(origin: str, evidence: str) -> str:
    from .Vertex import WordOf
    return WordOf(Tessellate(Prime(origin), evidence))


def Template(origin: str) -> str:
    return Prime(origin)
