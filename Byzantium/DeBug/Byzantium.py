import os
import sys
import json
import time
import threading
from functools import wraps

base = os.path.dirname(os.path.abspath(__file__))
game = os.path.join(base, 'Game')

if game not in sys.path:
    sys.path.insert(0, game)

LOG_PATH = os.path.join(base, 'Byzantium.log')
LOG_LOCK = threading.Lock()


def Short(value, n=16):
    text = str(value)
    return text if len(text) <= n else text[:n]


def RawShape(raw, limit=64):
    if not isinstance(raw, (bytes, bytearray)):
        return {'type': type(raw).__name__, 'size': None, 'headhex': ''}
    blob = bytes(raw)
    return {
        'type': type(raw).__name__,
        'size': len(blob),
        'headhex': blob[:max(0, int(limit))].hex(),
    }


def PacketShape(packet):
    try:
        import Field
    except Exception:
        Field = None

    if packet is None:
        return {'type': 'None'}

    if isinstance(packet, dict):
        shape = {
            'type': 'dict',
            'keys': sorted(str(k) for k in packet.keys()),
            'header': str(packet.get('header', '') or '').strip().lower(),
            'kind': str(packet.get('kind', '') or '').strip().lower(),
        }
        if 'souls' in packet:
            try:
                shape['soulscount'] = len(list(packet.get('souls') or []))
            except Exception:
                shape['soulscount'] = 'bad'
        if 'cells' in packet:
            try:
                shape['cellscount'] = len(list(packet.get('cells') or []))
            except Exception:
                shape['cellscount'] = 'bad'
        if 'saltbody' in packet:
            try:
                shape['saltcount'] = len(list(packet.get('saltbody') or []))
            except Exception:
                shape['saltcount'] = 'bad'
        if 'key' in packet:
            shape['keyhead'] = Short(packet.get('key', ''), 12)
        try:
            shape['packetsize'] = len(json.dumps(packet, sort_keys=True, separators=(',', ':'), default=str).encode('utf-8'))
        except Exception:
            shape['packetsize'] = 'unknown'
        return shape

    if Field is not None:
        try:
            if isinstance(packet, Field.State):
                return {
                    'type': 'Field.State',
                    'cellscount': len(tuple(packet.cells or ())),
                    'monumentcount': len(tuple(packet.monument or ())),
                    'salttotal': int(packet.saltTotal),
                    'selfsoul': str(packet.self[0] or ''),
                    'selfkeyhead': Short(packet.self[1] or '', 12),
                }
        except Exception:
            pass
        try:
            if isinstance(packet, Field.SaltGlyph):
                return {
                    'type': 'Field.SaltGlyph',
                    'keyhead': Short(packet.key, 12),
                    'saltcount': len(tuple(packet.saltbody or ())),
                    'texthead': Short(getattr(packet.textbody, 'text', ''), 32),
                }
        except Exception:
            pass

    if isinstance(packet, str):
        return {'type': 'str', 'text': Short(packet, 32)}

    return {'type': type(packet).__name__}


def Jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        return RawShape(value)
    if isinstance(value, tuple):
        return [Jsonable(v) for v in value]
    if isinstance(value, list):
        return [Jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): Jsonable(v) for k, v in value.items()}
    return PacketShape(value)


def Log(event, **payload):
    try:
        body = {
            'ts': time.strftime('%Y-%m-%d %H:%M:%S'),
            'thread': threading.current_thread().name,
            'event': str(event or ''),
        }
        for key, value in payload.items():
            body[str(key)] = Jsonable(value)
        line = json.dumps(body, sort_keys=True, default=str)
        with LOG_LOCK:
            with open(LOG_PATH, 'a', encoding='utf-8') as handle:
                handle.write(line + '\n')
    except Exception:
        pass


def StateCard(obj):
    try:
        state = getattr(obj, 'state', None)
        if state is None:
            return {'state': 'none'}
        return PacketShape(state)
    except Exception as exc:
        return {'state': 'error', 'error': repr(exc)}


def BoxCard(obj):
    try:
        box = getattr(obj, 'box', None)
        if box is None:
            return {'box': 'none'}
        out = {}
        for lane in ('vault', 'crypt', 'ashfall'):
            try:
                out[lane] = PacketShape(getattr(box, lane, None))
            except Exception as exc:
                out[lane] = {'type': 'error', 'error': repr(exc)}
        return out
    except Exception as exc:
        return {'box': 'error', 'error': repr(exc)}


def WrapMethod(cls, name, builder=None):
    original = getattr(cls, name, None)
    if original is None or getattr(original, '__byzantium_cocoon__', False):
        return

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        before = {}
        if builder is not None:
            try:
                before = builder(self, 'enter', args, kwargs, None)
            except Exception as exc:
                before = {'buildererror': repr(exc)}
        Log(f'{cls.__name__}.{name}.enter', **before)
        try:
            result = original(self, *args, **kwargs)
            after = {}
            if builder is not None:
                try:
                    after = builder(self, 'exit', args, kwargs, result)
                except Exception as exc:
                    after = {'buildererror': repr(exc)}
            Log(f'{cls.__name__}.{name}.exit', **after)
            return result
        except Exception as exc:
            fail = {}
            if builder is not None:
                try:
                    fail = builder(self, 'error', args, kwargs, None)
                except Exception as inner:
                    fail = {'buildererror': repr(inner)}
            fail['error'] = repr(exc)
            Log(f'{cls.__name__}.{name}.error', **fail)
            raise

    wrapped.__byzantium_cocoon__ = True
    setattr(cls, name, wrapped)


def CryptBuilder(self, phase, args, kwargs, result):
    data = {
        'phase': phase,
        'genesisdone': bool(getattr(self, 'genesisdone', False)),
        'bindhost': str(getattr(self, 'bindhost', '') or ''),
        'bindport': getattr(self, 'bindport', None),
        'statecard': StateCard(self),
        'glyphcard': PacketShape(getattr(self, 'glyph', None)),
        'selfcard': {
            'soul': str(getattr(getattr(self, 'self', None), 'soul', '') or ''),
            'keyhead': Short(getattr(getattr(self, 'self', None), 'key', '') or '', 12),
        },
    }
    if phase == 'enter':
        if args:
            first = args[0]
            if isinstance(first, (bytes, bytearray)):
                data['raw'] = RawShape(first)
            else:
                data['payload'] = PacketShape(first)
        if len(args) > 1:
            data['addr'] = args[1]
        if 'addr' in kwargs:
            data['addr'] = kwargs.get('addr')
    if phase == 'exit' and result is not None:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], (bytes, bytearray)):
            data['result'] = {'raw': RawShape(result[0]), 'addr': Jsonable(result[1])}
        else:
            data['result'] = PacketShape(result)
    return data


def DreamBuilder(self, phase, args, kwargs, result):
    data = {
        'phase': phase,
        'changed': bool(getattr(self, 'changed', False)),
        'bootflare': bool(getattr(self, 'bootflare', False)),
        'statecard': StateCard(self),
        'boxcard': BoxCard(self),
        'glyphcard': PacketShape(getattr(self, 'glyph', None)),
    }
    if phase == 'enter':
        if args:
            data['payload'] = PacketShape(args[0])
        if len(args) > 1:
            data['arg1'] = Jsonable(args[1])
        if 'source' in kwargs:
            data['source'] = kwargs.get('source')
        if 'publish' in kwargs:
            data['publish'] = kwargs.get('publish')
    if phase == 'exit' and result is not None:
        data['result'] = PacketShape(result)
    return data


def SanctumBuilder(self, phase, args, kwargs, result):
    data = {
        'phase': phase,
        'statecard': StateCard(self),
    }
    if phase == 'enter' and args:
        data['payload'] = PacketShape(args[0])
    if phase == 'exit' and result is not None:
        data['result'] = PacketShape(result)
    return data


def InstallCocoon():
    import Crypt
    import Dream
    import Sanctum

    Log('cocoon.install.start', module='Byzantium.py', logpath=LOG_PATH)

    for name in [
        'Start', 'Sleep', 'Listen', 'Wake', 'Tick', 'Summon', 'Poll', 'Receive',
        'Cryptkeeper', 'SoulSqueeze', 'Genesis', 'RouteGlyph', 'Emit', 'EmitGlyph',
        'EmitSouls', 'EmitCompleteSouls', 'BuildState', 'Encrypt', 'Decrypt'
    ]:
        WrapMethod(Crypt.Crypt, name, CryptBuilder)

    for name in [
        'Genesis', 'Wake', 'RouteVault', 'RouteCrypt', 'AcceptState', 'Publish',
        'Mutate', 'MutateSalt', 'MutateDream', 'MutatePurge', 'Commit', 'Forward',
        'Packet', 'BoxDream', 'Ashfall'
    ]:
        WrapMethod(Dream.Dream, name, DreamBuilder)

    for name in ['Genesis']:
        WrapMethod(Sanctum.Sanctum, name, SanctumBuilder)

    Log('cocoon.install.done')


InstallCocoon()

import Gateway


def main():
    Log('Byzantium.main.enter')
    Gateway.main()
    Log('Byzantium.main.exit')


if __name__ == '__main__':
    main()
