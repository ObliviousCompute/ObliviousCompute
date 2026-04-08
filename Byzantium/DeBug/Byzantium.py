import os
import sys
import json
import time
import threading
from functools import wraps

base = os.path.dirname(os.path.abspath(__file__))
game = os.path.join(base, "Game")

if game not in sys.path:
    sys.path.insert(0, game)

LOG_PATH = os.path.join(base, "Byzantium.log")
_LOG_LOCK = threading.Lock()


def _short(value, n=16):
    text = str(value)
    return text if len(text) <= n else text[:n]


def _rawshape(raw, limit=64):
    if not isinstance(raw, (bytes, bytearray)):
        return {"type": type(raw).__name__, "size": None, "headhex": ""}
    blob = bytes(raw)
    return {
        "type": type(raw).__name__,
        "size": len(blob),
        "headhex": blob[:max(0, int(limit))].hex(),
    }


def _packetshape(packet):
    try:
        import Field
    except Exception:
        Field = None

    if packet is None:
        return {"type": "None"}

    if isinstance(packet, dict):
        shape = {
            "type": "dict",
            "keys": sorted(str(k) for k in packet.keys()),
            "header": str(packet.get("header", "") or "").strip().upper(),
            "kind": str(packet.get("kind", "") or "").strip().upper(),
        }
        if "souls" in packet:
            try:
                shape["soulscount"] = len(list(packet.get("souls") or []))
            except Exception:
                shape["soulscount"] = "bad"
        if "cells" in packet:
            try:
                shape["cellscount"] = len(list(packet.get("cells") or []))
            except Exception:
                shape["cellscount"] = "bad"
        if "saltbody" in packet:
            try:
                shape["saltcount"] = len(list(packet.get("saltbody") or []))
            except Exception:
                shape["saltcount"] = "bad"
        if "key" in packet:
            shape["keyhead"] = _short(packet.get("key", ""), 12)
        try:
            shape["packetsize"] = len(json.dumps(packet, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8"))
        except Exception:
            shape["packetsize"] = "unknown"
        return shape

    if Field is not None:
        try:
            if isinstance(packet, Field.State):
                return {
                    "type": "Field.State",
                    "cellscount": len(tuple(packet.cells or ())),
                    "monumentcount": len(tuple(packet.monument or ())),
                    "saltTotal": int(packet.saltTotal),
                    "selfsoul": str(packet.self[0] or ""),
                    "selfkeyhead": _short(packet.self[1] or "", 12),
                }
        except Exception:
            pass
        try:
            if isinstance(packet, Field.SaltGlyph):
                return {
                    "type": "Field.SaltGlyph",
                    "keyhead": _short(packet.key, 12),
                    "saltcount": len(tuple(packet.saltbody or ())),
                    "texthead": _short(getattr(packet.textbody, "text", ""), 32),
                }
        except Exception:
            pass

    if isinstance(packet, str):
        return {"type": "str", "text": _short(packet, 32)}

    return {"type": type(packet).__name__}


def _statecard(obj):
    try:
        state = getattr(obj, "state", None)
        if state is None:
            return {"state": "none"}
        return _packetshape(state)
    except Exception as exc:
        return {"state": "error", "error": repr(exc)}


def _boxcard(obj):
    try:
        box = getattr(obj, "box", None)
        if box is None:
            return {"box": "none"}
        out = {}
        for lane in ("vault", "crypt", "ashfall"):
            try:
                out[lane] = _packetshape(getattr(box, lane, None))
            except Exception as exc:
                out[lane] = {"type": "error", "error": repr(exc)}
        return out
    except Exception as exc:
        return {"box": "error", "error": repr(exc)}


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (bytes, bytearray)):
        return _rawshape(value)
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return _packetshape(value)


def _log(event, **payload):
    try:
        body = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "thread": threading.current_thread().name,
            "event": str(event or ""),
        }
        for key, value in payload.items():
            body[str(key)] = _jsonable(value)
        line = json.dumps(body, sort_keys=True, default=str)
        with _LOG_LOCK:
            with open(LOG_PATH, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        pass


def _wrap_method(cls, name, phase_builder=None):
    original = getattr(cls, name, None)
    if original is None or getattr(original, "__byzantium_cocoon__", False):
        return

    @wraps(original)
    def wrapped(self, *args, **kwargs):
        before = {}
        if phase_builder is not None:
            try:
                before = phase_builder(self, "enter", args, kwargs, None)
            except Exception as exc:
                before = {"phase_builder_error": repr(exc)}
        _log(f"{cls.__name__}.{name}.enter", **before)
        try:
            result = original(self, *args, **kwargs)
            after = {}
            if phase_builder is not None:
                try:
                    after = phase_builder(self, "exit", args, kwargs, result)
                except Exception as exc:
                    after = {"phase_builder_error": repr(exc)}
            _log(f"{cls.__name__}.{name}.exit", **after)
            return result
        except Exception as exc:
            fail = {}
            if phase_builder is not None:
                try:
                    fail = phase_builder(self, "error", args, kwargs, None)
                except Exception as inner:
                    fail = {"phase_builder_error": repr(inner)}
            fail["error"] = repr(exc)
            _log(f"{cls.__name__}.{name}.error", **fail)
            raise

    wrapped.__byzantium_cocoon__ = True
    setattr(cls, name, wrapped)


def _crypt_builder(self, phase, args, kwargs, result):
    data = {
        "phase": phase,
        "genesisdone": bool(getattr(self, "genesisdone", False)),
        "bindhost": str(getattr(self, "bindhost", "") or ""),
        "bindport": getattr(self, "bindport", None),
        "statecard": _statecard(self),
        "selfcard": {
            "soul": str(getattr(getattr(self, "self", None), "soul", "") or ""),
            "keyhead": _short(getattr(getattr(self, "self", None), "key", "") or "", 12),
        },
    }

    if phase == "enter":
        if args:
            first = args[0]
            if isinstance(first, (bytes, bytearray)):
                data["raw"] = _rawshape(first)
            else:
                data["payload"] = _packetshape(first)
        if len(args) > 1:
            data["addr"] = args[1]
        if "addr" in kwargs:
            data["addr"] = kwargs.get("addr")

    if phase == "exit" and result is not None:
        data["result"] = _packetshape(result)

    return data


def _dream_builder(self, phase, args, kwargs, result):
    data = {
        "phase": phase,
        "changed": bool(getattr(self, "changed", False)),
        "bootflare": bool(getattr(self, "bootflare", False)),
        "statecard": _statecard(self),
        "boxcard": _boxcard(self),
    }

    if phase == "enter":
        if args:
            data["payload"] = _packetshape(args[0])
        if len(args) > 1:
            data["arg1"] = _jsonable(args[1])
        if "source" in kwargs:
            data["source"] = kwargs.get("source")
        if "publish" in kwargs:
            data["publish"] = kwargs.get("publish")

    if phase == "exit" and result is not None:
        data["result"] = _packetshape(result)

    return data


def install_cocoon():
    import Crypt
    import Dream

    _log("cocoon.install.start", module="Byzantium.py", logpath=LOG_PATH)

    for meth in [
        "receive",
        "Cryptkeeper",
        "routeGlyph",
        "emit",
        "emitGlyph",
        "emitSouls",
        "emitCompleteSouls",
        "Genesis",
        "encrypt",
        "decrypt",
    ]:
        _wrap_method(Crypt.Crypt, meth, _crypt_builder)

    for meth in [
        "routevault",
        "routecrypt",
        "mutate",
        "acceptstate",
        "forward",
        "publish",
        "packet",
        "boxdream",
        "mutatesalt",
        "mutatedream",
        "mutatepurge",
    ]:
        _wrap_method(Dream.Dream, meth, _dream_builder)

    _log("cocoon.install.done")


install_cocoon()

import Gateway


def main():
    _log("Byzantium.main.enter")
    Gateway.main()
    _log("Byzantium.main.exit")


if __name__ == "__main__":
    main()
