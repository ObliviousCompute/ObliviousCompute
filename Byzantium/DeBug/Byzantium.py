import os
import sys
import time
import json
from functools import wraps

base = os.path.dirname(os.path.abspath(__file__))
game = os.path.join(base, "Game")

if game not in sys.path:
    sys.path.insert(0, game)

import Gateway
import Crypt
import Dream

LOG_PATH = os.path.join(base, "Byzantium.log")


def Log(event, **data):
    try:
        line = {"t": round(time.time(), 6), "e": str(event or "")}
        line.update(data)
        with open(LOG_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True, default=str) + "\n")
    except Exception:
        pass


def Shape(value):
    try:
        import Field
    except Exception:
        Field = None

    try:
        if value is None:
            return {"type": "None"}

        if isinstance(value, dict):
            textbody = value.get("textbody", {}) or {}
            if not isinstance(textbody, dict):
                textbody = {}
            return {
                "type": "dict",
                "header": str(value.get("header", "") or "").strip().lower(),
                "kind": str(value.get("kind", "") or "").strip().lower(),
                "key": str(value.get("key", "") or "")[:8],
                "souls": len(list(value.get("souls", ()) or ())) if "souls" in value else None,
                "cells": len(list(value.get("cells", ()) or ())) if "cells" in value else None,
                "salt": len(list(value.get("saltbody", ()) or ())) if "saltbody" in value else None,
                "text": str(textbody.get("text", "") or "")[-16:],
            }

        if Field is not None:
            if isinstance(value, Field.State):
                return {
                    "type": "Field.State",
                    "cells": len(tuple(value.cells or ())),
                    "monument": len(tuple(value.monument or ())),
                    "self": str(value.self[0] or "")[:8],
                }

            if isinstance(value, Field.SaltGlyph):
                return {
                    "type": "Field.SaltGlyph",
                    "key": str(value.key or "")[:8],
                    "salt": len(tuple(value.saltbody or ())),
                    "text": str(getattr(value.textbody, "text", "") or "")[-16:],
                }

        if isinstance(value, str):
            return {"type": "str", "text": value[-16:]}

        return {"type": type(value).__name__}
    except Exception as exc:
        return {"type": "shapeerror", "error": repr(exc)}


def Install():
    if getattr(Install, "Done", False):
        return
    Install.Done = True

    orig_emitglyph = Crypt.Crypt.EmitGlyph

    @wraps(orig_emitglyph)
    def EmitGlyph(self, glyph):
        Log("WireOutGlyph", shape=Shape(glyph))
        return orig_emitglyph(self, glyph)

    Crypt.Crypt.EmitGlyph = EmitGlyph

    orig_emitsouls = Crypt.Crypt.EmitSouls

    @wraps(orig_emitsouls)
    def EmitSouls(self):
        try:
            soulcount = len(tuple(getattr(self, "souls", ()) or ()))
        except Exception:
            soulcount = None
        Log("WireOutSouls", souls=soulcount)
        return orig_emitsouls(self)

    Crypt.Crypt.EmitSouls = EmitSouls

    orig_emitcomplete = Crypt.Crypt.EmitCompleteSouls

    @wraps(orig_emitcomplete)
    def EmitCompleteSouls(self):
        try:
            soulcount = len(tuple(getattr(self, "complete", ()) or ()) or tuple(getattr(self, "souls", ()) or ()))
        except Exception:
            soulcount = None
        Log("WireOutSoulsFull", souls=soulcount)
        return orig_emitcomplete(self)

    Crypt.Crypt.EmitCompleteSouls = EmitCompleteSouls

    orig_routeglyph = Crypt.Crypt.RouteGlyph

    @wraps(orig_routeglyph)
    def RouteGlyph(self, packet, addr):
        Log("CryptUp", addr=str(addr), shape=Shape(packet))
        return orig_routeglyph(self, packet, addr)

    Crypt.Crypt.RouteGlyph = RouteGlyph

    orig_routecrypt = Dream.Dream.RouteCrypt

    @wraps(orig_routecrypt)
    def RouteCrypt(self):
        glyph = getattr(getattr(self, "box", None), "crypt", None)
        Log("DreamIn", shape=Shape(glyph))
        return orig_routecrypt(self)

    Dream.Dream.RouteCrypt = RouteCrypt

    orig_mutate = Dream.Dream.Mutate

    @wraps(orig_mutate)
    def Mutate(self, glyph, source=''):
        Log("DreamMutateStart", source=str(source or ""), shape=Shape(glyph))
        result = orig_mutate(self, glyph, source=source)
        Log("DreamMutateEnd", source=str(source or ""), changed=bool(getattr(self, "changed", False)), result=bool(result))
        return result

    Dream.Dream.Mutate = Mutate

    orig_forward = Dream.Dream.Forward

    @wraps(orig_forward)
    def Forward(self, glyph):
        Log("DreamOut", shape=Shape(glyph))
        return orig_forward(self, glyph)

    Dream.Dream.Forward = Forward

    Log("Install")


Install()


def main():
    Log("Start")
    Gateway.main()
    Log("Stop")


if __name__ == "__main__":
    main()
