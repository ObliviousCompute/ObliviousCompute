import re
import time
import sys
import termios
import tty
import os
import textwrap
from dataclasses import dataclass, replace
from typing import List
from select import select
from enum import Enum

class Focus(str, Enum):
    TITLE = "TITLE"
    MENU = "MENU"
    TABLE_MOVE = "TABLE_MOVE"
    TABLE_LOCK = "TABLE_LOCK"
    SPINE = "SPINE"

class Action(str, Enum):
    PURGE = "PURGE"
    WHISPER = "WHISPER"
    RALLY = "RALLY"
    WRATH = "WRATH"
    DEFECT = "DEFECT"
    MONUMENT = "MONUMENT"
    LORE = "LORE"
    EXIT = "EXIT"

# ---------------- Configuration/ Geometry ------------------------------------
# Terminal frame geometry
TERM_W      = 80   # total terminal width used by the renderer
FRAME_PAD   = 1    # left/right padding added when framing lines
INNER_W     = TERM_W - (FRAME_PAD * 2)

# Board geometry 
BOARD_COLS  = 4
BOARD_ROWS  = 6
CELL_COUNT  = BOARD_COLS * BOARD_ROWS

# Fixed field widths
NAME_W      = 8
COST_W      = 8
TABLE_SALT_W= 7     # salt number inside the city table cell

# Spacing
COL_GAP     = 4     # spaces between city columns
HLINE       = "=" * TERM_W

# ---------------- Focus / Channels -------------------------------------------
CHAN_FLAME = "FLAME"
CHAN_EMBER = "EMBER"
CHAN_ASH   = "ASH"

MENU = [Action.PURGE, Action.WHISPER, Action.RALLY, Action.WRATH, Action.DEFECT, Action.MONUMENT, Action.LORE, Action.EXIT]
ASHFALL_MAX = 7
BODY_FILL_LINES = 23  # body height before the bottom cap (keeps framing stable)
PACK_RACE_THRESH_DEFAULT = 25000
FPS = 55.0

# ======================================================================
# GLOBAL COLOR PALETTE
# ======================================================================
RESET    = "\x1b[0m"
ASH      = "\x1b[90m"
WHITE    = "\x1b[97m"
EMBER    = "\x1b[38;5;130m"
SALT     = "" 

FLICKER1 = "\x1b[38;5;208m"
FLICKER2 = "\x1b[38;5;214m"

def flame_pair(phase: int) -> tuple[str, str]:

    if phase % 2:
        return (FLICKER1, FLICKER2)
    return (FLICKER2, FLICKER1)

def palette(cache: "UiCache") -> dict[str, str]:

    flame, flare = flame_pair(cache.flame_phase)
    return {
        "reset": RESET,
        "ash": ASH,
        "white": WHITE,
        "ember": EMBER,
        "salt": SALT,
        "flame": flame,   # active flicker
        "flare": flare,   # inverse flicker
    }

def chan_color(pal: dict[str, str], chan: str) -> str:
    """Map feed channel -> color."""
    if chan == CHAN_FLAME:
        return pal["flame"]
    if chan == CHAN_EMBER:
        return pal["ember"]
    return pal["ash"]

# #############################################################################
# ##                )══{≡≡≡≡≡≡≡≡>  SALT CACHE  <≡≡≡≡≡≡≡≡}══(                 ##
# #############################################################################
# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
@dataclass(frozen=True)
class Cell:
    name: str
    salt: int
    team: int
    is_general: bool = False
    id4: str = ""

@dataclass(frozen=True)
class BoardSnapshot:
    cells: List[Cell]
    monuments: List[str]
    title_lines: List[str]

@dataclass
class UiCache:
    feed: list
    local_name: str
    monuments: list = None
    snap: "BoardSnapshot" = None
    me_idx: int = 0
    me_id4: str = ""
    focus: Focus = Focus.MENU
    menu_idx: int = 0
    city_idx: int = 0
    target_idx: int = None
    whisper_target: str = ""
    salt: int = 1
    text: str = ""
    visible_feed_count: int = 0
    flame_phase: int = 0
    flame_fed: bool = False
    pending_request: str | None = None
    show_banner: bool = True
    show_lore: bool = False

# ---------------- Derived Context --------------------------------------------
@dataclass(frozen=True)
class FrameCtx:
    """Derived context computed once per frame to avoid ad-hoc recomputation."""
    action: Action
    focus: Focus
    me_idx: int
    city_idx: int
    target_idx: int | None
    floor: int
    label: str | None
    needs_target: bool
    has_arm_phase: bool
    in_table_move: bool
    in_table_lock: bool
    in_spine: bool

def build_ctx(cache: "UiCache", snap: "BoardSnapshot") -> FrameCtx:
    action = MENU[cache.menu_idx]
    focus = cache.focus
    me_idx = getattr(cache, "me_idx", 0)
    city_idx = getattr(cache, "city_idx", 0)
    target_idx = getattr(cache, "target_idx", None)

    floor = action_floor(action, cache)
    label = action_spine_label(action, cache)

    meta = ACTION_META.get(action, {})
    needs_target = bool(meta.get("needs_target", False))
    has_arm_phase = bool(meta.get("has_arm_phase", False))

    return FrameCtx(
        action=action,
        focus=focus,
        me_idx=me_idx,
        city_idx=city_idx,
        target_idx=target_idx,
        floor=floor,
        label=label,
        needs_target=needs_target,
        has_arm_phase=has_arm_phase,
        in_table_move=(focus == Focus.TABLE_MOVE),
        in_table_lock=(focus == Focus.TABLE_LOCK),
        in_spine=(focus == Focus.SPINE),
    )

# ---------------- Genesis / Bootstrap ----------------------------------------
def genesis_snapshot(local_name: str = "SATOSHI") -> "BoardSnapshot":
    """Create a frozen 24-seat city with a 1,000,000-salt economy.
    Column = team, row = hierarchy. Balances are randomized once and then
    remain frozen until an ENTER commit mutates them.
    """
    import random

    names = [
        "SATOSHI","ONYX","ORION","CRITIAS","UNKNOWN","JADE",
        "AURELIUS","NOVA","SENECA","OBLIVION","VEGA","OPAL",
        "ATLAS","FINNY","RUNE","SIRIUS","PLATO","ANTIRIS",
        "DENEB","ACHILLES","VESTA","CICERO","TIMIUS","SOCRATES",
    ]

    # Ensure local_name exists and is seated at index 0 
    names[0] = local_name

    # Row totals (4 seats per row) sum to 1,000,000.
    row_totals = [300_000, 220_000, 180_000, 140_000, 90_000, 70_000]
    assert sum(row_totals) == 1_000_000

    salts = [0] * CELL_COUNT
    rng = random.Random(1337)  # stable seed 

    for row, total in enumerate(row_totals):
        # Randomly split 'total' across 4 columns for this row.
        cuts = sorted(rng.sample(range(1, total), 3))
        parts = [cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], total - cuts[2]]
        rng.shuffle(parts)

        for col in range(BOARD_COLS):
            idx = col * BOARD_ROWS + row
            salts[idx] = parts[col]

    cells = []
    assert len(names) == CELL_COUNT

    for i, n in enumerate(names):
        cells.append(Cell(
            name=n,
            salt=salts[i],
            team=i // BOARD_ROWS,
            is_general=(i % BOARD_ROWS == 0),
            id4=f"{i:04d}",
        ))

    return BoardSnapshot(
        cells=cells,
        monuments=[],
        title_lines=[
            "BYZANTIUM (frozen city)",
            "ENTER commits mutate salt • arrows move • ESC quits",
        ],
    )

def fake_snapshot(tick: int) -> BoardSnapshot:
    names = [
        "SATOSHI","ONYX","ORION","CRITIAS","UNKNOWN","JADE",
        "AURELIUS","NOVA","SENECA","OBLIVION","VEGA","OPAL",
        "ATLAS","FINNY","RUNE","SIRIUS","PLATO","ANTIRIS",
        "DENEB","ACHILLES","VESTA","CICERO","TIMIUS","SOCRATES",
    ]

    cells: List[Cell] = []
    assert len(names) == CELL_COUNT

    for i, n in enumerate(names):
        # rankings shifting without exploding
        base = 110_000 - (i * 4_250)
        wobble = int((tick * 157 + i * 211) % 19_000) - 9_500
        s = max(1, base + wobble)
        cells.append(Cell(
            name=n,
            salt=s,
            team=i // BOARD_ROWS,
            is_general=(i % BOARD_ROWS == 0),
            id4=f"{i:04d}",
        ))

    return BoardSnapshot(
        cells=cells,
        monuments=[],
        title_lines=[
            "BYZANTIUM (fake churn snapshot)",
            "ENTER commits mutate salt • arrows move • ESC quits",
        ],
    )

# ---------------- Eligibility / Viability Checks -----------------------------
def _find_by_name(snap: "BoardSnapshot", name: str) -> int:
    for i, c in enumerate(snap.cells):
        if c.name == name:
            return i
    return 0

def _find_by_id4(snap: "BoardSnapshot", id4: str) -> int | None:
    for i, c in enumerate(snap.cells):
        if c.id4 == id4:
            return i
    return None

def _defect_viable(snap: "BoardSnapshot", me: int, idx: int) -> bool:
    """True if `idx` is a legal DEFECT target for `me` (strictly weaker)."""
    if idx == me:
        return False
    if not (0 <= me < len(snap.cells)) or not (0 <= idx < len(snap.cells)):
        return False
    return (snap.cells[idx].salt < snap.cells[me].salt) and ((idx // BOARD_ROWS) != (me // BOARD_ROWS))

def _snap_defect_cursor(snap: "BoardSnapshot", me: int, start_idx: int) -> int | None:
    """Return a viable DEFECT target to start on, or None if none exist.
    Scans the whole board with wrap starting from `start_idx`.
    """
    if not (0 <= start_idx < len(snap.cells)):
        start_idx = 0
    n = len(snap.cells)
    for k in range(n):
        idx = (start_idx + k) % n
        if _defect_viable(snap, me, idx):
            return idx
    return None

# ---------------- DEFECT (single-source rules) -------------------------------
def defect_cost(me_idx: int) -> int:
    """Rank-dependent DEFECT cost. Generals pay 10,000; everyone else 500."""
    return 10000 if (me_idx % BOARD_ROWS) == 0 else 500

def defect_is_viable(snap: "BoardSnapshot", me_idx: int, tgt_idx: int) -> bool:
    """Alias for viability (single source)."""
    return _defect_viable(snap, me_idx, tgt_idx)

def defect_validate(snap: "BoardSnapshot", me_idx: int, tgt_idx: int) -> tuple[bool, str]:
    """Validate a DEFECT attempt. Returns (ok, reason_if_not_ok)."""
    if tgt_idx == me_idx:
        return (False, "cannot defect into yourself")
    if not defect_is_viable(snap, me_idx, tgt_idx):
        # Preserve prior user-facing phrasing where possible.
        if (0 <= tgt_idx < len(snap.cells)) and (0 <= me_idx < len(snap.cells)) and snap.cells[tgt_idx].salt >= snap.cells[me_idx].salt:
            return (False, "target too strong")
        return (False, "invalid target")
    cost = defect_cost(me_idx)
    have = snap.cells[me_idx].salt if 0 <= me_idx < len(snap.cells) else 0
    if have < cost:
        return (False, "insufficient salt")
    return (True, "")

def defect_recipients(snap: "BoardSnapshot", me_idx: int) -> list[int]:

    col = me_idx // BOARD_ROWS
    col_idxs = [col * BOARD_ROWS + r for r in range(BOARD_ROWS)]
    general_idx = col * BOARD_ROWS
    if me_idx == general_idx:
        return [i for i in col_idxs if i != me_idx]
    return [i for i in col_idxs if i != me_idx and i != general_idx]

def defect_apply(snap: "BoardSnapshot", cache: "UiCache", tgt_idx: int) -> tuple[int, int]:

    me_idx = cache.me_idx
    cost = defect_cost(me_idx)

    # Spend (redistribute) first.
    recips = defect_recipients(snap, me_idx)
    spent = _distribute_to_weakest_first(snap, me_idx, recips, cost)
    if spent <= 0:
        return (0, me_idx)

    # Swap seats 
    snap.cells[me_idx], snap.cells[tgt_idx] = snap.cells[tgt_idx], snap.cells[me_idx]

    _normalize_columns(snap)

    # Re-find "me" by stable identity (id4), because normalization can move me within my column.
    i_me = _find_by_id4(snap, cache.me_id4)
    if i_me is None:
        i_me = tgt_idx
    return (spent, i_me)

# ----------------------UI (preview logic only) -------------------------------
def _vislen(s: str) -> int:
    return len(re.sub(r"\x1b\[[0-9;]*m", "", s))

def _clipw(s: str, width: int) -> str:
    """
    Clip a string with ANSI codes to visible width.
    Strategy: measure visible length as we go, break when exceeded.
    """
    out = []
    vis = 0
    i = 0
    L = len(s)
    while i < L:
        if s[i] == "\x1b" and i + 1 < L and s[i+1] == "[":
            j = i + 2
            while j < L and s[j] != "m":
                j += 1
            if j < L and s[j] == "m":
                out.append(s[i:j+1])
                i = j + 1
                continue
        if vis >= width:
            break
        out.append(s[i])
        vis += 1
        i += 1
    return "".join(out)

def _centerw(s: str, width: int) -> str:
    """
    Center a string by visible width (ANSI-safe).
    """
    vis = _vislen(s)
    if vis >= width:
        return s
    pad = (width - vis) // 2
    return (" " * pad) + s

def _clip_term(s: str) -> str:
    """Clip to TERM_W visible columns (ANSI-safe)."""
    return _clipw(s, TERM_W)

def _center_term(s: str) -> str:
    """Center to TERM_W visible columns (ANSI-safe)."""
    return _centerw(s, TERM_W)

# Back-compat aliases
_clip80 = _clip_term
_center80 = _center_term

def _append_paragraph(
    lines: list[str],
    text: str,
    width: int | None = None,
    *,
    color: str | None = None,
    reset: str = "\x1b[0m",
):

    if width is None:
        width = INNER_W

    for block in text.split("\n"):
        wrapped = textwrap.wrap(
            block,
            width=width,
            replace_whitespace=False,
            drop_whitespace=False,
        )
        if not wrapped:
            lines.append("")
            continue

        for ln in wrapped:
            if color:
                lines.append(_clip_term(color + ln + reset))
            else:
                lines.append(_clip_term(ln))

# #############################################################################
# ##                )══{≡≡≡≡≡≡≡≡>   SALTBURN   <≡≡≡≡≡≡≡≡}══(                 ##
# #############################################################################
# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
ACTION_META = {
    Action.WHISPER: {
        "floor": 1,
        "desc": "transfer salt to one seat",
        "preview": "The value lies in words, not the medium.",
        "spine_label": None,
        "feed_chan": CHAN_ASH,
        "has_arm_phase": False,
        "needs_target": True,
    },
    Action.RALLY: {
        "floor": 100,
        "desc": "spend salt on your banner",
        "preview": "Strength grows when it is distributed.",
        "spine_label": "MORALE",
        "feed_chan": CHAN_EMBER,
        "has_arm_phase": True,   # TABLE_LOCK -> SPINE
        "needs_target": False,
    },
    Action.WRATH: {
        "floor": 1000,
        "desc": "spend salt on everyone",
        "preview": "Influence cannot be debased. It is felt by all.",
        "spine_label": "LEGION",
        "feed_chan": CHAN_FLAME,
        "has_arm_phase": True,   # TABLE_LOCK -> SPINE
        "needs_target": False,
    },
    Action.DEFECT: {
        # Dynamic floor (rank-based) is resolved by action_floor().
        "floor": 0,
        "desc": "rank-dependent cost to swap seats",
        "preview": "Change places with a weaker soul.",
        "spine_label": None,     # resolved by action_spine_label()
        "feed_chan": CHAN_EMBER,
        "has_arm_phase": False,
        "needs_target": True,
    },

    # Non-economic menu items
    Action.PURGE: {"preview": "Sweep away the ashes."},
    Action.MONUMENT: {"preview": "What carries weight is not forgotten."},
    Action.LORE: {"preview": "Voices from after the fall."},
    Action.EXIT: {"preview": "ABANDON POST."},
}

def action_preview(action: Action) -> str:
    return ACTION_META.get(action, {}).get("preview", action.value)

def action_base_floor(action: Action) -> int:
    return int(ACTION_META.get(action, {}).get("floor", 1) or 1)

def action_desc(action: Action) -> str:
    return ACTION_META.get(action, {}).get("desc", "")

def action_floor(action: Action, cache: "UiCache") -> int:
    """Resolve the floor cost for an action (dynamic where needed)."""
    if action == Action.DEFECT:
        me = getattr(cache, "me_idx", 0)
        is_gen = (me % BOARD_ROWS) == 0
        return 10000 if is_gen else 500
    return action_base_floor(action)

def action_spine_label(action: Action, cache: "UiCache") -> str | None:
    label = ACTION_META.get(action, {}).get("spine_label")
    if label:
        return str(label)
    if action == Action.DEFECT:
        return "MYRIAD" if action_floor(action, cache) >= 10000 else "COHORT"
    return None

# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
# ============================== INFLOW =======================================
# Final validation, assembling attempts, Plexus invocation
# =============================================================================
def _recompute_cell_flags(snap: "BoardSnapshot") -> None:
    """Keep derived flags consistent with seat positions.
    Cells are frozen; update by replacing list entries.
    """
    for i, c in enumerate(snap.cells):
        snap.cells[i] = replace(
            c,
            team=i // BOARD_ROWS,
            is_general=(i % BOARD_ROWS == 0),
        )

def _sort_column_by_salt(snap: "BoardSnapshot", col: int) -> None:
    """Sort a single column (team) in-place by salt descending.
    Highest salt becomes row 0 (the General seat), then descends.
    Ties break on id4 for stable ordering.
    """
    idxs = [col * BOARD_ROWS + r for r in range(BOARD_ROWS)]
    col_cells = [snap.cells[i] for i in idxs]
    col_cells.sort(key=lambda c: (-c.salt, c.id4))
    for r, c in enumerate(col_cells):
        snap.cells[col * BOARD_ROWS + r] = c

def _normalize_columns(snap: "BoardSnapshot") -> None:
    """Re-rank seats within each column without moving anyone across columns."""
    for col in range(BOARD_COLS):
        _sort_column_by_salt(snap, col)
    _recompute_cell_flags(snap)

def _distribute_to_weakest_first(
    snap: "BoardSnapshot",
    sender_idx: int,
    recipient_idxs: List[int],
    total: int,
) -> int:

    if total <= 0:
        return 0

    recips = []
    seen = set()
    for i in recipient_idxs:
        if i == sender_idx:
            continue
        if i in seen:
            continue
        seen.add(i)
        recips.append(i)

    n = len(recips)
    if n <= 0:
        return 0

    have = snap.cells[sender_idx].salt
    if have < total:
        return 0

    base = total // n
    rem = total - (base * n)

    if base:
        for i in recips:
            snap.cells[i] = replace(snap.cells[i], salt=snap.cells[i].salt + base)

    if rem:
        ordered = sorted(recips, key=lambda i: snap.cells[i].salt)
        for k in range(rem):
            i = ordered[k]
            snap.cells[i] = replace(snap.cells[i], salt=snap.cells[i].salt + 1)

    snap.cells[sender_idx] = replace(snap.cells[sender_idx], salt=have - total)
    return total

# ================================= OUTFLOW ===================================
# Handling Plexus return, applying changes, normalization
# =============================================================================
# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
# #############################################################################
# ##                 )══{≡≡≡≡≡≡≡≡>   ASHFALL   <≡≡≡≡≡≡≡≡}══(                 ##
# #############################################################################
# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
def _fmt_feed_line(speaker: str, cost: int, msg: str, name_w: int = 8, cost_w: int = 8) -> str:
    """
    Feed line format (18-space fixed width):
      NAME(8) SPACE COST(8, right-aligned) COLON MSG
    Total: NAME(8) + SPACE(1) + COST(8) + COLON(1) = 18 fixed
    """
    name = speaker.ljust(name_w)[:name_w]
    cost_raw = f"{cost:+,}"
    cost_s = cost_raw.rjust(cost_w)  # Right-align in 8-char field for 18-space total
    return f"{name} {cost_s}: {msg}"

def _parse_monument(m: str):
    # Try to extract: NAME (8 chars) + space + signed number + colon + rest
    if len(m) < 10:
        return None, None, m
    name = m[:NAME_W].strip()
    tail = m[NAME_W:].strip()
    match = re.match(r"^([+-]?[\d,]+):\s*(.*)", tail)
    if not match:
        return name, None, tail
    score = match.group(1).replace(",", "")
    post = match.group(2)
    return name, score, post

def _monument_anchor_col(monuments: List[str], anchor_name: str = "SATOSHI", name_w: int = 8) -> int:

    for m in monuments:
        name, score, _post = _parse_monument(m)
        if name == anchor_name and score is not None:
            core = f"{name.ljust(name_w)[:name_w]} {score}"
            return len(core)
    best = 0
    for m in monuments:
        name, score, _post = _parse_monument(m)
        if name is None:
            continue
        core = f"{name.ljust(name_w)[:name_w]} {score}"
        best = max(best, len(core))
    return best

def _align_monument_colon(m: str, anchor_col: int, name_w: int = 8) -> str:

    name, score, post = _parse_monument(m)
    if name is None:
        return m

    name_fixed = name.ljust(name_w)[:name_w]
    core = f"{name_fixed} {score}"
    pad = max(0, anchor_col - len(core))
    return core + (" " * pad) + ":" + post

# ---------------- Monument Inscription ----------------
def _maybe_inscribe(cache, speaker, cost, msg):
    """Consider a committed feed line for permanent Monument inscription."""
    line = _fmt_feed_line(speaker, +cost, msg)
    entry = (abs(cost), line)
    cache.monuments.append(entry)
    cache.monuments = sorted(cache.monuments, key=lambda x: x[0], reverse=True)[:3]

# #############################################################################
# ##                  )══{≡≡≡≡≡≡≡≡>  RENDER   <≡≡≡≡≡≡≡≡}══(                  ##
# #############################################################################
# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
# ---------------- Spine Builder ----------------------------------------------
def _fmt_spine_cost(cost: int, width: int | None = None) -> str:
    if width is None:
        width = COST_W
    cost_raw = f"{int(cost):+,}"
    if not cost_raw.startswith("+"):
        cost_raw = "+" + cost_raw
    return cost_raw.rjust(width)

def build_spine_lines(
    cache: "UiCache",
    snap: "BoardSnapshot",
    action: str,
    *,
    ctx: "FrameCtx | None" = None,
    pal: dict[str, str],
    anchor_col: int,
) -> list[str]:
    """Return the boxed spine lines (content line + hline).
    This collapses the per-action/per-focus spine formatting into one place.
    Render() can stay clean and call this once per frame.
    """
    ash = pal["ash"]
    flame = pal["flame"]
    reset = pal["reset"]

    if ctx is None:
        ctx = build_ctx(cache, snap)

    # MENU browsing: centered preview
    if cache.focus == Focus.MENU:
        spine = _center_term(action_preview(action))
        return [ash + spine + reset, HLINE]

    # WHISPER: target selection / compose
    if action == Action.WHISPER and cache.focus in (Focus.TABLE_MOVE, Focus.SPINE):
        target = (
            (cache.local_name if cache.city_idx == getattr(cache, "me_idx", 0) else snap.cells[cache.city_idx].name)
            if cache.focus == Focus.TABLE_MOVE
            else (cache.whisper_target or (cache.local_name if cache.city_idx == getattr(cache, "me_idx", 0) else snap.cells[cache.city_idx].name))
        )
        name_fixed = target.ljust(NAME_W)[:NAME_W]
        cost_s = _fmt_spine_cost(cache.salt)

        if cache.focus == Focus.TABLE_MOVE:
            spine = f"{ash}{name_fixed}{reset} {ash}{cost_s}{reset}{ash}:{reset}"
            return [spine, HLINE]

        draft = str(cache.text or "").replace("\n", " ").replace("\r", " ")
        spine = f"{flame}{name_fixed}{reset} {cost_s}{ash}:{reset} {draft}"
        return [spine, HLINE]

    # WRATH: arm phase / compose
    if action == Action.WRATH and ctx.focus in (Focus.TABLE_LOCK, Focus.SPINE):
        label = (ctx.label or "LEGION")
        name_fixed = label.ljust(NAME_W)[:NAME_W]
        cost_s = _fmt_spine_cost(cache.salt)

        if cache.focus == Focus.TABLE_LOCK:
            spine = f"{ash}{name_fixed}{reset} {ash}{cost_s}{reset}{ash}:{reset}"
            return [spine, HLINE]

        draft = str(cache.text or "").replace("\n", " ").replace("\r", " ")
        spine = f"{flame}{name_fixed}{reset} {cost_s}{ash}:{reset} {draft}"
        return [spine, HLINE]

    # RALLY: arm phase / compose
    if action == Action.RALLY and cache.focus in (Focus.TABLE_LOCK, Focus.SPINE):
        label = (ctx.label or "MORALE")
        name_fixed = label.ljust(NAME_W)[:NAME_W]
        cost_s = _fmt_spine_cost(cache.salt)

        if cache.focus == Focus.TABLE_LOCK:
            spine = f"{ash}{name_fixed}{reset} {ash}{cost_s}{reset}{ash}:{reset}"
            return [spine, HLINE]

        draft = str(cache.text or "").replace("\n", " ").replace("\r", " ")
        spine = f"{flame}{name_fixed}{reset} {cost_s}{ash}:{reset} {draft}"
        return [spine, HLINE]

    # DEFECT: targeting phase spine
    if action == Action.DEFECT and cache.focus == Focus.TABLE_MOVE:
        defect_cost = ctx.floor
        label = (ctx.label or "COHORT")
        name_fixed = label.ljust(NAME_W)[:NAME_W]
        cost_s = _fmt_spine_cost(defect_cost)
        spine = f"{ash}{name_fixed}{reset} {ash}{cost_s}{reset}{ash}:{reset}"
        return [spine, HLINE]

    # Default seam: mode description line
    spine = action_preview(action)
    if action in ACTION_META and action_desc(action):
        floor = action_floor(action, cache)
        desc = action_desc(action)

        name_fixed = action.ljust(NAME_W)[:NAME_W]
        cost_raw = f"{floor:+,}"
        cost_field_w = max(len(cost_raw), anchor_col - (NAME_W + 1))
        cost_s = cost_raw.rjust(cost_field_w)
        spine = f"{name_fixed} {cost_s}:{desc}"

    return [ash + spine + reset, HLINE]

def _team_totals_bottom_bar(
    snap: "BoardSnapshot",
    width: int | None = None,
    *,
    pal: dict[str, str],
    thresh: int = PACK_RACE_THRESH_DEFAULT,
) -> str:

    if width is None:
        width = TERM_W

    ash = pal["ash"]
    ember = pal["ember"]
    flame = pal["flame"]
    reset = pal["reset"]

    # Sum salt per team (4 columns)
    totals = [0] * BOARD_COLS
    for c in snap.cells:
        if 0 <= c.team < BOARD_COLS:
            totals[c.team] += int(c.salt)

    # --- Pack race detection ("chain" pack):
    pairs = sorted([(totals[i], i) for i in range(BOARD_COLS)])  # (value, idx)
    hot_ember: set[int] = set()

    best_lo = 0
    best_hi = -1
    lo = 0
    for j in range(1, BOARD_COLS):
        if (pairs[j][0] - pairs[j - 1][0]) <= thresh:
            pass
        else:
            if (j - lo) > (best_hi - best_lo + 1):
                best_lo, best_hi = lo, j - 1
            lo = j
    if (BOARD_COLS - lo) > (best_hi - best_lo + 1):
        best_lo, best_hi = lo, BOARD_COLS - 1

    best_len = best_hi - best_lo + 1
    if best_len >= 3:
        hot_ember.update(pairs[k][1] for k in range(best_lo, best_hi + 1))

    # --- Exact ties override to FLAME ---
    hot_flame: set[int] = set()
    buckets: dict[int, list[int]] = {}
    for i, v in enumerate(totals):
        buckets.setdefault(v, []).append(i)
    for idxs in buckets.values():
        if len(idxs) >= 2:
            hot_flame.update(idxs)

    def fmt(i: int, v: int) -> str:
        s = f"{v:,}"
        if i in hot_flame:
            return f"{flame}{s}{reset}{ash}"
        if i in hot_ember:
            return f"{ember}{s}{reset}{ash}"
        return f"{ash}{s}{reset}{ash}"

    dot = f"{reset}•{ash}"
    center = f" {dot} ".join(fmt(i, totals[i]) for i in range(BOARD_COLS))

    pad_total = width - _vislen(center) - 2
    if pad_total < 2:
        center = center[: max(0, width - 4)]
        pad_total = width - _vislen(center) - 2

    bias = 2
    left = max(0, (pad_total // 2) - bias)
    right = pad_total - left

    return ("=" * left) + " " + center + " " + ("=" * right)

# ---------------- Render Function (PREVIEW ONLY - no mutations) --------------
def build_banner(pal: dict[str, str], title: str = "BYZANTIUM") -> list[str]:
    """Pure 3-line banner primitive (uses frame palette)."""
    flame = pal["flame"]
    flare = pal["flare"]
    ash = pal["ash"]
    white = pal["white"]
    reset = pal["reset"]

    top = flare + "+    " + reset
    bot = flame + "+    " + reset

    mid = (
        flare + "•  •  · " +
        ash   + ")══{≡≡≡≡≡≡≡≡>     " +
        flare + "+  " +
        white + title + reset +
        flame + "  +     " +
        ash   + "<≡≡≡≡≡≡≡≡}══( " +
        flame + "·  •  •    " +
        reset
    )
    return [top, mid, bot]

def render_lore_screen(cache: UiCache) -> str:
    pal = palette(cache)
    flame1 = pal["flame"]
    flame2 = pal["flare"]
    ash = pal["ash"]
    white = pal["white"]
    reset = pal["reset"]
    lines: List[str] = []
    lines.append(_clip_term(ash + HLINE + reset))
    lines.append(_clip_term(_center_term(ash + "" + reset)))
    lines.append(
        _clip_term(
            _center_term(
                flame1   + "·  •  •  " +
                white           + Action.LORE.value + reset +
                flame2   + "  •  •  ·"
            )
        )
    )
    lines.append(_clip_term(_center_term(ash + "" + reset)))
    _append_paragraph(
        lines,
        """
The city fell in unity. By dawn, unity was already a lie. Byzantium begins
in the hours after victory, while the streets still burn and every general
wonders which alliance will fail first.

The city did not fall into a new economy. It fell into one already
compromised. Long before the gates were breached, the empire’s currency had
been diluted, clipped, stretched thin by promises that no longer carried
weight. Salt alone remained valuable.

In Byzantium, even a whisper carries weight. Influence moves through
what is said, what is promised, and what is left unsaid. A word placed at
the right moment can fracture an alliance or harden into command.

Will you distribute influence to hold the city together, or hoard it until
it becomes leverage? Will you spend trust freely, or let it accumulate until
it turns brittle and breaks?

Consensus was never trust.
""".strip(),
        color=ash,
        reset=reset,
)
    lines.append("" )

    while len(lines) < BODY_FILL_LINES:
        lines.append(_clip_term(""))

    lines.append(_clip_term(ash + HLINE + reset))

    framed = []
    for line in lines:
        core = _clipw(line, INNER_W)
        framed.append(" " + core + " ")

    return "\n".join(framed) + reset

def render_title_screen(cache: UiCache) -> str:
    pal = palette(cache)
    flame1 = pal["flame"]
    flame2 = pal["flare"]
    ash = pal["ash"]
    white = pal["white"]
    reset = pal["reset"]

    name = (cache.text or "")
    name = name.replace("\n", " ").replace("\r", " ")
    # keep it readable + consistent with table NAME(8)
    name_vis = name[:NAME_W]
    cursor = flame1 + ":" + reset
    prompt = "Do Remember Who You Are?"
    hint = ""

    lines: List[str] = []
    lines.append(_clip_term(ash + HLINE + reset))
    lines.append(_clip_term(_center_term(ash + ""+ reset)))
    lines.append(_clip_term(_center_term(ash + ""+ reset)))
    lines.append(_clip_term(_center_term(ash + ""+ reset)))
    lines.append(_clip_term(_center_term(ash + "."+ reset)))
    lines.append(
        _clip_term(
            _center_term(
                ash   + "." + reset +
                flame1 + "+" + reset +
                ash   + "." + reset
                )
            )
        )
    lines.append(_clip_term(_center_term(ash + ".   .   .   ." + reset)))
    lines.append(
        _clip_term(
            _center_term(
                flame1   + "+ " + reset +
                white           + "BYZANTIUM" + reset +
                flame2   + " +"
                )
            )
        )
    lines.append(_clip_term(_center_term(ash + "·   · ·   · ·   ·" + reset)))
    lines.append(
        _clip_term(
            _center_term(
                ash           + "·" + reset +
                flame2 + "+" + reset +
                ash           + "·"
                )
            )
        )
    lines.append(_clip_term(_center_term(ash + "·" + reset)))
    lines.append(_clip_term(_center_term("")))
    lines.append(_clip_term(_center_term(ash + prompt + reset)))
    lines.append(_clip_term(_center_term("")))

    # Input line: centered box-like feel
    typed = (white + name_vis + reset) if name_vis else ""
    line = f"{cursor}{typed}{cursor}"
    lines.append(_clip_term(_center_term(line)))

    lines.append(_clip_term(_center_term("")))
    lines.append(_clip_term(_center_term(ash + hint + reset)))

    while len(lines) < BODY_FILL_LINES:
        lines.append(_clip_term(""))

    lines.append(_clip_term(ash + HLINE + reset))

    framed = []
    for line in lines:
        core = _clipw(line, INNER_W)
        framed.append(" " + core + " ")
    return "\n".join(framed) + reset

def render(cache: UiCache, snap: BoardSnapshot, ctx: FrameCtx | None = None) -> str:
    if cache.focus == Focus.TITLE:
        return render_title_screen(cache)
    if getattr(cache, 'show_lore', False):
        return render_lore_screen(cache)
    if ctx is None:
        ctx = build_ctx(cache, snap)
    # Compute anchor column for alignment from monuments
    mons = [m[1] for m in (cache.monuments or [])]
    anchor_col = _monument_anchor_col(mons, cache.local_name, name_w=NAME_W)
    # Palette (frame)
    pal = palette(cache)
    flame1 = pal["flame"]
    flame2 = pal["flare"]

    lines: List[str] = []

# #############################################################################
# ##               )══{≡≡≡≡≡≡≡≡>   CITY LIMIT   <≡≡≡≡≡≡≡≡}══(                ##
# #############################################################################
# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
# =============================================================================
    # MENU BAND (boxed)
# =============================================================================
    action = MENU[cache.menu_idx]

    menu_chunks = []
    for i, act in enumerate(MENU):
        if i == cache.menu_idx:
            menu_chunks.append(f"{flame2}{act.value}{RESET}")
        else:
            menu_chunks.append(f"{ASH}{act.value}{RESET}")

    leader = (cache.local_name or "").ljust(NAME_W)[:NAME_W]
    menu_line = f"{flame1}{leader}{RESET} {flame1}>>>{RESET} " + " • ".join(menu_chunks)

    lines.append(_clip_term(ASH + HLINE + RESET))
    lines.append(_clip_term(menu_line))
    lines.append(_clip_term(ASH + HLINE + RESET))

# =============================================================================
# MONUMENT REGION (3 lines, under menu band)
# =============================================================================
    if getattr(cache, "show_banner", True):
        for line in build_banner(pal, title="BYZANTIUM"):
            lines.append(_clip_term(_center_term(line)))
    else:
        mons = [m[1] for m in (cache.monuments or [])]
        while len(mons) < 3:
            mons.append("...")
        for m in mons[:3]:
            lines.append(_clip_term(ASH + m + RESET))
    # box the city off
    lines.append(_clip_term(ASH + HLINE + RESET))

# =============================================================================
    # CITY BOARD (4 columns x 6 rows)
# =============================================================================
    def fmt_cell(idx: int) -> str:
        c = snap.cells[idx]

        base_name = cache.local_name if idx == ctx.me_idx else c.name
        name = base_name.ljust(NAME_W)[:NAME_W]
        salt = f"{c.salt:,}".rjust(TABLE_SALT_W)


        if action == Action.WRATH and ctx.focus in (Focus.TABLE_LOCK, Focus.SPINE):
            name_col = flame1
        elif (
            (ctx.focus == Focus.TABLE_MOVE and idx == ctx.city_idx)
            or
            (action == Action.WHISPER and ctx.focus == Focus.SPINE and idx == ctx.target_idx)
):
            name_col = flame1
        elif action == Action.RALLY and ctx.focus in (Focus.TABLE_LOCK, Focus.SPINE):
            # RALLY: your whole banner (column) ignites.
            me = cache.me_idx
            if (idx // BOARD_ROWS) == (me // BOARD_ROWS):
                name_col = flame1
            else:
                name_col = ASH
        elif action == Action.DEFECT and ctx.focus == Focus.TABLE_MOVE:
            # DEFECT targeting vision (single-source viability):

            me = cache.me_idx
            if idx == me:
                name_col = flame1
            elif defect_is_viable(snap, me, idx):
                name_col = EMBER
            else:
                name_col = ASH
        elif idx == ctx.me_idx:
        
            # Ownership beacon
            name_col = flame1 if (idx % BOARD_ROWS) == 0 else SALT
        elif (idx % BOARD_ROWS) == 0:
            name_col = EMBER
        else:
            name_col = ASH

        return f"{name_col}{name}{RESET} {salt}"

    for r in range(BOARD_ROWS):
        row_parts = []
        for col in range(BOARD_COLS):
            idx = col * BOARD_ROWS + r
            row_parts.append(fmt_cell(idx))
        row = (" " * 4).join(row_parts) + "  "
        lines.append(_clip_term(row))

    # bottom border for city
    lines.append(_clip_term(HLINE))

# =============================================================================
    # SPINE (boxed)
# =============================================================================
    spine_lines = build_spine_lines(cache, snap, action, ctx=ctx, pal=pal, anchor_col=anchor_col)
    for ln in spine_lines:
        lines.append(_clip_term(ln))

# =============================================================================
    # ASHFALL FEED (from live event log, colored)
# =============================================================================
    # Render live feed from bottom up (newest visible)
    vis = min(cache.visible_feed_count, len(cache.feed))
    feed_start = max(0, len(cache.feed) - vis)
    for chan, line in reversed(cache.feed[feed_start:]):
        fc = chan_color(pal, chan)
        lines.append(_clip_term(fc + str(line) + RESET))

    while len(lines) < BODY_FILL_LINES:
        lines.append(_clip_term(""))

    # bottom cap
    #team totals etched into the rail
    totals_bar = _team_totals_bottom_bar(snap, width=TERM_W, pal=pal)
    lines.append(_clip_term(ASH + totals_bar + RESET))

    # Frame all lines with left/right padding
    framed = []
    for line in lines:
        core = _clipw(line, INNER_W)
        framed.append(" " + core + " ")

    return "\n".join(framed) + RESET

# #############################################################################
# ##                )══{≡≡≡≡≡≡≡≡>   MAIN.LOOP   <≡≡≡≡≡≡≡≡}══(                ##
# #############################################################################
# ↓     ↓      ↓      ↓      ↓      ↓       ↓      ↓      ↓      ↓      ↓     ↓
def main():
    FRAME_DT = 1.0 / FPS

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        snap = genesis_snapshot("SATOSHI")
        _normalize_columns(snap)

        cache = UiCache(feed=[], local_name="SATOSHI", monuments=[])
        cache.focus = Focus.TITLE
        cache.text = ""
        cache.snap = snap
        i_me = _find_by_id4(snap, cache.me_id4)
        if i_me is not None:
            cache.me_idx = i_me

        cache.me_id4 = snap.cells[cache.me_idx].id4

        tick = 0
        should_quit = False

        def _parse_keys_buffered(buf: str):
            """Parse buf into tokens, returning (tokens, remaining_buf)."""
            out = []
            i = 0
            L = len(buf)
            while i < L:
                c = buf[i]
                if c == "\x03":
                    out.append(("CTRL_C", None)); i += 1; continue
                if c == "\n":
                    out.append(("ENTER", None)); i += 1; continue
                if c in ("\x7f", "\b"):
                    out.append(("BS", None)); i += 1; continue
                if c == "\x1b":
                    # Need at least 2 more bytes to decide.
                    if i + 1 >= L:
                        break  # keep ESC for next frame
                    n1 = buf[i+1]
                    # SS3 arrows: ESC O A/B/C/D
                    if n1 == "O":
                        if i + 2 >= L:
                            break
                        d = buf[i+2]
                        if d in ("A","B","C","D"):
                            out.append(("ARROW", d)); i += 3; continue
                        # Unknown SS3; consume ESC and continue.
                        i += 1; continue
                    # CSI arrows: ESC [ ... A/B/C/D
                    if n1 == "[":
                        j = i + 2
                        # scan until final byte or until we run out
                        while j < L:
                            d = buf[j]
                            if d in ("A","B","C","D"):
                                out.append(("ARROW", d))
                                i = j + 1
                                break
                            if d == "~":
                                i = j + 1
                                break
                            j += 1
                        else:
                            break  # incomplete CSI; keep for next frame
                        continue
                    # Unknown ESC; consume ESC
                    i += 1
                    continue

                out.append(("CH", c))
                i += 1
            return out, buf[i:]

        def _drain_stdin() -> List[str]:
            """Read and return ALL available input bytes as a list of chars."""
            out: List[str] = []
            # Read in chunks while stdin is ready; avoid backlog.
            while True:
                r, _w, _e = select([fd], [], [], 0)
                if not r:
                    break
                try:
                    b = os.read(fd, 4096)
                except BlockingIOError:
                    break
                if not b:
                    break
                out.extend([chr(x) for x in b])
            return out

        buf = ""

        # ---------------- Controls Bucket ------------------------------------
        # A single dispatcher that routes input tokens based on focus/mode.
        # ---------------------------------------------------------------------
        def _reset_to_menu():
            cache.focus = Focus.MENU
            cache.menu_idx = 0
            cache.salt = 1
            cache.text = ""
            cache.whisper_target = ""
            cache.target_idx = None

        def _feed_prune():
            # Keep only the newest ASHFALL_MAX entries.
            if len(cache.feed) > ASHFALL_MAX:
                cache.feed = cache.feed[-ASHFALL_MAX:]

        def _feed_norm_entry(chan, speaker=None, payload=None):

            if speaker is None and payload is None and isinstance(chan, tuple) and len(chan) == 3:
                chan, speaker, payload = chan

            # Event form: (cost, msg) -> fixed-width canonical feed line.
            if isinstance(payload, tuple) and len(payload) == 2:
                cost, msg = payload
                line = _fmt_feed_line(str(speaker or ""), int(cost), str(msg))
                return (chan, line)

            # Raw/system line: speaker empty -> payload becomes the whole line.
            if speaker is None or speaker == "":
                return (chan, str(payload or ""))

            # Fallback: legacy "speaker: payload".
            return (chan, f"{speaker}: {payload}")

        def _feed_push(chan, speaker=None, payload=None, *, reveal: bool = True, kick: bool = True):
            """Append one feed record in canonical form (chan, line), prune, and optionally reveal/kick."""
            cache.feed.append(_feed_norm_entry(chan, speaker, payload))
            _feed_prune()
            if reveal:
                cache.visible_feed_count = len(cache.feed)
            if kick:
                cache.flame_fed = True

        def _feed_clear(*, kick: bool = True):
            """Clear the live feed and reset visibility."""
            cache.feed = []
            cache.visible_feed_count = 0
            if kick:
                cache.flame_fed = True

        def _feed_clear_with_residue(items: list[tuple], *, reveal_count: int | None = None, kick: bool = True):
            """Clear feed, then seed it with `items`.

            `items` may be either:
              - canonical (chan, line)
              - legacy (chan, speaker, payload)
            """
            _feed_clear(kick=False)
            for it in items:
                # accept either 2-tuple canonical or 3-tuple legacy
                if isinstance(it, tuple) and len(it) == 2:
                    chan, line = it
                    cache.feed.append((chan, str(line)))
                else:
                    cache.feed.append(_feed_norm_entry(it))
            _feed_prune()
            cache.visible_feed_count = len(cache.feed) if reveal_count is None else int(reveal_count)
            if kick:
                cache.flame_fed = True

        def _handle_title(kind, payload):
            if kind == "ENTER":
                nm = (cache.text or "").strip() or "SATOSHI"
                nm = nm[:NAME_W]
                new_snap = genesis_snapshot(nm)
                _normalize_columns(new_snap)
                cache.snap = new_snap
                cache.local_name = nm
                cache.me_idx = _find_by_name(new_snap, cache.local_name)
                cache.city_idx = cache.me_idx
                cache.me_id4 = new_snap.cells[cache.me_idx].id4
                cache.city_idx = cache.me_idx
                cache.menu_idx = 0
                cache.visible_feed_count = 0
                cache.feed = []
                cache.monuments = cache.monuments or []
                cache.text = ""
                cache.focus = Focus.MENU
                cache.show_banner = True
                cache.show_lore = False
                cache.flame_fed = True
                return False

            if kind == "BS":
                cache.text = (cache.text or "")[:-1]
                return False

            if kind == "CH":
                ch = payload
                o = ord(ch)
                if 32 <= o <= 126:
                    cache.text = (cache.text or "")
                    if len(cache.text) < 12:
                        cache.text += ch
                return False

            # Ignore everything else on the title screen
            return False

        def _handle_lore(kind, payload):
            if kind == "ENTER" or (kind == "CH" and payload == " "):
                cache.show_lore = False
                cache.flame_fed = True
            return False

        def _handle_menu(kind, payload, action):
            if kind == "ARROW":
                d = payload
                if d == "C":      # right
                    cache.menu_idx = (cache.menu_idx + 1) % len(MENU)
                elif d == "D":    # left
                    cache.menu_idx = (cache.menu_idx - 1) % len(MENU)
                return False

            if kind != "ENTER":
                return False

            # ENTER in MENU
            if action == Action.EXIT:
                return True

            if action == Action.PURGE:
                cache.text = ""
                cache.show_banner = True
                cache.show_lore = False

                _feed_clear_with_residue(
                    [
                        (CHAN_ASH, _center_term("Only truth remains...")),
                        (CHAN_ASH, ""),
                        (CHAN_ASH, ""),
                        (CHAN_ASH, ""),
                    ],
                    reveal_count=4,
                )
                cache.pending_request = "DREAM"
                return False
            if action == Action.MONUMENT:
                cache.show_banner = not getattr(cache, "show_banner", True)
                cache.flame_fed = True
                return False

            if action == Action.LORE:
                cache.show_lore = not getattr(cache, "show_lore", False)
                cache.flame_fed = True
                return False

            if action == Action.WRATH:
                cache.focus = Focus.TABLE_LOCK
                cache.salt = action_floor(action, cache)
                cache.text = ""
                return False

            if action == Action.WHISPER:
                cache.focus = Focus.TABLE_MOVE
                cache.whisper_target = ""
                cache.salt = 1
                cache.text = ""
                # Start targeting 
                me = cache.me_idx
                cache.city_idx = (me + 1) % CELL_COUNT
                return False

            if action == Action.RALLY:
                cache.focus = Focus.TABLE_LOCK
                cache.salt = action_floor(action, cache)
                cache.text = ""
                return False

            if action == Action.DEFECT:
                cache.focus = Focus.TABLE_MOVE
                cache.salt = action_floor(action, cache)

                # Start on a viable target 
                start_idx = cache.city_idx
                if start_idx == cache.me_idx:
                    start_idx = (start_idx + 1) % len(snap.cells)
                tgt = _snap_defect_cursor(snap, cache.me_idx, start_idx)
                cache.city_idx = tgt if tgt is not None else cache.me_idx
                return False

            return False

        def _handle_table_lock(kind, payload, action):
            # SPACE back-out handled globally
            if kind != "ENTER":
                return False

            if action == Action.WRATH:
                cache.focus = Focus.SPINE
                cache.text = ""
                return False

            if action == Action.RALLY:
                cache.focus = Focus.SPINE
                cache.text = ""
                return False

            return False

        def _handle_table_move(kind, payload, action):
            # SPACE back-out handled globally
            if kind == "ARROW":
                d = payload
                old_idx = cache.city_idx
                row = old_idx % BOARD_ROWS
                col = old_idx // BOARD_ROWS

                if d == "A":      # up
                    row2, col2 = (row - 1) % BOARD_ROWS, col
                elif d == "B":    # down
                    row2, col2 = (row + 1) % BOARD_ROWS, col
                elif d == "D":    # left
                    row2, col2 = row, (col - 1) % BOARD_COLS
                elif d == "C":    # right
                    row2, col2 = row, (col + 1) % BOARD_COLS
                else:
                    row2, col2 = row, col

                cand = col2 * BOARD_ROWS + row2

                if action == Action.DEFECT:
                    me = cache.me_idx
                    step_r, step_c = 0, 0
                    if d == "A":
                        step_r = -1
                    elif d == "B":
                        step_r = +1
                    elif d == "D":
                        step_c = -1
                    elif d == "C":
                        step_c = +1

                    r, c = row2, col2
                    found = None
                    for _ in range(CELL_COUNT):
                        idx2 = c * BOARD_ROWS + r
                        if _defect_viable(snap, me, idx2):
                            found = idx2
                            break
                        r = (r + step_r) % BOARD_ROWS
                        c = (c + step_c) % BOARD_COLS

                    cache.city_idx = found if found is not None else old_idx
                else:
                    # WHISPER cannot target yourself
                    if action == Action.WHISPER:
                        me = cache.me_idx
                        if cand == me:
                            step_r, step_c = 0, 0
                            if d == "A":
                                step_r = -1
                            elif d == "B":
                                step_r = +1
                            elif d == "D":
                                step_c = -1
                            elif d == "C":
                                step_c = +1
                            r, c = row2, col2
                            for _ in range(CELL_COUNT):
                                r = (r + step_r) % BOARD_ROWS
                                c = (c + step_c) % BOARD_COLS
                                idx2 = c * BOARD_ROWS + r
                                if idx2 != me:
                                    cand = idx2
                                    break
                    cache.city_idx = cand
                return False

            if kind != "ENTER":
                return False

            # ENTER in TABLE_MOVE
            if action == Action.WHISPER:
                # Disallow whispering to yourself
                me = cache.me_idx
                if cache.city_idx == me:
                    cache.city_idx = (me + 1) % CELL_COUNT
                cache.target_idx = cache.city_idx
                cache.whisper_target = (cache.local_name if cache.city_idx == me else snap.cells[cache.city_idx].name)
                cache.focus = Focus.SPINE
                cache.salt = 1
                cache.text = ""
                return False

            if action == Action.DEFECT:
                me = cache.me_idx
                tgt = cache.city_idx
                if tgt == me:
                    _reset_to_menu()
                    return False

                if snap.cells[tgt].salt >= snap.cells[me].salt:
                    _feed_push(CHAN_ASH, "ASH", (0, "target too strong"))
                    _reset_to_menu()
                    return False

                defect_cost = 10000 if (me % BOARD_ROWS) == 0 else 500
                have = snap.cells[me].salt
                if have < defect_cost:
                    _feed_push((CHAN_ASH, "ASH", (0, "insufficient salt")))
                    _reset_to_menu()
                    return False

                old_col = me // BOARD_ROWS
                old_general = old_col * BOARD_ROWS
                if me == old_general:
                    recipients = [old_col * BOARD_ROWS + r for r in range(BOARD_ROWS) if (old_col * BOARD_ROWS + r) != me]
                else:
                    recipients = [old_col * BOARD_ROWS + r for r in range(BOARD_ROWS)]
                    recipients = [i for i in recipients if i != me and i != old_general]

                spent = _distribute_to_weakest_first(snap, me, recipients, defect_cost)
                if spent <= 0:
                    _feed_push((CHAN_ASH, "ASH", (0, "defect failed")))
                    _reset_to_menu()
                    return False

                snap.cells[me], snap.cells[tgt] = snap.cells[tgt], snap.cells[me]
                cache.me_idx = tgt
                cache.city_idx = tgt
                _normalize_columns(snap)
                i_me = _find_by_id4(snap, cache.me_id4)
                if i_me is not None:
                    cache.me_idx = i_me

                cache.city_idx = cache.me_idx
                _recompute_cell_flags(snap)
                _feed_push((CHAN_EMBER, cache.local_name, (spent, "defected")))
                _maybe_inscribe(cache, cache.local_name, spent, "defected")

                _reset_to_menu()
                return False

            return False

        def _handle_spine(kind, payload, action):
            if kind == "ARROW":
                d = payload
                floor = action_floor(action, cache)
                me = cache.me_idx
                have = snap.cells[me].salt if (0 <= me < len(snap.cells)) else 0

                # Cap the adjustable amount to what the player actually has.
                # If have < floor, keep the display at floor (commit will reject anyway).
                cap_max = have if have >= floor else floor

                if d == "A":
                    cache.salt = min(cap_max, int(cache.salt) + floor)
                elif d == "B":
                    cache.salt = max(floor, int(cache.salt) - floor)
                return False

            if kind == "BS":
                if action in (Action.WHISPER, Action.WRATH, Action.RALLY):
                    cache.text = (cache.text or "")[:-1]
                return False

            if kind == "CH":
                if action in (Action.WHISPER, Action.WRATH, Action.RALLY):
                    ch = payload
                    o = ord(ch)
                    if 32 <= o <= 126:
                        cache.text = (cache.text or "")
                        if len(cache.text) < 59:
                            cache.text += ch
                return False

            if kind != "ENTER":
                return False

            # ENTER commits while in SPINE
            if action == Action.WHISPER:
                floor = action_floor(action, cache)
                msg = (cache.text or "").strip() or "..."
                me = cache.me_idx
                tgt = cache.target_idx if cache.target_idx is not None else cache.city_idx

                have = snap.cells[me].salt
                if have < floor:
                    _feed_push(CHAN_ASH, "ASH", (0, "insufficient salt"))
                    return False

                cost = max(floor, int(cache.salt or floor))
                cost = min(cost, have)

                snap.cells[me] = replace(snap.cells[me], salt=snap.cells[me].salt - cost)
                snap.cells[tgt] = replace(snap.cells[tgt], salt=snap.cells[tgt].salt + cost)

                _normalize_columns(snap)
                i_me = _find_by_id4(snap, cache.me_id4)
                if i_me is not None:
                    cache.me_idx = i_me

                _feed_push((CHAN_ASH, cache.local_name, (cost, msg)))
                _maybe_inscribe(cache, cache.local_name, cost, msg)

                _reset_to_menu()
                return False

            if action == Action.WRATH:
                floor = action_floor(action, cache)
                msg = (cache.text or "").strip() or "..."
                me = cache.me_idx
                have = snap.cells[me].salt
                if have < floor:
                    _feed_push((CHAN_ASH, "ASH", (0, "insufficient salt")))
                    return False

                want = max(floor, int(cache.salt or floor))
                want = min(want, have)
                recipients = [i for i in range(len(snap.cells)) if i != me]

                spent = _distribute_to_weakest_first(snap, me, recipients, want)

                _normalize_columns(snap)
                i_me = _find_by_id4(snap, cache.me_id4)
                if i_me is not None:
                    cache.me_idx = i_me

                if spent <= 0:
                    _feed_push((CHAN_ASH, "ASH", (0, "wrath failed")))
                else:
                    _feed_push(CHAN_FLAME, cache.local_name, (spent, msg))
                    _maybe_inscribe(cache, cache.local_name, spent, msg)

                _reset_to_menu()
                return False

            if action == Action.RALLY:
                floor = action_floor(action, cache)
                msg = (cache.text or "").strip() or "..."
                me = cache.me_idx
                have = snap.cells[me].salt
                if have < floor:
                    _feed_push(CHAN_ASH, "ASH", (0, "insufficient salt"))
                    return False

                want = max(floor, int(cache.salt or floor))
                want = min(want, have)

                col = me // BOARD_ROWS
                team_idxs = [col * BOARD_ROWS + r for r in range(BOARD_ROWS) if (col * BOARD_ROWS + r) != me]
                if team_idxs:
                    share = want // len(team_idxs)
                    if share <= 0:
                        _feed_push(CHAN_ASH, "ASH", (0, "rally failed"))
                    else:
                        spent = share * len(team_idxs)
                        snap.cells[me] = replace(snap.cells[me], salt=snap.cells[me].salt - spent)
                        for j in team_idxs:
                            snap.cells[j] = replace(snap.cells[j], salt=snap.cells[j].salt + share)
                        _feed_push(CHAN_EMBER, cache.local_name, (spent, msg))
                        _maybe_inscribe(cache, cache.local_name, spent, msg)

                _normalize_columns(snap)
                i_me = _find_by_id4(snap, cache.me_id4)
                if i_me is not None:
                    cache.me_idx = i_me

                _reset_to_menu()
                return False

            return False

        _FOCUS_HANDLERS = {
            Focus.MENU: _handle_menu,
            Focus.TABLE_MOVE: _handle_table_move,
            Focus.TABLE_LOCK: _handle_table_lock,
            Focus.SPINE: _handle_spine,
        }

        def _dispatch_token(kind, payload, action):
            # Global quit
            if kind == "CTRL_C":
                return True

            # Title screen
            if cache.focus == Focus.TITLE:
                return _handle_title(kind, payload)

            # Lore 
            if getattr(cache, "show_lore", False):
                return _handle_lore(kind, payload)

            # Global SPACE back-out
            if kind == "CH" and payload == " ":
                if action in (Action.WHISPER, Action.DEFECT) and cache.focus == Focus.TABLE_MOVE:
                    _reset_to_menu()
                    return False
                if action in (Action.RALLY, Action.WRATH) and cache.focus == Focus.TABLE_LOCK:
                    _reset_to_menu()
                    return False

            # Default: route to focus handler
            h = _FOCUS_HANDLERS.get(cache.focus)
            if h is None:
                return False
            return h(kind, payload, action)

        while True:
            frame_t0 = time.monotonic()

            action = MENU[cache.menu_idx]

            # Drain input
            chars = _drain_stdin()
            buf += "".join(chars)
            tokens, buf = _parse_keys_buffered(buf)

            for kind, payload in tokens:
                    if _dispatch_token(kind, payload, action):
                        should_quit = True
                        break

            if should_quit:
                break

            # Flame flicker:
            action = MENU[cache.menu_idx]

            if cache.flame_fed:
                cache.flame_phase ^= 1
                cache.flame_fed = False
            elif (tick % 10) == 0:   # tune 6..14
                cache.flame_phase ^= 1

            ctx = build_ctx(cache, snap)
            print("\x1b[H\x1b[2J" + render(cache, snap, ctx), end="", flush=True)

            tick += 1

            # Frame pacing: wait, but wake early if input arrives.
            remaining = FRAME_DT - (time.monotonic() - frame_t0)
            if remaining > 0:
                select([fd], [], [], remaining)

    except KeyboardInterrupt:
        # Graceful exit (CTRL-C)
        pass

    finally:
        print("\x1b[1A\x1b[2K", end="")
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print("\x1b[0m")

if __name__ == "__main__":
    main()
