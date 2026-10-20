"""Shared helpers: DPI awareness, config, monitors and board geometry.

Import this module BEFORE mss / pyautogui / pynput. It makes the process
per-monitor DPI aware on import, so screen-capture pixels and mouse
coordinates are both physical pixels on every monitor.
"""
import ctypes
import json
from ctypes import wintypes
from pathlib import Path


def _enable_dpi_awareness():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except (AttributeError, OSError):
        ctypes.windll.user32.SetProcessDPIAware()


_enable_dpi_awareness()

user32 = ctypes.windll.user32

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
CONFIG_NAME = "config.json"

DEFAULT_CONFIG = {
    "engine_path": "engine/fairy-stockfish-largeboard_x86-64-bmi2.exe",
    "movetime_ms": 1000,
    "engine_threads": 2,
    "engine_hash_mb": 128,
    "depth_limit": 0,              # 0 = no limit
    "multipv": 3,                  # lines shown in the GUI (1-5)
    "my_color": "red",
    "move_mode": "click",          # "click" (click-click) or "drag"
    "click_delay_s": 0.15,
    "restore_cursor": True,        # put the mouse back where it was after each move
    "stable_frames": 4,            # board must be unchanged this many captures in a row
    "frame_interval_s": 0.12,
    "confirm_timeout_s": 4.0,      # how long to wait for my move to appear on screen
    "hotkeys": {"capture": "f8", "pause": "f9", "quit": "f10"},
    "templates_dir": "templates",
    "monitor": None,
    "grid": None,                  # x0, y0, dx, dy in absolute screen pixels
    "board_region": None,          # left, top, width, height in absolute screen pixels
    "patch_size": None,
}


def load_config(base_dir=PROJECT_DIR):
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    path = Path(base_dir) / CONFIG_NAME
    if path.exists():
        cfg.update(json.loads(path.read_text(encoding="utf-8")))
    return cfg


def save_config(cfg, base_dir=PROJECT_DIR):
    path = Path(base_dir) / CONFIG_NAME
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def resolve_path(p, base_dir=PROJECT_DIR):
    p = Path(p)
    return p if p.is_absolute() else Path(base_dir) / p


# --------------------------------------------------------------------------
# Monitors (absolute virtual-desktop coordinates; may be negative)
# --------------------------------------------------------------------------

class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32)]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _PathSource(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", ctypes.c_uint32),
                ("modeInfoIdx", ctypes.c_uint32), ("statusFlags", ctypes.c_uint32)]


class _PathTarget(ctypes.Structure):
    _fields_ = [("adapterId", _LUID), ("id", ctypes.c_uint32),
                ("modeInfoIdx", ctypes.c_uint32), ("outputTechnology", ctypes.c_uint32),
                ("rotation", ctypes.c_uint32), ("scaling", ctypes.c_uint32),
                ("refreshNum", ctypes.c_uint32), ("refreshDen", ctypes.c_uint32),
                ("scanLineOrdering", ctypes.c_uint32), ("targetAvailable", wintypes.BOOL),
                ("statusFlags", ctypes.c_uint32)]


class _PathInfo(ctypes.Structure):
    _fields_ = [("sourceInfo", _PathSource), ("targetInfo", _PathTarget),
                ("flags", ctypes.c_uint32)]


class _ModeInfo(ctypes.Structure):
    _fields_ = [("infoType", ctypes.c_uint32), ("id", ctypes.c_uint32),
                ("adapterId", _LUID), ("data", ctypes.c_uint64 * 6)]


class _DeviceInfoHeader(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("size", ctypes.c_uint32),
                ("adapterId", _LUID), ("id", ctypes.c_uint32)]


class _SourceName(ctypes.Structure):
    _fields_ = [("header", _DeviceInfoHeader), ("viewGdiDeviceName", wintypes.WCHAR * 32)]


class _TargetName(ctypes.Structure):
    _fields_ = [("header", _DeviceInfoHeader), ("flags", ctypes.c_uint32),
                ("outputTechnology", ctypes.c_uint32), ("edidManufactureId", ctypes.c_uint16),
                ("edidProductCodeId", ctypes.c_uint16), ("connectorInstance", ctypes.c_uint32),
                ("monitorFriendlyDeviceName", wintypes.WCHAR * 64),
                ("monitorDevicePath", wintypes.WCHAR * 128)]


def _friendly_monitor_names():
    """Map GDI device name (\\\\.\\DISPLAY1) -> EDID name ("Xiaomi ...")."""
    QDC_ONLY_ACTIVE_PATHS = 2
    n_paths, n_modes = ctypes.c_uint32(), ctypes.c_uint32()
    if user32.GetDisplayConfigBufferSizes(QDC_ONLY_ACTIVE_PATHS, ctypes.byref(n_paths),
                                          ctypes.byref(n_modes)) != 0:
        return {}
    paths = (_PathInfo * n_paths.value)()
    modes = (_ModeInfo * n_modes.value)()
    if user32.QueryDisplayConfig(QDC_ONLY_ACTIVE_PATHS, ctypes.byref(n_paths), paths,
                                 ctypes.byref(n_modes), modes, None) != 0:
        return {}
    names = {}
    for p in paths[:n_paths.value]:
        src = _SourceName()
        src.header.type, src.header.size = 1, ctypes.sizeof(src)
        src.header.adapterId, src.header.id = p.sourceInfo.adapterId, p.sourceInfo.id
        tgt = _TargetName()
        tgt.header.type, tgt.header.size = 2, ctypes.sizeof(tgt)
        tgt.header.adapterId, tgt.header.id = p.targetInfo.adapterId, p.targetInfo.id
        if (user32.DisplayConfigGetDeviceInfo(ctypes.byref(src)) == 0
                and user32.DisplayConfigGetDeviceInfo(ctypes.byref(tgt)) == 0):
            names[src.viewGdiDeviceName] = tgt.monitorFriendlyDeviceName
    return names


def list_monitors():
    """Return monitors as dicts with absolute left/top/width/height, sorted left-to-right."""
    found = []
    proc_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
                                   ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

    def callback(hmon, _hdc, _rect, _lparam):
        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        user32.GetMonitorInfoW(hmon, ctypes.byref(info))
        r = info.rcMonitor
        found.append({"device": info.szDevice, "left": r.left, "top": r.top,
                      "width": r.right - r.left, "height": r.bottom - r.top,
                      "primary": bool(info.dwFlags & 1)})
        return True

    user32.EnumDisplayMonitors(None, None, proc_type(callback), 0)
    try:
        names = _friendly_monitor_names()
    except OSError:
        names = {}
    found.sort(key=lambda m: (m["left"], m["top"]))
    for i, m in enumerate(found, 1):
        m["index"] = i
        m["name"] = names.get(m["device"]) or m["device"]
    return found


def point_in_monitor(x, y, mon):
    return (mon["left"] <= x < mon["left"] + mon["width"]
            and mon["top"] <= y < mon["top"] + mon["height"])


def cursor_pos():
    pt = wintypes.POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


# --------------------------------------------------------------------------
# Board geometry
#
# Board coordinates: board[row][col], row 0 = Black's back rank (FEN rank 10),
# row 9 = Red's back rank (rank 1), col 0 = file 'a' (Red's left).
# Screen cells: (sr, sc) with sr 0 = top row on screen, sc 0 = leftmost column.
# When I play Red the two coincide; when I play Black the board is rotated 180.
# --------------------------------------------------------------------------

FILES = "abcdefghi"
START_BOARD_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR"
PIECE_NAMES = {"k": "king", "a": "advisor", "b": "elephant", "n": "horse",
               "r": "chariot", "c": "cannon", "p": "pawn"}


def piece_label(ch):
    return ("red_" if ch.isupper() else "black_") + PIECE_NAMES[ch.lower()]


def board_from_fen(fen_board):
    rows = []
    for rank in fen_board.split()[0].split("/"):
        row = []
        for ch in rank:
            row.extend(["."] * int(ch) if ch.isdigit() else [ch])
        rows.append(row)
    return rows


def square_name(row, col):
    return f"{FILES[col]}{10 - row}"


def board_to_cell(row, col, my_color):
    """Board (row, col) -> screen cell (sr, sc). Its own inverse."""
    return (row, col) if my_color == "red" else (9 - row, 8 - col)


def cell_center(sr, sc, grid):
    return grid["x0"] + sc * grid["dx"], grid["y0"] + sr * grid["dy"]


def board_to_screen(row, col, cfg):
    sr, sc = board_to_cell(row, col, cfg["my_color"])
    return cell_center(sr, sc, cfg["grid"])


# --------------------------------------------------------------------------
# Patch comparison
# --------------------------------------------------------------------------

DISC_RADIUS = 0.34   # fraction of patch size: the glyph area inside the piece's rim
SHIFT_TOLERANCE = 2  # pixels of misalignment tolerated in each direction


def make_disc_template(template):
    """Precompute (core, mask) for disc_score from a square BGR/gray template."""
    import cv2
    import numpy as np
    if template.ndim == 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    n, m = template.shape[0], SHIFT_TOLERANCE
    c = n - 2 * m
    mask = np.zeros((c, c), np.uint8)
    cv2.circle(mask, (c // 2, c // 2), int(DISC_RADIUS * n), 255, -1)
    return template[m:m + c, m:m + c].copy(), mask


def disc_score(patch_gray, disc_template):
    import cv2
    core, mask = disc_template
    return float(cv2.matchTemplate(patch_gray, core, cv2.TM_CCOEFF_NORMED, mask=mask).max())


def disc_similarity(patch, template):
    """Correlation of the template's inner disc against `patch`, best over small shifts.

    Only the inner disc is compared, so board lines, palace diagonals and
    highlight overlays around the piece don't affect the score. Both images
    are same-size BGR or grayscale patches.
    """
    import cv2
    if patch.ndim == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return disc_score(patch, make_disc_template(template))
