"""One-time calibration.

1. Pick the monitor and your colour.
2. Hover the TOP-LEFT intersection as seen on screen, press the capture key (F8);
   then the BOTTOM-RIGHT intersection, press it again. Esc aborts.
3. The grid is refined by detecting the 32 pieces of the starting position,
   piece templates and empty-square patches are saved to templates/, and
   everything goes into config.json. Check calibration_preview.png afterwards.

The board must be in the starting position with no move highlight.

Offline test against a screenshot (image pixel coordinates):
    python calibrate.py --image start.png --tl 875,168 --br 1373,672 --color black --out somedir
"""
import argparse
import shutil
import sys
import threading
from pathlib import Path

import common  # must be imported first: sets DPI awareness

import cv2
import mss
import numpy as np
from pynput import keyboard


# ------------------------------------------------------------------ input --

def parse_key(name):
    try:
        return keyboard.Key[name.lower()]
    except KeyError:
        return keyboard.KeyCode.from_char(name.lower())


class CalibrationError(RuntimeError):
    pass


def wait_for_hotkey(name, cancel=None):
    """Block until `name` is pressed anywhere; return the cursor position then.

    Esc, or setting the `cancel` Event, raises CalibrationError.
    """
    target, result, done = parse_key(name), {}, threading.Event()

    def on_press(key):
        if key == keyboard.Key.esc:
            result["abort"] = True
        elif key == target or (isinstance(key, keyboard.KeyCode) and key.char
                               and key.char.lower() == getattr(target, "char", None)):
            result["pos"] = common.cursor_pos()
        else:
            return None
        done.set()
        return False

    with keyboard.Listener(on_press=on_press):
        while not done.wait(0.1):
            if cancel is not None and cancel.is_set():
                raise CalibrationError("Calibration cancelled.")
    if result.get("abort"):
        raise CalibrationError("Calibration aborted (Esc).")
    return result["pos"]


def ask(prompt, choices, default=None):
    while True:
        ans = input(f"{prompt} ").strip().lower() or default
        if ans in choices:
            return ans
        print(f"  Please answer one of: {', '.join(choices)}")


def choose_monitor(monitors):
    print("\nMonitors:")
    for m in monitors:
        print(f"  [{m['index']}] {m['name']:<20} {m['width']}x{m['height']} at "
              f"({m['left']}, {m['top']}){'  primary' if m['primary'] else ''}")
    under_mouse = next((m for m in monitors if common.point_in_monitor(*common.cursor_pos(), m)),
                       monitors[0])
    idx = ask(f"Which monitor is the board on? [{under_mouse['index']}]",
              [str(m["index"]) for m in monitors], str(under_mouse["index"]))
    return monitors[int(idx) - 1]


# --------------------------------------------------------------- geometry --

def grid_from_points(tl, br):
    return {"x0": float(tl[0]), "y0": float(tl[1]),
            "dx": (br[0] - tl[0]) / 8.0, "dy": (br[1] - tl[1]) / 9.0}


def region_for_grid(grid, bounds=None):
    """Board capture rectangle: all intersections plus a margin for the outer pieces."""
    m = 0.75 * max(grid["dx"], grid["dy"])
    left, top = int(grid["x0"] - m), int(grid["y0"] - m)
    right, bottom = int(grid["x0"] + 8 * grid["dx"] + m), int(grid["y0"] + 9 * grid["dy"] + m)
    if bounds:
        left, top = max(left, bounds["left"]), max(top, bounds["top"])
        right = min(right, bounds["left"] + bounds["width"])
        bottom = min(bottom, bounds["top"] + bounds["height"])
    return {"left": left, "top": top, "width": right - left, "height": bottom - top}


def occupied_cells(my_color):
    board = common.board_from_fen(common.START_BOARD_FEN)
    cells = {}
    for row in range(10):
        for col in range(9):
            sr, sc = common.board_to_cell(row, col, my_color)
            cells[(sr, sc)] = board[row][col]
    return cells  # screen cell -> piece char or "."


def refine_grid(img, origin, grid, cells):
    """Fit the grid to detected piece circles. Returns (grid, report string)."""
    s = min(grid["dx"], grid["dy"])
    half = int(0.8 * s)
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    xs, ys = [], []  # (sc, x) and (sr, y) in absolute screen pixels
    for (sr, sc), ch in cells.items():
        if ch == ".":
            continue
        cx, cy = common.cell_center(sr, sc, grid)
        lx, ly = int(round(cx - origin[0])), int(round(cy - origin[1]))
        x0, y0 = max(lx - half, 0), max(ly - half, 0)
        win = gray[y0:ly + half, x0:lx + half]
        if win.size == 0:
            continue
        circles = cv2.HoughCircles(win, cv2.HOUGH_GRADIENT, dp=1, minDist=s / 2,
                                   param1=100, param2=20,
                                   minRadius=int(0.28 * s), maxRadius=int(0.55 * s))
        if circles is None:
            continue
        best = min(circles[0], key=lambda c: (c[0] + x0 - lx) ** 2 + (c[1] + y0 - ly) ** 2)
        bx, by = best[0] + x0 + origin[0], best[1] + y0 + origin[1]
        if np.hypot(bx - cx, by - cy) < 0.35 * s:
            xs.append((sc, bx))
            ys.append((sr, by))
    if len(xs) < 12:
        return grid, f"only {len(xs)} pieces detected; kept your hover points"
    xs, ys = np.array(xs), np.array(ys)
    dx, x0 = np.polyfit(xs[:, 0], xs[:, 1], 1)
    dy, y0 = np.polyfit(ys[:, 0], ys[:, 1], 1)
    resid = max(np.abs(xs[:, 1] - (x0 + dx * xs[:, 0])).max(),
                np.abs(ys[:, 1] - (y0 + dy * ys[:, 0])).max())
    if resid > 0.12 * s:
        return grid, f"fit residual {resid:.1f}px too large; kept your hover points"
    shift = np.hypot(x0 - grid["x0"], y0 - grid["y0"])
    new = {"x0": float(x0), "y0": float(y0), "dx": float(dx), "dy": float(dy)}
    return new, (f"fitted to {len(xs)} pieces, max residual {resid:.1f}px, "
                 f"top-left moved {shift:.1f}px from your hover point")


def crop(img, origin, x, y, size):
    lx, ly = int(round(x - origin[0])), int(round(y - origin[1]))
    h = size // 2
    patch = img[ly - h:ly - h + size, lx - h:lx - h + size]
    return patch if patch.shape[:2] == (size, size) else None


# ---------------------------------------------------------------- outputs --

def save_templates(img, origin, grid, cells, size, out_dir):
    """Save every piece instance and every empty start square. Returns {label: [patches]}."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "empty").mkdir(parents=True)
    by_label, counts = {}, {}
    for (sr, sc), ch in sorted(cells.items()):
        x, y = common.cell_center(sr, sc, grid)
        patch = crop(img, origin, x, y, size)
        if patch is None:
            raise SystemExit(f"Cell ({sr},{sc}) is outside the captured image - recalibrate.")
        if ch == ".":
            cv2.imwrite(str(out_dir / "empty" / f"cell_{sr}_{sc}.png"), patch)
            continue
        label = common.piece_label(ch)
        counts[label] = counts.get(label, 0) + 1
        cv2.imwrite(str(out_dir / f"{label}_{counts[label]}.png"), patch)
        by_label.setdefault(label, []).append(patch)
    return by_label


def template_report(by_label):
    """Same-type instances should match closely; different types clearly less."""
    sim = common.disc_similarity
    same = min((sim(q, p[0]), lab) for lab, p in by_label.items() for q in p[1:])
    firsts = [(lab, p[0]) for lab, p in by_label.items()]
    cross = max((sim(b, a), f"{la} vs {lb}")
                for i, (la, a) in enumerate(firsts) for lb, b in firsts[i + 1:])
    lines = [f"  lowest same-type match:       {same[0]:.3f} ({same[1]})  (want > 0.90)",
             f"  highest different-type match: {cross[0]:.3f} ({cross[1]})  (want well below)"]
    ok = same[0] > 0.90 and same[0] - cross[0] > 0.05
    return ok, "\n".join(lines)


def save_preview(img, origin, grid, cfg, path):
    out = img.copy()
    for row in range(10):
        for col in range(9):
            sr, sc = common.board_to_cell(row, col, cfg["my_color"])
            x, y = common.cell_center(sr, sc, grid)
            lx, ly = int(round(x - origin[0])), int(round(y - origin[1]))
            cv2.circle(out, (lx, ly), 4, (0, 255, 0), -1)
            cv2.putText(out, common.square_name(row, col), (lx + 5, ly - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1, cv2.LINE_AA)
    half = cfg["patch_size"] // 2
    x, y = common.cell_center(0, 0, grid)
    lx, ly = int(round(x - origin[0])), int(round(y - origin[1]))
    cv2.rectangle(out, (lx - half, ly - half), (lx + half, ly + half), (255, 255, 0), 1)
    cv2.imwrite(str(path), out)


# ------------------------------------------------------------------- main --

def capture(region):
    with (getattr(mss, "MSS", None) or mss.mss)() as sct:
        return np.array(sct.grab(region))[:, :, :3].copy()


def parse_point(s):
    x, y = s.split(",")
    return float(x), float(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="calibrate from a screenshot instead of the live screen")
    ap.add_argument("--tl", type=parse_point, help="offline: top-left intersection x,y")
    ap.add_argument("--br", type=parse_point, help="offline: bottom-right intersection x,y")
    ap.add_argument("--color", choices=["red", "black"])
    ap.add_argument("--out", default=str(common.PROJECT_DIR), help="where to write config/templates")
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = common.load_config(out_dir)

    color = args.color or {"r": "red", "b": "black"}[
        ask("Do you play Red or Black? [r/b]", ["r", "b"])]

    try:
        if args.image:
            if not (args.tl and args.br):
                sys.exit("--image needs --tl and --br")
            full = cv2.imread(args.image)
            if full is None:
                sys.exit(f"Cannot read {args.image}")
            bounds = {"left": 0, "top": 0, "width": full.shape[1], "height": full.shape[0]}
            grab = lambda r: full[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]].copy()
            ok = run_calibration(color, out_dir, tl=args.tl, br=args.br, grab=grab, bounds=bounds)
        else:
            monitor = choose_monitor(common.list_monitors())
            key = cfg["hotkeys"]["capture"]

            def wait_point(prompt):
                print(prompt)
                return wait_for_hotkey(key)
            ok = run_calibration(color, out_dir, monitor=monitor, wait_point=wait_point)
    except CalibrationError as e:
        sys.exit(str(e))
    if not ok:
        sys.exit(1)


def run_calibration(color, out_dir=common.PROJECT_DIR, *, monitor=None, wait_point=None,
                    tl=None, br=None, grab=None, bounds=None, say=print):
    """Calibrate and save config.json + templates. Returns True if the templates look consistent.

    Live: pass `monitor` and wait_point(prompt) -> (x, y), which shows the prompt and
    waits for the capture key. Offline: pass tl, br, grab(region) and bounds.
    Raises CalibrationError on bad input.
    """
    out_dir = Path(out_dir)
    cfg = common.load_config(out_dir)
    cfg["my_color"] = color
    cells = occupied_cells(color)

    if monitor is not None:
        bounds, grab = monitor, capture
        key = cfg["hotkeys"]["capture"].upper()
        say("Make sure the board shows the STARTING position with no highlighted move, "
            "and don't change browser zoom, window size or scroll after this.")
        tl = wait_point(f"Hover the TOP-LEFT intersection (as seen on screen) and press {key}.  Esc = abort")
        say(f"  got {tl}")
        br = wait_point(f"Hover the BOTTOM-RIGHT intersection and press {key}.  Esc = abort")
        say(f"  got {br}")
        for p in (tl, br):
            if not common.point_in_monitor(*p, monitor):
                raise CalibrationError(f"Point {p} is not on monitor [{monitor['index']}] {monitor['name']}.")

    grid = grid_from_points(tl, br)
    if min(grid["dx"], grid["dy"]) < 15:
        raise CalibrationError("Points are too close together (or in the wrong order). "
                               "Top-left first, then bottom-right.")

    region = region_for_grid(grid, bounds)
    img = grab(region)
    grid, report = refine_grid(img, (region["left"], region["top"]), grid, cells)
    say(f"Grid refinement: {report}")
    say(f"  spacing dx={grid['dx']:.1f}px dy={grid['dy']:.1f}px")

    region = region_for_grid(grid, bounds)
    img = grab(region)
    origin = (region["left"], region["top"])
    size = int(round(0.9 * min(grid["dx"], grid["dy"]))) | 1  # odd, so it has a centre pixel

    cfg.update({"grid": grid, "board_region": region, "patch_size": size,
                "monitor": monitor and {k: monitor[k] for k in
                                        ("index", "name", "device", "left", "top", "width", "height")}})
    templates_dir = common.resolve_path(cfg["templates_dir"], out_dir)
    by_label = save_templates(img, origin, grid, cells, size, templates_dir)
    cv2.imwrite(str(out_dir / "board_start.png"), img)
    save_preview(img, origin, grid, cfg, out_dir / "calibration_preview.png")
    cfg_path = common.save_config(cfg, out_dir)

    ok, text = template_report(by_label)
    say(f"Saved {sum(map(len, by_label.values()))} piece templates "
        f"({len(by_label)} types) and {sum(c == '.' for c in cells.values())} empty patches "
        f"of {size}x{size}px to {templates_dir}")
    say(text)
    say(f"Config: {cfg_path}")
    say(f"Preview: {out_dir / 'calibration_preview.png'}  (green dots should sit on every intersection)")
    if not ok:
        say("WARNING: templates look inconsistent - check the preview and recalibrate if needed.")
    return ok


if __name__ == "__main__":
    main()
