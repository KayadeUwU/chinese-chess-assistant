"""Board indices -> mouse actions.

move_mode in config.json: "click" (click source, wait, click destination)
or "drag" (press on source, drag, release on destination).

Fail-safe: slam the mouse into a corner of the PRIMARY monitor to abort.
With a monitor to the left of the primary, use the top-RIGHT corner.

    python controller.py h3e3 --dry-run   hover source then destination, no clicks
    python controller.py h3e3             play the move (orientation read from the screen)
    python controller.py h3e3 --bottom black --mode drag
"""
import argparse
import ctypes
import time

import common  # must be imported first: sets DPI awareness

import pyautogui

import engine

pyautogui.FAILSAFE = True   # never turn this off
pyautogui.PAUSE = 0.02


class ClickRefused(RuntimeError):
    """A guard said it is not safe to click (e.g. a window covers the board)."""


class Controller:
    def __init__(self, cfg=None):
        self.cfg = cfg or common.load_config()
        if not self.cfg.get("grid"):
            raise SystemExit("No calibration found - run calibrate.py first.")
        assert pyautogui.FAILSAFE

    def square_to_screen(self, row, col, bottom):
        """Board (row, col) -> absolute screen pixel, for the colour at the bottom."""
        sr, sc = common.board_to_cell(row, col, bottom)
        x, y = common.cell_center(sr, sc, self.cfg["grid"])
        x, y = int(round(x)), int(round(y))
        r = self.cfg["board_region"]
        if not (r["left"] <= x < r["left"] + r["width"] and r["top"] <= y < r["top"] + r["height"]):
            raise ValueError(f"({x}, {y}) is outside the calibrated board - recalibrate")
        return x, y

    def play(self, move, bottom, dry_run=False, guard=None):
        """Make `move` (engine.Move) on screen. Returns the two screen points used.

        guard(), if given, runs before every click (and before a drag's press)
        and raises ClickRefused to cancel.
        """
        src = self.square_to_screen(*move.src, bottom)
        dst = self.square_to_screen(*move.dst, bottom)
        delay = self.cfg["click_delay_s"]
        check = guard or (lambda: None)
        restore = common.cursor_pos() if self.cfg.get("restore_cursor", True) else None

        if dry_run:
            pyautogui.moveTo(*src, duration=0.3)
            time.sleep(0.6)
            pyautogui.moveTo(*dst, duration=0.3)
            time.sleep(0.6)
        elif self.cfg["move_mode"] == "drag":
            check()
            pyautogui.moveTo(*src, duration=0.05)
            pyautogui.mouseDown()
            try:
                time.sleep(delay)
                pyautogui.moveTo(*dst, duration=max(0.15, delay))
                time.sleep(delay)
            finally:
                pyautogui.mouseUp()
        else:
            check()
            pyautogui.click(*src, duration=0.05)
            time.sleep(delay)
            check()
            pyautogui.click(*dst, duration=0.05)

        if restore:
            # Move back where the user left the mouse; SetCursorPos is instant and skips the fail-safe.
            ctypes.windll.user32.SetCursorPos(*restore)
        return src, dst


def detect_bottom(cfg):
    import vision
    reading = vision.BoardReader(cfg).read_stable(attempts=3)
    if reading is None or reading.bottom is None:
        raise SystemExit("Could not read the board orientation - pass --bottom red|black.")
    return reading.bottom


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("move", help="engine notation, e.g. h3e3 or h10g8")
    ap.add_argument("--bottom", choices=["red", "black"], help="colour at the bottom (default: read screen)")
    ap.add_argument("--mode", choices=["click", "drag"], help="override move_mode from config.json")
    ap.add_argument("--dry-run", action="store_true", help="only hover the two squares")
    args = ap.parse_args()

    cfg = common.load_config()
    if args.mode:
        cfg["move_mode"] = args.mode
    move = engine.parse_move(args.move)
    bottom = args.bottom or detect_bottom(cfg)
    ctl = Controller(cfg)
    src = ctl.square_to_screen(*move.src, bottom)
    dst = ctl.square_to_screen(*move.dst, bottom)
    what = "hovering" if args.dry_run else f"playing ({cfg['move_mode']})"
    print(f"{what} {move.uci}: {engine.index_to_square(*move.src)} at {src} -> "
          f"{engine.index_to_square(*move.dst)} at {dst}   (bottom: {bottom})")
    print("Fail-safe: slam the mouse into the top-right corner of the primary monitor.")
    time.sleep(1.0)
    ctl.play(move, bottom, dry_run=args.dry_run)
    print("done")


if __name__ == "__main__":
    main()
