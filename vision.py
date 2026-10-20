"""Screen -> board state.

Each of the 90 intersections is classified in two steps:
  1. Colour of the piece disc: board-coloured (empty), red, or dark (black).
     A "last move" highlight only darkens the board, so it stays "empty".
  2. For a piece, its glyph (inner disc only, grayscale correlation) is matched
     against that colour's templates. Rims, selection rings and highlights
     around the piece are outside the compared disc.
Squares that don't clear the thresholds are reported as unknown ('?'),
never guessed.

The screen orientation (which colour is at the bottom) is taken from the red
king's position, and the side to move from the last-move highlight.

    python vision.py                 read the live board once
    python vision.py --watch         re-read every time the board settles after a change
    python vision.py --image shot.png --debug
                                     read a full-monitor screenshot, save vision_debug.png
"""
import argparse
import time
from dataclasses import dataclass, field

import common  # must be imported first: sets DPI awareness

import cv2
import mss
import numpy as np

# Disc colour rules, measured on PlayOK (OpenCV HSV: H 0-179, S/V 0-255).
DISC_COLOR_RADIUS = 0.40   # fraction of patch size
RED_PIXEL = dict(hue_lo=8, hue_hi=170, s_min=120, v_min=100)
DARK_V_MAX = 90
BOARD_HUE = (9, 30)
BOARD_S_MIN = 60
MIN_COLOR_FRACTION = 0.45  # winning colour must cover this much of the disc
MIN_GLYPH_FRACTION = 0.08  # a piece's white character covers ~16-24% of the disc

MIN_PIECE_SCORE = 0.80     # glyph correlation needed to accept a piece
MIN_PIECE_MARGIN = 0.05    # ...and lead over the second-best piece type
LOW_CONFIDENCE = 0.90      # accepted, but flagged in the GUI below this
HIGHLIGHT_V_RATIO = 0.88   # board corners darker than this x normal = highlighted

# PlayOK's end-of-game panel: a solid rectangle in the pieces' red, several
# squares wide. Pieces are single discs, so size alone separates the two.
GAME_OVER_MIN_W = 3.0      # x grid spacing
GAME_OVER_MIN_H = 1.3      # x grid spacing
GAME_OVER_MIN_FILL = 0.5

MSS = getattr(mss, "MSS", None) or mss.mss

MAX_COUNTS = {"k": 1, "a": 2, "b": 2, "n": 2, "r": 2, "c": 2, "p": 5}


@dataclass
class Reading:
    board: list                    # 10x9 in board coordinates, '.' empty, '?' unknown
    bottom: str                    # colour at the bottom of the screen, or None
    side_to_move: str = None       # 'red'/'black' from the highlight, or None if unknown
    unknown: list = field(default_factory=list)      # (square, reason)
    highlighted: list = field(default_factory=list)  # board squares, e.g. 'e4'
    problems: list = field(default_factory=list)     # sanity-check failures
    infos: dict = field(default_factory=dict)        # square -> classification detail
    image: object = field(default=None, repr=False)  # the board capture this was read from
    game_over: tuple = None                          # (x, y, w, h) of the end-of-game panel
    confidence: dict = field(default_factory=dict)   # square -> 0..1 (glyph score or colour share)

    @property
    def min_confidence(self):
        return min(self.confidence.values()) if self.confidence else None

    @property
    def low_confidence(self):
        """Squares that were accepted but only just, plus unknown ones."""
        return sorted(sq for sq, c in self.confidence.items() if c < LOW_CONFIDENCE)

    @property
    def ok(self):
        return not self.unknown and not self.problems

    def fen(self, side=None):
        side = side or self.side_to_move or "red"
        ranks = []
        for row in self.board:
            out, empty = "", 0
            for ch in row:
                if ch == ".":
                    empty += 1
                    continue
                if empty:
                    out, empty = out + str(empty), 0
                out += ch
            ranks.append(out + (str(empty) if empty else ""))
        return f"{'/'.join(ranks)} {'w' if side == 'red' else 'b'} - - 0 1"

    def text(self):
        """Board as seen on screen, with board-square labels around it."""
        files = common.FILES if self.bottom != "black" else common.FILES[::-1]
        rows = range(10) if self.bottom != "black" else range(9, -1, -1)
        lines = ["     " + "  ".join(files)]
        for row in rows:
            cells = self.board[row] if self.bottom != "black" else self.board[row][::-1]
            lines.append(f"{10 - row:>3}  " + "  ".join(cells))
        return "\n".join(lines)


class BoardReader:
    def __init__(self, cfg=None):
        self.cfg = cfg or common.load_config()
        if not self.cfg.get("grid"):
            raise SystemExit("No calibration found - run calibrate.py first.")
        self.size = self.cfg["patch_size"]
        self.region = self.cfg["board_region"]
        tdir = common.resolve_path(self.cfg["templates_dir"])
        self.templates = {"red": {}, "black": {}}  # colour -> piece char -> [disc templates]
        for path in sorted(tdir.glob("*_*_*.png")):
            colour, name, _ = path.stem.split("_")
            ch = next(k for k, v in common.PIECE_NAMES.items() if v == name)
            ch = ch.upper() if colour == "red" else ch
            img = cv2.imread(str(path))
            self.templates[colour].setdefault(ch, []).append(common.make_disc_template(img))
        if len(self.templates["red"]) != 7 or len(self.templates["black"]) != 7:
            raise SystemExit(f"Templates in {tdir} are incomplete - rerun calibrate.py.")
        n = self.size
        yy, xx = np.mgrid[:n, :n]
        dist = np.hypot(xx - n // 2, yy - n // 2)
        self.disc_mask = dist <= DISC_COLOR_RADIUS * n
        self.corner_mask = dist >= 0.47 * n

    # ---------------------------------------------------------- capture --

    def capture(self):
        with MSS() as sct:
            return np.array(sct.grab(self.region))[:, :, :3].copy()

    def crop_from_screenshot(self, full, origin=None):
        """Board region out of a full-monitor screenshot (pixel 0,0 = monitor top-left)."""
        mon = self.cfg.get("monitor") or {"left": 0, "top": 0}
        ox, oy = origin or (mon["left"], mon["top"])
        r = self.region
        x, y = r["left"] - ox, r["top"] - oy
        return full[y:y + r["height"], x:x + r["width"]].copy()

    def wait_stable(self, frames=4, interval=0.12, timeout=30.0, threshold=1.0):
        """Capture until `frames` consecutive captures are identical (no animation).

        Returns the last image, or None on timeout.
        """
        deadline = time.monotonic() + timeout
        prev, same = self.capture(), 1
        while time.monotonic() < deadline:
            time.sleep(interval)
            cur = self.capture()
            same = same + 1 if frame_diff(prev, cur) < threshold else 1
            if same >= frames:
                return cur
            prev = cur
        return None

    # ---------------------------------------------------------- reading --

    def patch(self, img, sr, sc):
        x, y = common.cell_center(sr, sc, self.cfg["grid"])
        lx = int(round(x - self.region["left"]))
        ly = int(round(y - self.region["top"]))
        h = self.size // 2
        return img[ly - h:ly - h + self.size, lx - h:lx - h + self.size]

    def disc_colour(self, hsv):
        h, s, v = (hsv[..., i][self.disc_mask].astype(int) for i in range(3))
        fractions = {
            "red": np.mean(((h < RED_PIXEL["hue_lo"]) | (h > RED_PIXEL["hue_hi"]))
                           & (s >= RED_PIXEL["s_min"]) & (v >= RED_PIXEL["v_min"])),
            "black": np.mean(v <= DARK_V_MAX),
            "empty": np.mean((h >= BOARD_HUE[0]) & (h <= BOARD_HUE[1])
                             & (s >= BOARD_S_MIN) & (v > DARK_V_MAX)),
        }
        best = max(fractions, key=fractions.get)
        glyph = np.mean((s < 60) & (v > 170))
        return best, fractions[best], glyph

    def corner_board_v(self, hsv):
        h, s, v = (hsv[..., i][self.corner_mask].astype(int) for i in range(3))
        board = (h >= BOARD_HUE[0]) & (h <= BOARD_HUE[1]) & (s >= BOARD_S_MIN)
        return float(np.median(v[board])) if board.sum() > 20 else None

    def classify(self, patch):
        """Return (char, info, confidence): char is '.', a piece letter, or '?'."""
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        colour, frac, glyph = self.disc_colour(hsv)
        if frac < MIN_COLOR_FRACTION:
            return "?", f"unclear colour ({colour} {frac:.2f})", 0.0
        if colour == "empty":
            return ".", f"empty {frac:.2f}", float(frac)
        if glyph < MIN_GLYPH_FRACTION:
            return "?", f"{colour} disc without a character (glyph {glyph:.2f})", 0.0
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        # A flat patch makes the correlation undefined (nan/inf): count that as no match.
        scores = sorted(((float(np.nan_to_num(max(common.disc_score(gray, t) for t in ts),
                                              nan=-1.0, posinf=-1.0, neginf=-1.0)), ch)
                         for ch, ts in self.templates[colour].items()), reverse=True)
        (best, ch), (second, ch2) = scores[0], scores[1]
        info = f"{common.piece_label(ch)} {best:.2f} (next {common.piece_label(ch2)} {second:.2f})"
        if best < MIN_PIECE_SCORE or best - second < MIN_PIECE_MARGIN:
            return "?", "low confidence: " + info, max(best, 0.0)
        return ch, info, best

    def find_game_over(self, img):
        """Bounding box (x, y, w, h) of PlayOK's end-of-game panel, or None."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        red = (((h < RED_PIXEL["hue_lo"]) | (h > RED_PIXEL["hue_hi"]))
               & (s >= RED_PIXEL["s_min"]) & (v >= RED_PIXEL["v_min"])).astype(np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(red)
        dx, dy = self.cfg["grid"]["dx"], self.cfg["grid"]["dy"]
        for x, y, w, hh, area in stats[1:]:
            if (w >= GAME_OVER_MIN_W * dx and hh >= GAME_OVER_MIN_H * dy
                    and area >= GAME_OVER_MIN_FILL * w * hh):
                return int(x), int(y), int(w), int(hh)
        return None

    def read(self, img=None):
        img = self.capture() if img is None else img
        cells, infos, corner_v, conf = {}, {}, {}, {}
        for sr in range(10):
            for sc in range(9):
                p = self.patch(img, sr, sc)
                cells[sr, sc], infos[sr, sc], conf[sr, sc] = self.classify(p)
                corner_v[sr, sc] = self.corner_board_v(cv2.cvtColor(p, cv2.COLOR_BGR2HSV))

        bottom = detect_bottom(cells)
        orient = bottom or self.cfg["my_color"]
        board = [["."] * 9 for _ in range(10)]
        reading = Reading(board=board, bottom=bottom, image=img, game_over=self.find_game_over(img))
        for (sr, sc), ch in cells.items():
            row, col = common.board_to_cell(sr, sc, orient)
            board[row][col] = ch
            reading.confidence[common.square_name(row, col)] = conf[sr, sc]
            if ch == "?":
                reading.unknown.append((common.square_name(row, col), infos[sr, sc]))
        if reading.game_over:
            reading.problems.append("game-over panel on the board")
            return reading
        if len(reading.unknown) > 20:
            reading.problems.append(f"{len(reading.unknown)} squares unreadable - "
                                    "is the board covered, scrolled or zoomed?")
            return reading

        normal_v = np.median([v for v in corner_v.values() if v is not None])
        lit = [(sr, sc) for (sr, sc), v in corner_v.items()
               if v is not None and v < HIGHLIGHT_V_RATIO * normal_v]
        reading.highlighted = [common.square_name(*common.board_to_cell(sr, sc, orient))
                               for sr, sc in lit]
        movers = {("red" if cells[c].isupper() else "black") for c in lit
                  if cells[c] not in ".?"}
        if len(movers) == 1:
            reading.side_to_move = "black" if movers == {"red"} else "red"
        elif not lit and board_matches_start(board):
            reading.side_to_move = "red"

        # Placement rules need the real orientation; with it unknown, only count pieces.
        reading.problems = sanity_problems(board, placement=bottom is not None) if not reading.unknown else []
        if bottom is None and not reading.unknown:
            reading.problems.append("cannot tell orientation: red king not in a palace")
        reading.infos = {common.square_name(*common.board_to_cell(sr, sc, orient)): i
                         for (sr, sc), i in infos.items()}
        return reading

    def read_stable(self, attempts=5, frames=4, timeout=30.0):
        """Wait for a still board and read it; re-capture until it passes the checks."""
        reading = None
        for _ in range(attempts):
            img = self.wait_stable(frames=frames, timeout=timeout)
            if img is None:
                continue
            reading = self.read(img)
            if reading.ok:
                return reading
            time.sleep(0.3)
        return reading

    def debug_image(self, img, reading, path):
        out = img.copy()
        orient = reading.bottom or self.cfg["my_color"]
        h = self.size // 2
        for row in range(10):
            for col in range(9):
                sr, sc = common.board_to_cell(row, col, orient)
                x, y = common.cell_center(sr, sc, self.cfg["grid"])
                lx, ly = int(round(x - self.region["left"])), int(round(y - self.region["top"]))
                ch = reading.board[row][col]
                name = common.square_name(row, col)
                colour = ((0, 0, 255) if ch == "?" else (255, 0, 255) if name in reading.highlighted
                          else (0, 200, 0))
                if ch == "?" or name in reading.highlighted:
                    cv2.rectangle(out, (lx - h, ly - h), (lx + h, ly + h), colour, 2)
                if ch != ".":
                    cv2.putText(out, ch, (lx + h - 14, ly + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(out, ch, (lx + h - 14, ly + h - 2), cv2.FONT_HERSHEY_SIMPLEX,
                                0.55, (255, 255, 0), 1, cv2.LINE_AA)
        if reading.game_over:
            x, y, w, hh = reading.game_over
            cv2.rectangle(out, (x - 3, y - 3), (x + w + 3, y + hh + 3), (255, 255, 0), 3)
        cv2.imwrite(str(path), out)


# ---------------------------------------------------------------- helpers --

def frame_diff(a, b):
    return float(np.mean(cv2.absdiff(a, b)))


def detect_bottom(cells):
    """Colour at the bottom of the screen, from the red king's screen row."""
    rows = [sr for (sr, _), ch in cells.items() if ch == "K"]
    if len(rows) != 1:
        return None
    return "red" if rows[0] >= 7 else "black" if rows[0] <= 2 else None


def board_matches_start(board):
    return board == common.board_from_fen(common.START_BOARD_FEN)


PALACE = {"red": {(r, c) for r in (7, 8, 9) for c in (3, 4, 5)},
          "black": {(r, c) for r in (0, 1, 2) for c in (3, 4, 5)}}
ADVISOR = {"red": {(7, 3), (7, 5), (8, 4), (9, 3), (9, 5)},
           "black": {(0, 3), (0, 5), (1, 4), (2, 3), (2, 5)}}
ELEPHANT = {"red": {(5, 2), (5, 6), (7, 0), (7, 4), (7, 8), (9, 2), (9, 6)},
            "black": {(0, 2), (0, 6), (2, 0), (2, 4), (2, 8), (4, 2), (4, 6)}}


def pawn_ok(colour, row, col):
    if colour == "red":
        return row <= 4 or (row in (5, 6) and col % 2 == 0)
    return row >= 5 or (row in (3, 4) and col % 2 == 0)


def sanity_problems(board, placement=True):
    """Piece counts and placement rules every legal Xiangqi position obeys."""
    problems, counts = [], {}
    for row in range(10):
        for col in range(9):
            ch = board[row][col]
            if ch in ".?":
                continue
            counts[ch] = counts.get(ch, 0) + 1
            colour, kind, sq = ("red" if ch.isupper() else "black"), ch.lower(), common.square_name(row, col)
            if not placement:
                continue
            if kind == "k" and (row, col) not in PALACE[colour]:
                problems.append(f"{colour} king outside palace at {sq}")
            elif kind == "a" and (row, col) not in ADVISOR[colour]:
                problems.append(f"{colour} advisor on illegal point {sq}")
            elif kind == "b" and (row, col) not in ELEPHANT[colour]:
                problems.append(f"{colour} elephant on illegal point {sq}")
            elif kind == "p" and not pawn_ok(colour, row, col):
                problems.append(f"{colour} pawn on illegal point {sq}")
    for kind, limit in MAX_COUNTS.items():
        for ch in (kind.upper(), kind):
            n = counts.get(ch, 0)
            colour = "red" if ch.isupper() else "black"
            if kind == "k" and n != 1:
                problems.append(f"{colour} has {n} kings")
            elif n > limit:
                problems.append(f"{colour} has {n} x {common.PIECE_NAMES[kind]} (max {limit})")
    return problems


def report(reading, verbose=False):
    print(reading.text())
    print(f"\nbottom: {reading.bottom or '?'}   side to move: {reading.side_to_move or '? (unknown)'}"
          f"   highlighted: {', '.join(reading.highlighted) or 'none'}")
    if reading.game_over:
        print(f"GAME OVER: end-of-game panel at {reading.game_over}")
        return
    for sq, why in reading.unknown[:8]:
        print(f"  UNKNOWN {sq}: {why}")
    if len(reading.unknown) > 8:
        print(f"  ... and {len(reading.unknown) - 8} more unknown squares")
    for p in reading.problems:
        print(f"  PROBLEM {p}")
    if verbose:
        for sq, info in sorted(reading.infos.items()):
            if not info.startswith("empty"):
                print(f"  {sq:>3}: {info}")
    if reading.ok:
        print(f"\nFEN: {reading.fen()}")
    else:
        print("\nBoard not trusted - no FEN.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", help="full-monitor screenshot instead of the live screen")
    ap.add_argument("--watch", action="store_true", help="print the board each time it changes")
    ap.add_argument("--debug", action="store_true", help="save vision_debug.png and print piece scores")
    args = ap.parse_args()
    reader = BoardReader()

    if args.image:
        full = cv2.imread(args.image)
        if full is None:
            raise SystemExit(f"Cannot read {args.image}")
        img = reader.crop_from_screenshot(full)
        t = time.perf_counter()
        reading = reader.read(img)
        print(f"(read in {1000 * (time.perf_counter() - t):.0f} ms)\n")
    elif args.watch:
        last = None
        print("Watching the board - Ctrl+C to stop.")
        try:
            while True:
                img = reader.wait_stable()
                if img is None or (last is not None and frame_diff(img, last) < 1.0):
                    continue
                last = img
                print("\n" + time.strftime("%H:%M:%S"))
                report(reader.read(img), args.debug)
        except KeyboardInterrupt:
            return
    else:
        img = reader.wait_stable()
        if img is None:
            raise SystemExit("Board never stopped changing.")
        reading = reader.read(img)

    report(reading, args.debug)
    if args.debug:
        path = common.PROJECT_DIR / "vision_debug.png"
        reader.debug_image(img, reading, path)
        print(f"Debug image: {path}")


if __name__ == "__main__":
    main()
