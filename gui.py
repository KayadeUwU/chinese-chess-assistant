"""Control panel (PySide6). Started by main.py.

Threads:
  - GUI thread: widgets only.
  - Worker thread (QThread): owns the engine, the board reader and the mouse.
    It gets commands through a queue and reports back only through Qt signals.
  - pynput thread: global hotkeys (work while the browser has focus); they only
    set events / queue commands on the worker.
"""
import ctypes
import logging
import math
import os
import queue
import threading
import time
from ctypes import wintypes

import common  # must be imported first: sets DPI awareness

from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QPlainTextEdit,
    QPushButton, QSpinBox, QTabWidget, QVBoxLayout, QWidget)

import calibrate
import controller
import engine
import game
import notation
import vision

log = logging.getLogger("xiangqi")

PIECE_CHARS = {"R": "俥", "N": "傌", "B": "相", "A": "仕", "K": "帥", "C": "炮", "P": "兵",
               "r": "車", "n": "馬", "b": "象", "a": "士", "k": "將", "c": "砲", "p": "卒"}
SETTING_KEYS = ("movetime_ms", "depth_limit", "multipv", "engine_threads", "engine_hash_mb")


# ==========================================================================
# Helpers
# ==========================================================================

def window_overlaps_region(hwnd, region):
    """Does window `hwnd` (physical pixels) overlap `region` on screen?"""
    user32 = ctypes.windll.user32
    if not hwnd or not region or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return False
    r = wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(r))
    return not (r.right <= region["left"] or r.left >= region["left"] + region["width"]
                or r.bottom <= region["top"] or r.top >= region["top"] + region["height"])


def side_of_fen(fen):
    return "black" if fen.split()[1] == "b" else "red"


def red_score(cp, mate, side):
    """Engine scores are for the side to move; turn them into Red's point of view."""
    sign = 1 if side == "red" else -1
    return (None if cp is None else sign * cp), (None if mate is None else sign * mate)


def format_score(cp, mate):
    if mate is not None:
        return f"M{mate}" if mate > 0 else f"-M{-mate}"
    if cp is None:
        return "–"
    return f"{cp / 100:+.2f}"


def red_share(cp, mate):
    """Fraction of the eval bar that is red: a sigmoid, so +2 and +20 don't look alike."""
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    if cp is None:
        return 0.5
    return 1 / (1 + math.exp(-0.00368208 * cp))


class StableWatcher:
    """Feeds captures one by one; says when a new, unchanged-for-N-frames image is ready."""

    def __init__(self, frames):
        self.frames, self.prev, self.count, self.done = frames, None, 0, None

    def update(self, img):
        same = self.prev is not None and vision.frame_diff(self.prev, img) < 1.0
        self.count = self.count + 1 if same else 1
        self.prev = img
        if self.count >= self.frames and (self.done is None or vision.frame_diff(self.done, img) >= 1.0):
            self.done = img
            return True
        return False


class LogBridge(QObject):
    line = Signal(str)


class QtLogHandler(logging.Handler):
    """Forwards log records (from any thread) to the GUI's log tab."""

    def __init__(self):
        super().__init__()
        self.bridge = LogBridge()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        self.bridge.line.emit(self.format(record))


class WorkerThread(QThread):
    """Runs Worker.run(). The worker's signals cross to the GUI as queued connections."""

    def __init__(self, worker, parent=None):
        super().__init__(parent)
        self.worker = worker

    def run(self):
        self.worker.run()


# ==========================================================================
# Worker
# ==========================================================================

class Worker(QObject):
    reading_ready = Signal(object)        # vision.Reading
    analysis = Signal(object)             # snapshot dict
    status = Signal(str)
    message = Signal(str, str)            # level, text
    move_made = Signal(str, str, object)  # colour, uci, board before the move
    history_reset = Signal(object)        # board the move list restarts from
    engine_name = Signal(str)
    mode_changed = Signal(str)
    calibration_prompt = Signal(str)

    MODE_COMMANDS = ("read", "analyze", "play_one", "auto", "calibrate", "shutdown")

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.commands = queue.Queue()
        self.control = type("Control", (), {})()
        self.control.paused, self.control.quit = threading.Event(), threading.Event()
        self.mode = "idle"
        self.hwnd = None
        self.eng = self.reader = self.ctl = self.game = None
        self.track_board, self.track_history = None, None
        self.analysis_paused = False
        self.overlap_warned = False

    # ---------------------------------------------------- called from any thread --

    def request(self, cmd, **kw):
        """Queue a command. Mode commands interrupt whatever is running."""
        if cmd in self.MODE_COMMANDS and self.mode in ("auto", "play_one", "calibrate", "read"):
            self.control.quit.set()
        self.commands.put((cmd, kw))

    def stop_all(self):
        self.control.paused.clear()
        self.control.quit.set()
        self.commands.put(("stop", {}))

    def toggle_pause(self):
        if self.mode in ("auto", "play_one"):
            if self.control.paused.is_set():
                self.control.paused.clear()
                self.status.emit("Resuming")
            else:
                self.control.paused.set()
                self.status.emit("Paused")
        elif self.mode == "analyze":
            self.analysis_paused = not self.analysis_paused
            self.status.emit("Paused" if self.analysis_paused else "Analyzing")

    # ------------------------------------------------------------ worker thread --

    def run(self):
        self.start_engine()
        self.load_vision()
        while True:
            cmd, kw = self.commands.get()
            self.control.quit.clear()
            if cmd == "shutdown":
                break
            try:
                self.dispatch(cmd, kw)
            except game.Quit:
                pass
            except controller.pyautogui.FailSafeException:
                self.message.emit("warning", "Mouse fail-safe triggered (mouse in a screen corner) - stopped.")
            except (engine.EngineError, OSError) as e:
                log.exception("worker error")
                self.message.emit("error", f"{type(e).__name__}: {e}")
                self.status.emit("Error")
                self.set_mode("idle")
                continue
            except Exception as e:  # keep the worker alive whatever happens
                log.exception("worker error")
                self.message.emit("error", f"{type(e).__name__}: {e}")
                self.status.emit("Error")
                self.set_mode("idle")
                continue
            if self.eng and self.eng.searching:
                self.eng.stop_search()
            self.control.paused.clear()
            self.set_mode("idle")
            self.status.emit("Idle")
        if self.eng:
            self.eng.close()

    def set_mode(self, mode):
        self.mode = mode
        self.mode_changed.emit(mode)

    def start_engine(self):
        try:
            self.eng = engine.Engine.from_config(self.cfg)
            self.engine_name.emit(self.eng.name)
        except engine.EngineError as e:
            self.eng = None
            self.engine_name.emit("no engine")
            self.message.emit("error", str(e))

    def load_vision(self):
        try:
            self.reader = vision.BoardReader(self.cfg)
            self.ctl = controller.Controller(self.cfg)
        except SystemExit as e:   # not calibrated / templates missing
            self.reader = self.ctl = None
            self.message.emit("warning", f"{e}  Use Calibrate.")

    def need(self, *what):
        if "vision" in what and self.reader is None:
            self.message.emit("warning", "Not calibrated yet - press Calibrate first.")
            return False
        if "engine" in what and self.eng is None:
            self.message.emit("error", "Engine not running - check engine_path in config.json.")
            return False
        return True

    def dispatch(self, cmd, kw):
        if cmd == "stop":
            return
        if cmd == "settings":
            return self.apply_engine_settings()
        if cmd == "read" and self.need("vision"):
            self.set_mode("read")
            self.status.emit("Reading board")
            reading = self.reader.read_stable(attempts=2, frames=self.cfg["stable_frames"], timeout=3.0)
            if reading is None:
                self.message.emit("warning", "The board never stopped changing.")
                return
            self.publish_reading(reading)
        elif cmd == "analyze" and self.need("vision", "engine"):
            self.set_mode("analyze")
            self.analyze()
        elif cmd in ("play_one", "auto") and self.need("vision", "engine"):
            self.set_mode(cmd)
            self.play(max_moves=1 if cmd == "play_one" else None)
        elif cmd == "calibrate":
            self.set_mode("calibrate")
            self.run_calibration(**kw)

    def apply_engine_settings(self):
        if self.eng:
            self.eng.set_option("Threads", self.cfg["engine_threads"])
            self.eng.set_option("Hash", self.cfg["engine_hash_mb"])

    # --------------------------------------------------------- board tracking --

    def publish_reading(self, reading):
        """Show a reading and follow the moves between successive boards."""
        self.reading_ready.emit(reading)
        if not reading.ok or reading.game_over:
            if reading.game_over:
                self.message.emit("info", "Game over panel on the board.")
            return None
        board = reading.board
        side = self.side_for(reading)
        fen = reading.fen(side)
        if self.track_board is None or (vision.board_matches_start(board)
                                        and not vision.board_matches_start(self.track_board)):
            self.reset_track(board, fen)
        elif board != self.track_board:
            for mover in ("red", "black"):
                mv = game.infer_move(self.track_board, board, mover)
                if mv:
                    self.move_made.emit(mover, mv.uci, self.track_board)
                    if self.track_history:
                        self.track_history[1].append(mv.uci)
                    break
            else:
                self.reset_track(board, fen)
        self.track_board = board
        return fen

    def reset_track(self, board, fen):
        self.track_history = (fen, [])
        self.history_reset.emit(board)

    def side_for(self, reading):
        if reading.side_to_move:
            return reading.side_to_move
        if vision.board_matches_start(reading.board):
            return "red"
        if self.track_board is not None:
            for mover in ("red", "black"):
                if game.infer_move(self.track_board, reading.board, mover):
                    return game.other(mover)
        return reading.bottom or self.cfg["my_color"]

    def snapshot(self, result, fen, mode):
        return {"fen": fen, "board": common.board_from_fen(fen), "side": side_of_fen(fen),
                "depth": result.depth, "nps": result.nps, "nodes": result.nodes,
                "lines": [(n, l.depth, l.score_cp, l.mate_in, list(l.pv))
                          for n, l in sorted(result.lines.items())],
                "best": result.move.uci if result.move else None,
                "finished": result.finished, "mode": mode}

    def check_overlap(self):
        if window_overlaps_region(self.hwnd, self.cfg.get("board_region")):
            if not self.overlap_warned:
                self.message.emit("warning", "This window overlaps the board on screen - "
                                             "move it away, or the board can't be read or clicked.")
                self.overlap_warned = True
        else:
            self.overlap_warned = False

    # -------------------------------------------------------------- analysis --

    def analyze(self):
        """Live analysis of whatever is on screen. Never clicks."""
        self.status.emit("Analyzing")
        watcher = StableWatcher(self.cfg["stable_frames"])
        interval = self.cfg["frame_interval_s"]
        current, settings = None, None
        next_capture = last_emit = 0.0
        self.analysis_paused = False
        while True:
            try:
                cmd, kw = self.commands.get_nowait()
            except queue.Empty:
                cmd = None
            if cmd == "settings":
                self.apply_engine_settings()
            elif cmd is not None:
                self.eng.stop_search()
                self.commands.put((cmd, kw))   # handled by the main loop
                return
            if self.control.quit.is_set():
                self.eng.stop_search()
                return
            if self.analysis_paused:
                self.eng.stop_search()
                current = None
                time.sleep(0.1)
                continue

            now = time.monotonic()
            if now >= next_capture:
                next_capture = now + interval
                self.check_overlap()
                if watcher.update(self.reader.capture()):
                    reading = self.reader.read(watcher.done)
                    fen = self.publish_reading(reading)
                    if fen is None:
                        watcher.done = None        # misread or game over: try again later
                        if reading.game_over:
                            self.eng.stop_search()
                            current = None
                        time.sleep(0.3)
                    elif fen != current:
                        current, settings = fen, None
            want = (self.cfg["multipv"], self.cfg["depth_limit"],
                    self.cfg["engine_threads"], self.cfg["engine_hash_mb"])
            if current and want != settings:
                # New position or new settings: stop, wait for bestmove, then restart.
                self.apply_engine_settings()
                self.eng.begin_search(current, self.track_history, depth=self.cfg["depth_limit"] or None,
                                      multipv=self.cfg["multipv"])
                settings = want
            was_searching = self.eng.searching
            if not was_searching:
                time.sleep(0.03)       # depth limit reached or nothing to analyse yet
                continue
            changed, _ = self.eng.poll(0.03)
            just_finished = not self.eng.searching
            if current and (just_finished or (changed and time.monotonic() - last_emit >= 0.1)):
                self.analysis.emit(self.snapshot(self.eng.current, current, "analyze"))
                last_emit = time.monotonic()

    # ------------------------------------------------------------------ play --

    def play(self, max_moves=None):
        worker = self

        class Hooks(game.Hooks):
            last = 0.0

            def status(self, state):
                worker.status.emit(state)

            def reading(self, reading):
                worker.reading_ready.emit(reading)

            def search_info(self, result, fen):
                now = time.monotonic()
                if now - Hooks.last >= 0.1:
                    Hooks.last = now
                    worker.analysis.emit(worker.snapshot(result, fen, "play"))

            def move(self, colour, move, board_before):
                worker.move_made.emit(colour, move.uci, board_before)
                worker.track_board = engine.apply_move(board_before, move)

            def history_reset(self, board):
                # Continue the move list the worker already shows, if this board follows it.
                prev = worker.track_board
                worker.track_board = board
                if prev is not None:
                    if board == prev:
                        return
                    for mover in ("red", "black"):
                        mv = game.infer_move(prev, board, mover)
                        if mv:
                            worker.move_made.emit(mover, mv.uci, prev)
                            return
                worker.history_reset.emit(board)

            def message(self, level, text):
                worker.message.emit(level, text)

        def guard():
            if window_overlaps_region(self.hwnd, self.cfg.get("board_region")):
                raise controller.ClickRefused("this window overlaps the board - move it off the board")

        g = game.Game(self.cfg, self.reader, self.ctl, self.eng, self.control,
                      hooks=Hooks(), click_guard=guard)
        result = g.run(max_moves=max_moves)
        if g.history:
            self.track_history = g.history
        if result == "done":
            self.message.emit("info", "Played one move.")

    # ----------------------------------------------------------- calibration --

    def run_calibration(self, color, monitor_index):
        monitor = next(m for m in common.list_monitors() if m["index"] == monitor_index)
        self.status.emit("Calibrating")

        def wait_point(prompt):
            self.calibration_prompt.emit(prompt)
            return calibrate.wait_for_hotkey(self.cfg["hotkeys"]["capture"], cancel=self.control.quit)

        try:
            ok = calibrate.run_calibration(color, monitor=monitor, wait_point=wait_point,
                                           say=lambda text: log.info("%s", text))
        except calibrate.CalibrationError as e:
            self.calibration_prompt.emit("")
            self.message.emit("warning", str(e))
            return
        self.calibration_prompt.emit("")
        fresh = common.load_config()
        if window_overlaps_region(self.hwnd, fresh.get("board_region")):
            self.message.emit("warning", "This window overlapped the board while calibrating - "
                                         "move it away and calibrate again.")
        for k, v in fresh.items():
            if k not in SETTING_KEYS:
                self.cfg[k] = v
        self.load_vision()
        self.track_board = self.track_history = None
        self.message.emit("info" if ok else "warning",
                          "Calibration saved." if ok else "Calibration saved, but the templates look "
                                                          "inconsistent - check calibration_preview.png.")


# ==========================================================================
# Widgets
# ==========================================================================

class BoardView(QWidget):
    WOOD = QColor("#e8b866")
    LINE = QColor("#7a4a1c")

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 470)
        self.board = None
        self.bottom = "red"
        self.manual_flip = False
        self.last_opp_move = None      # (src, dst) board coordinates
        self.screen_highlight = []     # squares PlayOK highlights
        self.arrows = []               # (uci, rank) rank 0 = best
        self.low_conf = set()
        self.unknown = set()
        self.game_over = False
        self.font_families = ["KaiTi", "STKaiti", "Microsoft YaHei", "SimSun"]

    @property
    def flipped(self):
        return (self.bottom == "black") != self.manual_flip

    def geometry_(self):
        # Room around the grid for half a piece plus the coordinate labels.
        s = min(self.width() / 9.7, self.height() / 10.7)
        ox = (self.width() - 8 * s) / 2
        oy = (self.height() - 9 * s) / 2
        return s, ox, oy

    def point(self, row, col):
        s, ox, oy = self.geometry_()
        sr, sc = (9 - row, 8 - col) if self.flipped else (row, col)
        return QPointF(ox + sc * s, oy + sr * s)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s, ox, oy = self.geometry_()
        p.fillRect(self.rect(), self.WOOD.darker(115))
        p.fillRect(QRectF(ox - s * 0.82, oy - s * 0.82, 8 * s + 1.64 * s, 9 * s + 1.64 * s), self.WOOD)
        self.draw_grid(p, s, ox, oy)
        self.draw_highlights(p, s)
        if self.board:
            self.draw_pieces(p, s)
        self.draw_arrows(p, s)
        if self.board is None:
            p.setPen(self.LINE)
            p.drawText(self.rect(), Qt.AlignCenter, "Press Read Board or Analyze Only")
        if self.game_over:
            p.fillRect(self.rect(), QColor(0, 0, 0, 90))
            f = QFont(self.font())
            f.setPointSizeF(22)
            f.setBold(True)
            p.setFont(f)
            p.setPen(Qt.white)
            p.drawText(self.rect(), Qt.AlignCenter, "Game over")
        p.end()

    def draw_grid(self, p, s, ox, oy):
        pen = QPen(self.LINE, 1.6)
        p.setPen(pen)
        for r in range(10):
            p.drawLine(QPointF(ox, oy + r * s), QPointF(ox + 8 * s, oy + r * s))
        for c in range(9):
            x = ox + c * s
            if c in (0, 8):
                p.drawLine(QPointF(x, oy), QPointF(x, oy + 9 * s))
            else:   # inner files stop at the river
                p.drawLine(QPointF(x, oy), QPointF(x, oy + 4 * s))
                p.drawLine(QPointF(x, oy + 5 * s), QPointF(x, oy + 9 * s))
        for top in (0, 7):  # palaces
            p.drawLine(QPointF(ox + 3 * s, oy + top * s), QPointF(ox + 5 * s, oy + (top + 2) * s))
            p.drawLine(QPointF(ox + 5 * s, oy + top * s), QPointF(ox + 3 * s, oy + (top + 2) * s))
        p.setPen(QPen(self.LINE, 2.6))
        p.drawRect(QRectF(ox - 5, oy - 5, 8 * s + 10, 9 * s + 10))
        f = QFont()
        f.setFamilies(self.font_families)
        f.setPixelSize(int(s * 0.45))
        p.setFont(f)
        p.setPen(self.LINE)
        p.drawText(QRectF(ox + 0.5 * s, oy + 4 * s, 3 * s, s), Qt.AlignCenter, "楚 河")
        p.drawText(QRectF(ox + 4.5 * s, oy + 4 * s, 3 * s, s), Qt.AlignCenter, "漢 界")
        # file / rank labels
        f.setFamilies(["Segoe UI"])
        f.setPixelSize(max(10, int(s * 0.24)))
        p.setFont(f)
        p.setPen(self.LINE.darker(130))
        for col in range(9):
            pt = self.point(0, col)
            p.drawText(QRectF(pt.x() - 12, oy + 9 * s + s * 0.5, 24, s * 0.3), Qt.AlignCenter,
                       common.FILES[col])
        for row in range(10):
            pt = self.point(row, 0)
            p.drawText(QRectF(ox - s * 0.8, pt.y() - 9, s * 0.3, 18), Qt.AlignCenter, str(10 - row))

    def draw_highlights(self, p, s):
        squares = []
        if self.last_opp_move:
            squares = list(self.last_opp_move)
        elif self.screen_highlight:
            squares = [engine.square_to_index(sq) for sq in self.screen_highlight]
        for row, col in squares:
            pt = self.point(row, col)
            p.fillRect(QRectF(pt.x() - s * 0.48, pt.y() - s * 0.48, s * 0.96, s * 0.96),
                       QColor(30, 120, 255, 95))

    def draw_pieces(self, p, s):
        f = QFont()
        f.setFamilies(self.font_families)
        f.setPixelSize(int(s * 0.5))
        f.setBold(True)
        p.setFont(f)
        r = s * 0.44
        for row in range(10):
            for col in range(9):
                ch = self.board[row][col]
                pt = self.point(row, col)
                sq = common.square_name(row, col)
                if ch == "?":
                    p.setPen(QPen(QColor("#e6c200"), 2.5, Qt.DashLine))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(pt, r, r)
                    p.drawText(QRectF(pt.x() - r, pt.y() - r, 2 * r, 2 * r), Qt.AlignCenter, "?")
                    continue
                if ch != ".":
                    red = ch.isupper()
                    colour = QColor("#c8171e") if red else QColor("#222222")
                    p.setPen(QPen(QColor(0, 0, 0, 60), 1))
                    p.setBrush(QColor(0, 0, 0, 50))
                    p.drawEllipse(pt + QPointF(1.5, 2), r, r)
                    p.setBrush(QColor("#f6e3c0"))
                    p.setPen(QPen(colour, 2.2))
                    p.drawEllipse(pt, r, r)
                    p.setPen(QPen(colour, 1))
                    p.setBrush(Qt.NoBrush)
                    p.drawEllipse(pt, r * 0.82, r * 0.82)
                    p.setPen(colour)
                    p.drawText(QRectF(pt.x() - r, pt.y() - r, 2 * r, 2 * r), Qt.AlignCenter,
                               PIECE_CHARS[ch])
                if sq in self.low_conf:
                    p.setPen(QPen(QColor("#ffd400"), 3))
                    p.setBrush(Qt.NoBrush)
                    p.drawRect(QRectF(pt.x() - s * 0.5, pt.y() - s * 0.5, s, s))

    def draw_arrows(self, p, s):
        for uci, rank in reversed(self.arrows):
            try:
                m = engine.parse_move(uci)
            except ValueError:
                continue
            a, b = self.point(*m.src), self.point(*m.dst)
            alpha = 210 if rank == 0 else max(60, 140 - 35 * rank)
            width = s * (0.16 if rank == 0 else 0.09)
            colour = QColor(20, 150, 70, alpha) if rank == 0 else QColor(30, 90, 200, alpha)
            dx, dy = b.x() - a.x(), b.y() - a.y()
            length = math.hypot(dx, dy) or 1
            ux, uy = dx / length, dy / length
            head = width * 2.6
            start = QPointF(a.x() + ux * s * 0.2, a.y() + uy * s * 0.2)
            base = QPointF(b.x() - ux * head, b.y() - uy * head)
            p.setPen(QPen(colour, width, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(start, base)
            p.setPen(Qt.NoPen)
            p.setBrush(colour)
            px, py = -uy, ux
            p.drawPolygon(QPolygonF([b, QPointF(base.x() + px * head * 0.7, base.y() + py * head * 0.7),
                                     QPointF(base.x() - px * head * 0.7, base.y() - py * head * 0.7)]))


class EvalBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(30)
        self.share = 0.5
        self.flipped = False
        self.mate_text = ""

    def set_eval(self, cp, mate):
        self.share = red_share(cp, mate)
        self.mate_text = format_score(None, mate) if mate is not None else ""
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(2, 2, self.width() - 4, self.height() - 4)
        path = QPainterPath()
        path.addRoundedRect(rect, 5, 5)
        p.setClipPath(path)
        p.fillRect(rect, QColor("#1f1f1f"))
        red_h = rect.height() * self.share
        if self.flipped:   # red at the top when the board shows Black at the bottom
            p.fillRect(QRectF(rect.left(), rect.top(), rect.width(), red_h), QColor("#d0262d"))
        else:
            p.fillRect(QRectF(rect.left(), rect.bottom() - red_h, rect.width(), red_h), QColor("#d0262d"))
        p.setPen(QPen(QColor(255, 255, 255, 120), 1))
        p.drawLine(QPointF(rect.left(), rect.center().y()), QPointF(rect.right(), rect.center().y()))
        if self.mate_text:
            f = QFont(self.font())
            f.setPixelSize(11)
            f.setBold(True)
            p.setFont(f)
            p.setPen(Qt.white)
            y = rect.top() + 4 if (self.share > 0.5) == self.flipped else rect.bottom() - 18
            p.drawText(QRectF(rect.left(), y, rect.width(), 14), Qt.AlignCenter, self.mate_text)
        p.setClipping(False)
        p.setPen(QPen(QColor("#555"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rect, 5, 5)
        p.end()


class CalibrateDialog(QDialog):
    def __init__(self, parent, cfg):
        super().__init__(parent)
        self.setWindowTitle("Calibrate")
        form = QFormLayout(self)
        self.colour = QComboBox()
        self.colour.addItems(["Red", "Black"])
        self.colour.setCurrentIndex(1 if cfg.get("my_color") == "black" else 0)
        self.monitor = QComboBox()
        self.monitors = common.list_monitors()
        current = (cfg.get("monitor") or {}).get("device")
        for m in self.monitors:
            self.monitor.addItem(f"[{m['index']}] {m['name']}  {m['width']}x{m['height']} "
                                 f"at ({m['left']}, {m['top']})", m["index"])
            if m["device"] == current:
                self.monitor.setCurrentIndex(self.monitor.count() - 1)
        form.addRow("I play", self.colour)
        form.addRow("Board is on", self.monitor)
        key = cfg["hotkeys"]["capture"].upper()
        note = QLabel(f"Show the STARTING position with no highlighted move.\n"
                      f"After OK: hover the top-left intersection and press {key},\n"
                      f"then the bottom-right one and press {key}. Esc cancels.")
        note.setWordWrap(True)
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)


# ==========================================================================
# Main window
# ==========================================================================

class MainWindow(QMainWindow):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setWindowTitle("Xiangqi assistant")
        self.resize(1240, 780)

        self.snapshot = None
        self.reading = None
        self.history = []          # ("sep", board) or (colour, uci, board_before)
        self.dirty = False
        self.mode = "idle"

        self.build_ui()

        # Worker thread
        self.worker = Worker(cfg)
        self.thread = WorkerThread(self.worker, self)
        self.worker.reading_ready.connect(self.on_reading)
        self.worker.analysis.connect(self.on_analysis)
        self.worker.status.connect(self.on_status)
        self.worker.message.connect(self.on_message)
        self.worker.move_made.connect(self.on_move)
        self.worker.history_reset.connect(self.on_history_reset)
        self.worker.engine_name.connect(lambda n: self.engine_label.setText(n))
        self.worker.mode_changed.connect(self.on_mode)
        self.worker.calibration_prompt.connect(self.on_calibration_prompt)

        self.log_handler = QtLogHandler()
        self.log_handler.bridge.line.connect(self.log_box.appendPlainText)
        self.log_handler.setLevel(logging.INFO)
        log.addHandler(self.log_handler)

        # Global hotkeys - work while the browser has focus
        self.hotkeys = game.Hotkeys(cfg, on_pause=self.worker.toggle_pause, on_quit=self.worker.stop_all)

        # Repaint at most ~10 times a second
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

        self.save_timer = QTimer(self)
        self.save_timer.setSingleShot(True)
        self.save_timer.timeout.connect(self.save_settings)

        self.thread.start()

    # ------------------------------------------------------------------ UI --

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        root.addLayout(top, 1)

        # Eval bar + score
        eval_col = QVBoxLayout()
        self.score_label = QLabel("–")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setMinimumWidth(56)
        f = self.score_label.font()
        f.setBold(True)
        f.setPointSize(11)
        self.score_label.setFont(f)
        self.eval_bar = EvalBar()
        eval_col.addWidget(self.score_label)
        eval_col.addWidget(self.eval_bar, 1, Qt.AlignHCenter)
        top.addLayout(eval_col)

        self.board_view = BoardView()
        top.addWidget(self.board_view, 3)

        # Analysis panel
        right = QVBoxLayout()
        top.addLayout(right, 2)

        engine_box = QGroupBox("Analysis")
        eb = QVBoxLayout(engine_box)
        hdr = QHBoxLayout()
        self.depth_label = QLabel("Depth –")
        self.nps_label = QLabel("– n/s")
        self.notation_box = QComboBox()
        self.notation_box.addItems(["Engine (h2e2)", "Chinese (炮二平五)"])
        self.notation_box.currentIndexChanged.connect(self.notation_changed)
        hdr.addWidget(self.depth_label)
        hdr.addWidget(self.nps_label)
        hdr.addStretch(1)
        hdr.addWidget(self.notation_box)
        eb.addLayout(hdr)
        self.lines_list = QListWidget()
        mono = QFont("Consolas")
        mono.setStyleHint(QFont.Monospace)
        self.lines_list.setFont(mono)
        self.lines_list.setMinimumHeight(130)
        self.lines_list.setWordWrap(True)
        eb.addWidget(self.lines_list)
        right.addWidget(engine_box, 2)

        tabs = QTabWidget()
        self.history_list = QListWidget()
        self.history_list.setFont(mono)
        tabs.addTab(self.history_list, "Moves")
        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setFont(mono)
        tabs.addTab(self.log_box, "Log")
        right.addWidget(tabs, 3)

        fen_row = QHBoxLayout()
        self.fen_edit = QLineEdit()
        self.fen_edit.setReadOnly(True)
        self.fen_edit.setPlaceholderText("FEN")
        copy = QPushButton("Copy")
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.fen_edit.text()))
        fen_row.addWidget(QLabel("FEN"))
        fen_row.addWidget(self.fen_edit, 1)
        fen_row.addWidget(copy)
        right.addLayout(fen_row)

        # Controls
        controls = QHBoxLayout()
        root.addLayout(controls)
        self.btn = {}
        for key, text, slot in [
                ("calibrate", "Calibrate", self.do_calibrate),
                ("read", "Read Board", lambda: self.worker.request("read")),
                ("analyze", "Analyze Only", lambda: self.worker.request("analyze")),
                ("play_one", "Play Best Move", lambda: self.worker.request("play_one")),
                ("auto", "Auto Play", self.do_auto),
                ("stop", "Stop", self.worker_stop),
                ("flip", "Flip Board", self.do_flip)]:
            b = QPushButton(text)
            b.clicked.connect(slot)
            b.setMinimumHeight(34)
            b.setCheckable(key in ("read", "analyze", "play_one", "auto", "calibrate"))
            controls.addWidget(b)
            self.btn[key] = b
        self.btn["stop"].setStyleSheet("QPushButton { font-weight: bold; color: #b00000; }")

        # Settings
        settings = QGroupBox("Settings")
        grid = QGridLayout(settings)
        self.spins = {}
        specs = [("movetime_ms", "Move time (ms)", 100, 60000, 100),
                 ("depth_limit", "Depth limit (0 = none)", 0, 99, 1),
                 ("multipv", "MultiPV", 1, 5, 1),
                 ("engine_threads", "Threads", 1, max(1, os.cpu_count() or 2), 1),
                 ("engine_hash_mb", "Hash (MB)", 16, 4096, 16)]
        for i, (key, label, lo, hi, step) in enumerate(specs):
            spin = QSpinBox()
            spin.setRange(lo, hi)
            spin.setSingleStep(step)
            spin.setValue(int(self.cfg.get(key, lo)))
            spin.valueChanged.connect(lambda v, k=key: self.setting_changed(k, v))
            grid.addWidget(QLabel(label), 0, i * 2)
            grid.addWidget(spin, 0, i * 2 + 1)
            self.spins[key] = spin
        root.addWidget(settings)

        # Status bar
        sb = self.statusBar()
        self.state_label = QLabel("Idle")
        self.state_label.setMinimumWidth(180)
        f = self.state_label.font()
        f.setBold(True)
        self.state_label.setFont(f)
        self.msg_label = QLabel("")
        self.engine_label = QLabel("starting engine…")
        self.conf_label = QLabel("Vision: –")
        sb.addWidget(self.state_label)
        sb.addWidget(self.msg_label, 1)
        sb.addPermanentWidget(self.conf_label)
        sb.addPermanentWidget(self.engine_label)

    # ------------------------------------------------------------- buttons --

    def do_auto(self):
        if self.mode == "auto":
            self.worker.toggle_pause()
        else:
            self.worker.request("auto")

    def worker_stop(self):
        self.worker.stop_all()

    def do_flip(self):
        self.board_view.manual_flip = not self.board_view.manual_flip
        self.dirty = True

    def do_calibrate(self):
        dlg = CalibrateDialog(self, self.cfg)
        if dlg.exec() != QDialog.Accepted:
            return
        colour = "black" if dlg.colour.currentIndex() == 1 else "red"
        self.worker.request("calibrate", color=colour, monitor_index=dlg.monitor.currentData())

    def setting_changed(self, key, value):
        self.cfg[key] = value            # the worker reads these on its next search
        if key in ("engine_threads", "engine_hash_mb"):
            self.worker.request("settings")
        self.save_timer.start(600)

    def save_settings(self):
        disk = common.load_config()
        for k in SETTING_KEYS:
            disk[k] = self.cfg[k]
        common.save_config(disk)

    # ------------------------------------------------------- worker signals --

    def on_mode(self, mode):
        self.mode = mode
        self.btn["auto"].setText("Pause" if mode == "auto" else "Auto Play")
        for key in ("read", "analyze", "play_one", "auto", "calibrate"):
            self.btn[key].setChecked(mode == key)

    def on_status(self, state):
        if state == "Paused" and self.mode == "auto":
            self.btn["auto"].setText("Resume")
        elif state != "Paused" and self.mode == "auto":
            self.btn["auto"].setText("Pause")
        self.state_label.setText(state)

    def on_message(self, level, text):
        colour = {"error": "#b00000", "warning": "#a06000"}.get(level, "#205020")
        self.msg_label.setStyleSheet(f"color: {colour};")
        self.msg_label.setText(text)
        (log.error if level == "error" else log.warning if level == "warning" else log.info)("%s", text)

    def on_calibration_prompt(self, text):
        if text:
            self.state_label.setText("Calibrating")
            self.msg_label.setStyleSheet("color: #0040a0; font-weight: bold;")
            self.msg_label.setText(text)
            log.info("%s", text)

    def on_reading(self, reading):
        if len(reading.unknown) > 20 and not reading.game_over:
            # Covered/scrolled board: keep showing the last good one.
            self.on_message("warning", "Board not visible - " + (reading.problems or ["unreadable"])[0])
            self.conf_label.setText("Vision: board not visible")
            return
        if not reading.ok and not reading.game_over:
            self.on_message("warning", "Misread squares (yellow): " +
                            ", ".join(sq for sq, _ in reading.unknown[:6]) + " - not sent to the engine")
        self.reading = reading
        self.dirty = True

    def on_analysis(self, snap):
        self.snapshot = snap
        self.dirty = True

    def on_move(self, colour, uci, board_before):
        self.history.append((colour, uci, board_before))
        self.dirty = True

    def on_history_reset(self, board):
        if self.history and self.history[-1][0] == "sep":
            self.history[-1] = ("sep", board)
        else:
            self.history.append(("sep", board))
        self.dirty = True

    def notation_changed(self):
        self.dirty = True

    # -------------------------------------------------------------- refresh --

    def chinese(self):
        return self.notation_box.currentIndex() == 1

    def refresh(self):
        if not self.dirty:
            return
        self.dirty = False
        r, snap, bv = self.reading, self.snapshot, self.board_view

        if r is not None:
            bv.board = r.board
            bv.bottom = r.bottom or bv.bottom
            bv.screen_highlight = r.highlighted
            bv.low_conf = set(r.low_confidence)
            bv.game_over = bool(r.game_over)
            mc = r.min_confidence
            self.conf_label.setText("Vision: –" if mc is None else f"Vision: {mc:.2f} min")
            if r.ok:
                self.fen_edit.setText(r.fen(snap["side"] if snap and snap["board"] == r.board
                                            else r.side_to_move or None))

        me = bv.bottom
        opp = [h for h in self.history if h[0] != "sep" and h[0] != me]
        last = self.history[-1] if self.history else None
        if opp and last and last[0] != "sep":
            m = engine.parse_move(opp[-1][1])
            bv.last_opp_move = (m.src, m.dst)
        else:
            bv.last_opp_move = None

        if snap and (bv.board is None or snap["board"] == bv.board):
            self.fen_edit.setText(snap["fen"])
            bv.arrows = [(pv[0], i) for i, (_, _, _, _, pv) in enumerate(snap["lines"]) if pv]
            best = snap["lines"][0] if snap["lines"] else None
            if best:
                cp, mate = red_score(best[2], best[3], snap["side"])
                self.eval_bar.set_eval(cp, mate)
                self.score_label.setText(format_score(cp, mate))
            self.depth_label.setText(f"Depth {snap['depth'] or '–'}")
            nps = snap["nps"]
            self.nps_label.setText("– n/s" if not nps else
                                   f"{nps / 1e6:.2f} Mn/s" if nps >= 1e6 else f"{nps / 1e3:.0f} kn/s")
            self.lines_list.clear()
            for n, depth, cp, mate, pv in snap["lines"]:
                rcp, rmate = red_score(cp, mate, snap["side"])
                moves = (notation.line_to_chinese(snap["board"], pv[:8]) if self.chinese() else pv[:8])
                self.lines_list.addItem(f"{n}. {format_score(rcp, rmate):>7}  d{depth:<2}  {' '.join(moves)}")
        else:
            bv.arrows = []
        self.eval_bar.update()
        self.render_history()
        bv.update()

    def render_history(self):
        rows, num, pending = [], 0, None
        for item in self.history:
            if item[0] == "sep":
                if pending:
                    rows.append(pending)
                    pending = None
                rows.append("— position read from screen —")
                continue
            colour, uci, before = item
            text = uci
            if self.chinese():
                try:
                    text = notation.to_chinese(before, uci)
                except ValueError:
                    pass
            if colour == "red" or pending is None:
                if pending:
                    rows.append(pending)
                num += 1
                pending = f"{num:>3}. {text:<10}" if colour == "red" else f"{num:>3}. {'…':<10} {text}"
                if colour == "black":
                    rows.append(pending)
                    pending = None
            else:
                rows.append(f"{pending} {text}")
                pending = None
        if pending:
            rows.append(pending)
        if self.history_list.count() != len(rows) or any(
                self.history_list.item(i).text() != t for i, t in enumerate(rows)):
            self.history_list.clear()
            self.history_list.addItems(rows)
            self.history_list.scrollToBottom()

    # ---------------------------------------------------------------- close --

    def showEvent(self, e):
        super().showEvent(e)
        self.worker.hwnd = int(self.winId())

    def closeEvent(self, e):
        self.hotkeys.stop()
        self.worker.stop_all()
        self.worker.request("shutdown")
        if not self.thread.wait(8000):
            log.warning("Worker did not stop in time.")
        log.removeHandler(self.log_handler)
        super().closeEvent(e)


def main():
    game.setup_logging(console=True)
    app = QApplication.instance() or QApplication([])
    cfg = common.load_config()
    win = MainWindow(cfg)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
