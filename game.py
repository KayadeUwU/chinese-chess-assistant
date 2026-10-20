"""Game logic shared by the GUI and the command line.

Game.run() waits for my turn, reads the board, asks the engine, plays the move
and confirms it landed. It reports what it does through a Hooks object, so a
GUI can show it without the game code touching any widgets.
"""
import logging
import threading
import time
import winsound

import common

from pynput import keyboard

import calibrate
import controller
import engine
import vision

log = logging.getLogger("xiangqi")
LOG_DIR = common.PROJECT_DIR / "logs"


def colour_of(piece):
    return "red" if piece.isupper() else "black"


def other(colour):
    return "black" if colour == "red" else "red"


def infer_move(before, after, mover):
    """The single move by `mover` that turns board `before` into `after`, or None."""
    changed = [(r, c) for r in range(10) for c in range(9) if before[r][c] != after[r][c]]
    if len(changed) != 2:
        return None
    for src, dst in (changed, changed[::-1]):
        piece, target = before[src[0]][src[1]], before[dst[0]][dst[1]]
        if (piece not in ".?" and colour_of(piece) == mover
                and after[src[0]][src[1]] == "." and after[dst[0]][dst[1]] == piece
                and (target == "." or colour_of(target) != mover)):
            return engine.parse_move(common.square_name(*src) + common.square_name(*dst))
    return None


def alert():
    for _ in range(3):
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        time.sleep(0.4)


class Hooks:
    """What the game reports while it runs. The GUI overrides these."""

    def status(self, state):                  # Waiting for opponent / Thinking / Moving / Paused ...
        pass

    def reading(self, reading):               # every trusted board read
        pass

    def search_info(self, result, fen):       # engine progress while thinking
        pass

    def move(self, colour, move, board_before):   # a move by either side
        pass

    def history_reset(self, board):           # move list restarts from this board
        pass

    def message(self, level, text):           # warnings worth showing the user
        pass


class Hotkeys:
    """Global hotkeys (work whichever window has focus)."""

    def __init__(self, cfg, on_pause=None, on_quit=None):
        self.paused = threading.Event()
        self.quit = threading.Event()
        self.pause_key = calibrate.parse_key(cfg["hotkeys"]["pause"])
        self.quit_key = calibrate.parse_key(cfg["hotkeys"]["quit"])
        self.on_pause, self.on_quit = on_pause, on_quit
        self.listener = keyboard.Listener(on_press=self._on_press, daemon=True)
        self.listener.start()

    def _on_press(self, key):
        if key == self.quit_key:
            log.warning("Quit hotkey pressed.")
            self.quit.set()
            if self.on_quit:
                self.on_quit()
        elif key == self.pause_key:
            if self.on_pause:
                self.on_pause()
                return
            if self.paused.is_set():
                self.paused.clear()
                log.info("Resumed.")
            else:
                self.paused.set()
                log.info("Paused - press %s to resume.", self.pause_key)

    def stop(self):
        self.listener.stop()


class Quit(Exception):
    """Stop was requested (quit hotkey, Stop button)."""


class Game:
    """`control` has two threading.Events: paused and quit."""

    def __init__(self, cfg, reader, ctl, eng, control, colour=None, click=True,
                 hooks=None, click_guard=None):
        self.cfg, self.reader, self.ctl, self.eng = cfg, reader, ctl, eng
        self.keys, self.fixed_colour, self.click = control, colour, click
        self.hooks = hooks or Hooks()
        self.click_guard = click_guard
        self.me = colour
        self.last_board = None      # board when it became the opponent's turn
        self.history = None         # (base_fen, [moves]) for the engine
        self.misreads = 0
        self.last_reading = None

    # ------------------------------------------------------------ helpers --

    def checkpoint(self):
        if self.keys.quit.is_set():
            raise Quit
        if self.keys.paused.is_set():
            self.hooks.status("Paused")
            while self.keys.paused.is_set():
                if self.keys.quit.is_set():
                    raise Quit
                time.sleep(0.1)
            self.last_board, self.history = None, None   # anything may have happened
            self.hooks.status("Reading board")

    def pause(self, why):
        log.error("%s - pausing. Fix it on screen, then press %s to resume.",
                  why, self.cfg["hotkeys"]["pause"].upper())
        self.hooks.message("error", why)
        self.keys.paused.set()
        alert()

    def read(self):
        """A board that passed every check, waiting through animations and misreads."""
        warned = False
        while True:
            self.checkpoint()
            reading = self.reader.read_stable(attempts=1, frames=self.cfg["stable_frames"], timeout=1.0)
            if reading is not None and (reading.ok or reading.game_over):
                return reading
            if reading is not None and not warned:
                problems = [f"{sq}: {why}" for sq, why in reading.unknown[:5]] + reading.problems[:5]
                log.warning("Board not trusted, re-capturing: %s", "; ".join(problems))
                log.debug("Misread board:\n%s", reading.text())
                self.hooks.reading(reading)
                self.save_misread(reading)
                warned = True
            time.sleep(0.3)

    def save_misread(self, reading):
        if self.misreads >= 50 or reading.image is None:
            return
        self.misreads += 1
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"misread_{time.strftime('%H%M%S')}_{self.misreads}.png"
        self.reader.debug_image(reading.image, reading, path)
        log.info("Saved %s", path.name)

    def game_over(self, reading, how):
        """PlayOK's end-of-game panel is up: log it, keep a screenshot, never click it."""
        log.info("Game over: %s.", how)
        self.hooks.message("info", f"Game over: {how}.")
        if reading is not None and reading.image is not None:
            LOG_DIR.mkdir(exist_ok=True)
            path = LOG_DIR / f"game_over_{time.strftime('%H%M%S')}.png"
            self.reader.debug_image(reading.image, reading, path)
            log.info("Saved %s", path.name)

    def whose_turn(self, reading):
        if reading.side_to_move:
            return reading.side_to_move
        if vision.board_matches_start(reading.board):
            return "red"
        if self.last_board is not None:          # changed since my move: opponent replied
            return self.me
        log.warning("Can't tell whose turn it is (no last-move highlight) - assuming mine.")
        return self.me

    def search(self, fen):
        return self.eng.search(
            fen, self.cfg["movetime_ms"], self.history,
            depth=self.cfg.get("depth_limit") or None, multipv=self.cfg.get("multipv", 1),
            on_info=lambda r: self.hooks.search_info(r, fen),
            should_abort=self.keys.quit.is_set)

    # --------------------------------------------------------------- loop --

    def run(self, max_moves=None):
        """Play until the game ends, or until `max_moves` of my moves have landed.

        Returns "game over", "done" (max_moves reached) or raises Quit.
        """
        log.info("Started. Waiting for my turn...")
        played = 0
        waiting_logged = False
        self.hooks.status("Reading board")
        while True:
            reading = self.read()
            if reading.game_over:
                if self.last_board is not None:   # squares under the panel: last known board
                    reading.board = [[k if c == "?" else c for c, k in zip(row, known)]
                                     for row, known in zip(reading.board, self.last_board)]
                self.hooks.reading(reading)
                self.game_over(reading, "end-of-game panel appeared while waiting "
                                        "(checkmate, resignation, timeout or draw)")
                return "game over"
            self.me = self.fixed_colour or reading.bottom
            board = reading.board

            if self.last_board is not None and board == self.last_board:
                time.sleep(0.1)                  # opponent still thinking
                continue
            self.hooks.reading(reading)
            if vision.board_matches_start(board) and self.history and self.history[1]:
                log.info("Board is back at the start position - new game.")
                self.history = None
            if self.whose_turn(reading) != self.me:
                if not waiting_logged:
                    log.info("Opponent to move.")
                    waiting_logged = True
                self.hooks.status("Waiting for opponent")
                self.last_board = board
                time.sleep(0.1)
                continue

            waiting_logged = False
            outcome = self.my_turn(reading)
            if outcome == "stop":
                return "game over"
            if outcome == "played":
                played += 1
                if max_moves and played >= max_moves:
                    return "done"

    def my_turn(self, reading):
        board, me = reading.board, self.me
        fen = reading.fen(me)

        opp_move = infer_move(self.last_board, board, other(me)) if self.last_board else None
        if opp_move:
            self.hooks.move(other(me), opp_move, self.last_board)
        if self.history and opp_move:
            self.history[1].append(opp_move.uci)
            log.info("Opponent played %s", opp_move.uci)
        else:
            if self.history:
                log.info("Couldn't work out the opponent's move - restarting history from this position.")
            self.history = (fen, [])
            if not opp_move:
                self.hooks.history_reset(board)

        log.info("FEN %s", fen)
        self.hooks.status("Thinking")
        result = self.search(fen)
        self.checkpoint()
        log.info("Engine: %s%s", engine.describe(result), "" if result.used_history else "  [no history]")
        if result.used_history is False and self.history[1]:
            self.history = (fen, [])
        if result.move is None:
            log.info("No legal move for %s - game over.", me)
            self.hooks.message("info", f"No legal move for {me} - game over.")
            return "stop"

        move = result.move
        if not self.click:
            log.info(">>> Play %s", move.uci)
            self.last_board = board
            self.history = None   # my move isn't tracked, so don't build on it
            return "suggested"

        expected = engine.apply_move(board, move)
        mating = result.mate_in == 1
        for attempt in (1, 2):
            self.checkpoint()
            self.hooks.status("Moving")
            try:
                src, dst = self.ctl.play(move, reading.bottom or me, guard=self.click_guard)
            except controller.ClickRefused as e:
                self.pause(f"Refused to click: {e}")
                self.last_board, self.history = None, None
                return None
            log.info("Played %s (%s -> %s)%s", move.uci, src, dst, " [retry]" if attempt == 2 else "")
            outcome = self.confirm(board, expected)
            if outcome == "game over":
                self.hooks.move(me, move, board)
                self.show_after_my_move(expected, reading.bottom or me)
                self.game_over(self.last_reading,
                               "checkmate delivered" if mating else "end-of-game panel appeared after my move")
                return "stop"
            if outcome == "landed":
                break
            if mating and outcome != "unchanged":
                # A game-over panel may cover the board right after mate.
                log.info("Board %s after the mating move - taking it as played.",
                         "unreadable" if outcome == "board unreadable" else "changed")
                outcome = "landed"
                break
            log.warning("Move %s did not land (%s).", move.uci, outcome)
            if outcome != "unchanged":
                break       # only re-click when the old position is visibly still there
        else:
            outcome = "unchanged after retry"
        if outcome != "landed":
            self.pause(f"Move {move.uci} did not land ({outcome})")
            self.last_board, self.history = None, None
            return None

        self.hooks.move(me, move, board)
        self.show_after_my_move(expected, reading.bottom or me)
        self.history[1].append(move.uci)
        self.last_board = expected
        if mating:
            log.info("Checkmate delivered - game over.")
            self.hooks.message("info", "Checkmate delivered - game over.")
            return "stop"
        return "played"

    def show_after_my_move(self, expected, bottom):
        """Report the board with my move on it.

        The main loop skips boards equal to the one it expects after my move, so
        without this the GUI would keep showing the position before it. Uses the
        read that confirmed the move; squares hidden under a game-over panel are
        filled in from the expected board.
        """
        r = self.last_reading
        if r is not None and r.ok:
            self.hooks.reading(r)
            return
        board = expected
        if r is not None and r.game_over:
            board = [[e if c == "?" else c for c, e in zip(row, exp)]
                     for row, exp in zip(r.board, expected)]
        self.hooks.reading(vision.Reading(board=board, bottom=bottom,
                                          game_over=r.game_over if r is not None else None))

    def confirm(self, before, expected):
        """'landed', 'unchanged', 'game over', 'board unreadable' or 'unexpected board'.

        Leaves the reading that decided it in self.last_reading.
        """
        deadline = time.monotonic() + self.cfg["confirm_timeout_s"]
        board, last_bad = None, None
        self.last_reading = None
        while time.monotonic() < deadline:
            self.checkpoint()
            reading = self.reader.read_stable(attempts=1, frames=self.cfg["stable_frames"], timeout=1.0)
            if reading is not None and reading.game_over:
                self.last_reading = reading
                return "game over"
            if reading is None or not reading.ok:
                last_bad = reading
                time.sleep(0.1)
                continue
            board = reading.board
            if board == expected:
                self.last_reading = reading
                return "landed"
            if board != before:
                # The opponent may already have replied before we looked.
                if infer_move(expected, board, other(self.me)):
                    self.last_reading = reading
                    return "landed"
                break
            time.sleep(0.1)
        if board is None:
            if last_bad is not None:
                why = [f"{sq}: {w}" for sq, w in last_bad.unknown[:3]] + last_bad.problems[:3]
                log.warning("Board unreadable after the move: %s", "; ".join(why))
                self.save_misread(last_bad)
            return "board unreadable"
        if board == before:
            return "unchanged"
        log.debug("Expected:\n%s\nGot:\n%s", vision.Reading(expected, self.me).text(),
                  vision.Reading(board, self.me).text())
        return "unexpected board"


def setup_logging(console=True):
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"game_{time.strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    if console:
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        sh.setLevel(logging.INFO)
        log.addHandler(sh)
    log.setLevel(logging.DEBUG)
    return path
