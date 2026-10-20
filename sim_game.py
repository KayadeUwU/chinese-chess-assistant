"""Offline test of main.py's game loop: a fake screen and mouse, and a second
Fairy-Stockfish as the opponent. Nothing on the real screen is touched.

    python sim_game.py
"""
import logging
import random
import threading
import time

import common
import engine
import game as main
import vision

FAST = {"stable_frames": 1, "confirm_timeout_s": 1.5, "movetime_ms": 60}


class FakeScreen:
    """Board + last-move highlight; the opponent answers my moves in a thread."""

    def __init__(self, opp_engine, opp_colour, bottom, misclicks=0, opp_movetime=20,
                 cover_on_mate=False, cover_after=None, start_fen=None,
                 panel=False, resign_after=None):
        self.board = common.board_from_fen(start_fen or common.START_BOARD_FEN)
        self.last_mover, self.lock = None, threading.Lock()
        if start_fen:   # the opponent just moved into this position
            self.last_mover = opp_colour
        self.opp, self.opp_colour, self.bottom = opp_engine, opp_colour, bottom
        self.misclicks, self.opp_movetime, self.opp_history = misclicks, opp_movetime, []
        self.over = False
        self.cover_on_mate, self.cover_after = cover_on_mate, cover_after
        self.covered, self.plays = False, 0
        self.panel, self.resign_after = panel, resign_after   # panel: cover = end-of-game panel
        if opp_colour == "red" and not start_fen:
            self._opponent_move()

    # reader interface
    def read_stable(self, attempts=1, frames=1, timeout=None):
        time.sleep(0.01)
        with self.lock:
            if self.covered and self.panel:
                return vision.Reading(board=[r[:] for r in self.board], bottom=self.bottom,
                                      game_over=(100, 100, 280, 95),
                                      problems=["game-over panel on the board"])
            if self.covered:   # something unrecognised over the board
                return vision.Reading(board=[["?"] * 9 for _ in range(10)], bottom=None,
                                      problems=["90 squares unreadable"])
            side = None if self.last_mover is None else main.other(self.last_mover)
            return vision.Reading(board=[r[:] for r in self.board], bottom=self.bottom, side_to_move=side)

    # controller interface
    def play(self, move, bottom, guard=None):
        assert bottom == self.bottom
        self.plays += 1
        if self.misclicks > 0:
            self.misclicks -= 1
            return (0, 0), (0, 0)
        with self.lock:
            self.board = engine.apply_move(self.board, move)
            self.last_mover = main.other(self.opp_colour)
            self.opp_history.append(move.uci)
            if self.cover_after is not None and self.plays >= self.cover_after:
                self.covered = True
                return (1, 1), (2, 2)
            if self.cover_on_mate:
                fen = vision.Reading(self.board, self.bottom).fen(self.opp_colour)
                if self.opp.search(fen, 1).move is None:
                    self.covered = True
                    return (1, 1), (2, 2)
        threading.Thread(target=self._opponent_move, daemon=True).start()
        return (1, 1), (2, 2)

    def _opponent_move(self):
        time.sleep(random.uniform(0.0, 0.2))
        with self.lock:
            hist = (engine.START_FEN, list(self.opp_history))
            fen = vision.Reading(self.board, self.bottom).fen(self.opp_colour)
            try:
                res = self.opp.search(fen, self.opp_movetime, hist)
            except engine.EngineError:   # game already over and engines closed
                return
            if res.move is None:
                self.over = True
                return
            if self.resign_after is not None and len(self.opp_history) >= self.resign_after:
                self.covered = True   # opponent resigns: panel appears, no move
                return
            self.board = engine.apply_move(self.board, res.move)
            self.last_mover = self.opp_colour
            self.opp_history.append(res.move.uci)


class FakeKeys:
    def __init__(self):
        self.paused, self.quit = threading.Event(), threading.Event()


def run_game(me, misclicks=0, timeout=120, **screen_opts):
    cfg = {**common.load_config(), **FAST}
    with engine.Engine.from_config(cfg) as mine, engine.Engine.from_config(cfg) as opp:
        screen = FakeScreen(opp, main.other(me), bottom=me, misclicks=misclicks, **screen_opts)
        keys = FakeKeys()
        game = main.Game(cfg, screen, screen, mine, keys, colour=None)
        result = {}

        def target():
            try:
                game.run()
                result["end"] = "stopped"
            except main.Quit:
                result["end"] = "quit"
        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            keys.quit.set()
            t.join(5)
            result.setdefault("end", "timeout -> quit hotkey")
        moves = len(screen.opp_history)
        return result["end"], moves, keys.paused.is_set(), screen


def main_():
    logging.basicConfig(level=logging.INFO, format="  %(levelname)-7s %(message)s")
    main.log.propagate = True
    for me in ("red", "black"):
        print(f"\n=== full game, I play {me} (strong) vs weak opponent")
        end, n, paused, screen = run_game(me)
        print(f"=> {end} after {n} plies, paused={paused}")
        print(vision.Reading(screen.board, me).text())
    print("\n=== game-over panel covers the board right after mate: should stop")
    # Position from a real PlayOK game log: Red mates with g3g10.
    mate_in_1 = "r1b1kabn1/4a2Rr/2n1c4/p1p5p/4C4/2P3P2/P3P3N/6C2/9/RNBAKAB2"
    end, n, paused, screen = run_game("red", timeout=15, cover_on_mate=True, start_fen=mate_in_1)
    print(f"=> {end}, moves played {screen.opp_history}, clicks={screen.plays} (want 1), paused={paused}")
    print("\n=== PlayOK end-of-game panel right after mate: should stop as game over")
    end, n, paused, screen = run_game("red", timeout=15, cover_on_mate=True, panel=True, start_fen=mate_in_1)
    print(f"=> {end}, moves played {screen.opp_history}, clicks={screen.plays} (want 1), paused={paused}")
    print("\n=== opponent resigns mid-game (panel while waiting): should stop")
    end, n, paused, screen = run_game("black", timeout=15, panel=True, resign_after=12)
    print(f"=> {end} after {n} plies, clicks={screen.plays}, paused={paused}")
    print("\n=== board covered after an ordinary move: should pause without re-clicking")
    end, n, paused, screen = run_game("red", timeout=8, cover_after=3)
    print(f"=> {end}, clicks={screen.plays} (want 3), paused={paused}")
    print("\n=== one misclick: should retry and continue")
    end, n, paused, _ = run_game("red", misclicks=1, timeout=8)
    print(f"=> {end}, plies {n}, paused={paused}")
    print("\n=== two misclicks: should pause and alert")
    end, n, paused, _ = run_game("red", misclicks=2, timeout=8)
    print(f"=> {end}, plies {n}, paused={paused}")


if __name__ == "__main__":
    main_()
