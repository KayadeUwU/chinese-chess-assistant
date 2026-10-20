"""Fairy-Stockfish wrapper (UCI over a subprocess).

Move notation: files a-i, ranks 1-10 (rank 1 = Red's back rank), so a move
like "h10g8" has two-character ranks. Board indices are (row, col) with
row 0 = rank 10 and col 0 = file a, as in common.py.

Command ordering: only one search runs at a time. Starting a new search, or
changing an option, first sends "stop" and waits for that search's
"bestmove", so the engine never receives commands out of order. An Engine
must be used from one thread only.

    python engine.py                        best move from the start position
    python engine.py --fen "<fen>" --movetime 2000 --multipv 3
    python engine.py --from-screen          read the live board with vision.py and ask
"""
import argparse
import queue
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

import common

START_FEN = common.START_BOARD_FEN + " w - - 0 1"
MOVE_RE = re.compile(r"^([a-i])(10|[1-9])([a-i])(10|[1-9])$")


@dataclass
class Move:
    uci: str
    src: tuple   # (row, col)
    dst: tuple   # (row, col)


@dataclass
class PVLine:
    multipv: int
    depth: int
    score_cp: int = None      # from the side to move's point of view
    mate_in: int = None       # moves to mate; negative = getting mated
    pv: list = field(default_factory=list)


@dataclass
class SearchResult:
    move: Move = None         # None when the side to move has no legal move
    score_cp: int = None      # of the best line, from the side to move's point of view
    mate_in: int = None
    depth: int = None
    used_history: bool = False
    lines: dict = field(default_factory=dict)   # multipv number -> PVLine
    nodes: int = None
    nps: int = None
    finished: bool = False


def square_to_index(sq):
    return 10 - int(sq[1:]), ord(sq[0]) - ord("a")


def index_to_square(row, col):
    return common.square_name(row, col)


def parse_move(uci):
    m = MOVE_RE.match(uci)
    if not m:
        raise ValueError(f"not a xiangqi move: {uci!r}")
    f1, r1, f2, r2 = m.groups()
    return Move(uci, square_to_index(f1 + r1), square_to_index(f2 + r2))


def apply_move(board, move):
    """Return a copy of a 10x9 board with `move` played (captures overwrite)."""
    new = [row[:] for row in board]
    (r1, c1), (r2, c2) = move.src, move.dst
    new[r2][c2], new[r1][c1] = new[r1][c1], "."
    return new


def parse_info(line):
    """Fields of a UCI 'info' line as a dict (pv as a list), or None."""
    t = line.split()
    if not t or t[0] != "info" or "string" in t[:2]:
        return None
    info, i = {}, 1
    while i < len(t):
        key = t[i]
        if key == "pv":
            info["pv"] = t[i + 1:]
            break
        if key == "score" and i + 2 < len(t):
            info["score_kind"], info["score"] = t[i + 1], int(t[i + 2])
            i += 3
            while i < len(t) and t[i] in ("lowerbound", "upperbound"):
                info["bound"] = t[i]
                i += 1
            continue
        if key in ("depth", "seldepth", "multipv", "nodes", "nps", "time", "hashfull") and i + 1 < len(t):
            try:
                info[key] = int(t[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        i += 1
    return info


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, path, threads=1, hash_mb=64):
        path = common.resolve_path(path)
        if not path.exists():
            raise EngineError(f"engine not found: {path} (set engine_path in config.json)")
        self.proc = subprocess.Popen(
            [str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.lines = queue.Queue()
        self.name = "engine"
        self.options = {}
        self.searching = False
        self.current = None
        threading.Thread(target=self._pump, daemon=True).start()
        self.send("uci")

        def on_line(line):
            if line.startswith("id name "):
                self.name = line[8:]
        self._wait_for(lambda l: l == "uciok", 10, on_line)
        self.set_option("UCI_Variant", "xiangqi")
        self.set_option("Threads", threads)
        self.set_option("Hash", hash_mb)
        self.ready()

    @classmethod
    def from_config(cls, cfg=None):
        cfg = cfg or common.load_config()
        return cls(cfg["engine_path"], cfg.get("engine_threads", 1), cfg.get("engine_hash_mb", 64))

    # ------------------------------------------------------------ plumbing --

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.strip())
        self.lines.put(None)  # EOF

    def send(self, cmd):
        if self.proc.poll() is not None:
            raise EngineError(f"engine exited (code {self.proc.returncode})")
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()

    def _wait_for(self, done, timeout, on_line=None):
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise EngineError("engine did not answer in time") from None
            if line is None:
                raise EngineError("engine exited unexpectedly")
            if on_line:
                on_line(line)
            if done(line):
                return line

    def ready(self):
        self.stop_search()
        self.send("isready")
        self._wait_for(lambda l: l == "readyok", 10)

    def set_option(self, name, value):
        """Set a UCI option (stops any running search first). Skips unchanged values."""
        if self.options.get(name) == str(value):
            return
        self.stop_search()
        self.send(f"setoption name {name} value {value}")
        self.options[name] = str(value)

    def new_game(self):
        self.stop_search()
        self.send("ucinewgame")
        self.ready()

    # ------------------------------------------------------------ position --

    def engine_fen(self):
        """The engine's current position as a FEN (from the 'd' command)."""
        self.stop_search()
        self.send("d")
        fen = []
        self._wait_for(lambda l: l.startswith("Checkers:"), 5,
                       lambda l: fen.append(l[5:]) if l.startswith("Fen: ") else None)
        return fen[-1] if fen else None

    def set_position(self, fen, history=None):
        """Load `fen`, preferably as history = (base_fen, [moves]) so the engine knows
        about repetitions. The history is used only if it ends exactly in `fen`."""
        self.stop_search()
        if history and history[1]:
            base, moves = history
            self.send(f"position fen {base} moves {' '.join(moves)}")
            echoed = self.engine_fen()
            if echoed and echoed.split()[:2] == fen.split()[:2]:
                return True
        self.send(f"position fen {fen}")
        return False

    # ------------------------------------------------------------- search --

    def begin_search(self, fen, history=None, movetime_ms=None, depth=None, multipv=1):
        """Start searching `fen` and return at once; follow it with poll().

        No movetime and no depth means "go infinite" (analysis): it runs until
        stop_search() or the next begin_search().
        """
        self.stop_search()
        self.set_option("MultiPV", max(1, int(multipv)))
        used = self.set_position(fen, history)
        go = ["go"]
        if movetime_ms:
            go += ["movetime", str(int(movetime_ms))]
        if depth:
            go += ["depth", str(int(depth))]
        if len(go) == 1:
            go.append("infinite")
        self.current = SearchResult(used_history=used)
        self.send(" ".join(go))
        self.searching = True
        return self.current

    def poll(self, timeout=0.05):
        """Read engine output for up to `timeout` seconds.

        Updates self.current. Returns (changed, finished): whether any info
        arrived, and whether the search has ended with "bestmove".
        """
        if not self.searching:
            return False, True
        deadline, changed = time.monotonic() + timeout, False
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return changed, False
            if line is None:
                self.searching = False
                raise EngineError("engine exited unexpectedly")
            if line.startswith("bestmove"):
                parts = line.split()
                best = parts[1] if len(parts) > 1 else "(none)"
                if best not in ("(none)", "0000"):
                    self.current.move = parse_move(best)
                self.current.finished = True
                self.searching = False
                return True, True
            changed |= self._take_info(line)

    def _take_info(self, line):
        info = parse_info(line)
        if not info:
            return False
        r = self.current
        r.nodes, r.nps = info.get("nodes", r.nodes), info.get("nps", r.nps)
        if "score" not in info or "pv" not in info or info.get("bound"):
            return "nps" in info
        n = info.get("multipv", 1)
        kind, value = info["score_kind"], info["score"]
        r.lines[n] = PVLine(n, info.get("depth", 0), value if kind == "cp" else None,
                            value if kind == "mate" else None, info["pv"])
        if n == 1:
            r.depth, r.score_cp, r.mate_in = r.lines[1].depth, r.lines[1].score_cp, r.lines[1].mate_in
        return True

    def stop_search(self):
        """Stop the running search (if any) and wait for its bestmove."""
        if not self.searching:
            return self.current
        self.send("stop")
        deadline = time.monotonic() + 10
        while self.searching:
            if time.monotonic() > deadline:
                raise EngineError("engine did not answer 'stop'")
            self.poll(0.1)
        return self.current

    def search(self, fen, movetime_ms=1000, history=None, depth=None, multipv=1,
               on_info=None, should_abort=None):
        """Blocking timed search. Returns the finished SearchResult.

        on_info(result) is called as lines arrive (at most ~20 times a second);
        should_abort() is checked as well and stops the search early.
        """
        self.begin_search(fen, history, movetime_ms, depth, multipv)
        deadline = time.monotonic() + movetime_ms / 1000 + 10
        while True:
            changed, finished = self.poll(0.05)
            if changed and on_info:
                on_info(self.current)
            if finished:
                return self.current
            if (should_abort and should_abort()) or time.monotonic() > deadline:
                return self.stop_search()

    def get_best_move(self, fen, movetime_ms=1000):
        """Best move as a Move, or None when there is no legal move (mate/stalemate)."""
        return self.search(fen, movetime_ms).move

    def close(self):
        if self.proc.poll() is None:
            try:
                self.stop_search()
                self.send("quit")
                self.proc.wait(timeout=3)
            except (EngineError, OSError, subprocess.TimeoutExpired):
                self.proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def describe(result):
    if result.move is None:
        return "no legal move (checkmate or stalemate)"
    m = result.move
    score = (f"mate in {result.mate_in}" if result.mate_in is not None
             else f"{result.score_cp:+d} cp" if result.score_cp is not None else "?")
    return (f"bestmove {m.uci}  =  {index_to_square(*m.src)} {m.src} -> "
            f"{index_to_square(*m.dst)} {m.dst}   score {score}, depth {result.depth}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fen", default=START_FEN)
    ap.add_argument("--movetime", type=int, help="ms (default: movetime_ms in config.json)")
    ap.add_argument("--multipv", type=int, default=1)
    ap.add_argument("--from-screen", action="store_true")
    args = ap.parse_args()
    cfg = common.load_config()
    movetime = args.movetime or cfg["movetime_ms"]

    fen = args.fen
    if args.from_screen:
        import vision
        reading = vision.BoardReader(cfg).read_stable()
        vision.report(reading)
        if not reading or not reading.ok:
            raise SystemExit("Board not readable - not asking the engine.")
        fen = reading.fen(reading.side_to_move or cfg["my_color"])

    with Engine.from_config(cfg) as eng:
        print(f"{eng.name}\nposition: {fen}")
        t = time.perf_counter()
        result = eng.search(fen, movetime, multipv=args.multipv)
        print(f"{describe(result)}   ({time.perf_counter() - t:.2f}s, {result.nps or 0:,} nps)")
        for n, line in sorted(result.lines.items()):
            score = f"M{line.mate_in}" if line.mate_in is not None else f"{line.score_cp:+d}"
            print(f"  {n}. {score:>6}  {' '.join(line.pv[:8])}")


if __name__ == "__main__":
    main()
