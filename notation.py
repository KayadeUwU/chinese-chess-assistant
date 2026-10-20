"""Engine moves (h3e3) -> traditional Chinese notation (炮二平五).

Red counts files from its right with Chinese numerals (file i = 一 ... a = 九);
Black counts from its own right with digits (file a = １ ... i = ９).
進 = forward, 退 = back, 平 = sideways. Straight movers (車 炮 兵 帥) give the
number of ranks for 進/退; diagonal movers (馬 相 仕) give the destination file.
Identical pieces on the same file are told apart with 前/後 (前/中/後, or
一二三... for more pawns) instead of the file number.
"""
import engine

RED_NUM = "一二三四五六七八九"
BLACK_NUM = "１２３４５６７８９"
NAMES = {"R": "車", "N": "馬", "B": "相", "A": "仕", "K": "帥", "C": "炮", "P": "兵",
         "r": "車", "n": "馬", "b": "象", "a": "士", "k": "將", "c": "砲", "p": "卒"}
DIAGONAL = set("NBAnba")


def _file_number(col, red):
    return 9 - col if red else col + 1


def _num(n, red):
    return (RED_NUM if red else BLACK_NUM)[n - 1]


def to_chinese(board, move):
    """One move in Chinese notation, given the board before it. `move` is a Move or uci."""
    if isinstance(move, str):
        move = engine.parse_move(move)
    (r1, c1), (r2, c2) = move.src, move.dst
    piece = board[r1][c1]
    if piece in ".?":
        raise ValueError(f"no piece on {move.uci[:2]}")
    red = piece.isupper()
    name = NAMES[piece]

    # Same pieces on the same file, ordered front to back from the mover's side.
    same = sorted((r for r in range(10) if board[r][c1] == piece), reverse=not red)
    if len(same) == 1:
        head = name + _num(_file_number(c1, red), red)
    else:
        idx = same.index(r1)
        if len(same) == 2:
            tag = "前後"[idx]
        elif len(same) == 3:
            tag = "前中後"[idx]
        else:
            tag = RED_NUM[idx]
        head = tag + name

    forward = (r2 < r1) if red else (r2 > r1)
    if r1 == r2:
        return head + "平" + _num(_file_number(c2, red), red)
    action = "進" if forward else "退"
    if piece in DIAGONAL:
        return head + action + _num(_file_number(c2, red), red)
    return head + action + _num(abs(r2 - r1), red)


def line_to_chinese(board, moves):
    """A list of uci moves played in order from `board`; stops at the first bad move."""
    out, b = [], [row[:] for row in board]
    for uci in moves:
        try:
            m = engine.parse_move(uci)
            out.append(to_chinese(b, m))
        except ValueError:
            break
        b = engine.apply_move(b, m)
    return out


if __name__ == "__main__":
    import common
    start = common.board_from_fen(common.START_BOARD_FEN)
    for uci, want in [("h3e3", "炮二平五"), ("h10g8", "馬８進７"), ("b1c3", "馬八進七"),
                      ("i1i2", "車一進一"), ("g4g5", "兵三進一"), ("c1e3", "相七進五"),
                      ("f1e2", "仕四進五"), ("b8e8", "砲２平５"), ("e1e2", "帥五進一"),
                      ("e7e6", "卒５進１"), ("a10a9", "車１進１")]:
        got = to_chinese(start, uci)
        print(f"{uci:>6} {got}  {'OK' if got == want else 'WRONG, want ' + want}")
    # two cannons on one file: 前炮 / 後炮
    b = common.board_from_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1C1P1P/1C2C4/9/RNBAKABNR")
    for uci, want in [("e4e7", "前炮進三"), ("e3d3", "後炮平六")]:
        got = to_chinese(b, uci)
        print(f"{uci:>6} {got}  {'OK' if got == want else 'WRONG, want ' + want}")
    print(" ".join(line_to_chinese(start, ["h3e3", "h8e8", "h1g3", "h10g8"])))
