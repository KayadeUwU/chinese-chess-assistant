# Xiangqi assistant

Reads a Chinese chess (Xiangqi) board from the screen, analyses it with
Fairy-Stockfish, and can play moves with the mouse. Built for PlayOK
(playok.com/vi/cotuong) on Windows.

## Setup

1. Python 3.11+, then:

   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```

2. Download the Xiangqi build of Fairy-Stockfish (NNUE built in) into `engine/`:
   [fairy-stockfish-largeboard_x86-64-bmi2.exe](https://github.com/fairy-stockfish/Fairy-Stockfish/releases/tag/fairy_sf_14_0_1_xq)
   (use the `-modern` or plain `x86-64` build on older CPUs and set `engine_path` in `config.json`).

3. Start the control panel and press **Calibrate** with the board in the
   starting position:

   ```
   .venv\Scripts\python main.py
   ```

## Using it

- **Read Board** – recognise the board once.
- **Analyze Only** – live analysis with arrows and an eval bar; never clicks.
- **Play Best Move** – plays one move when it is your turn.
- **Auto Play / Pause** – plays until the game ends.
- **Stop** – halts everything.

Global hotkeys (work while the browser has focus): **F9** pause/resume,
**F10** stop. Mouse fail-safe: push the mouse into a corner of the primary
monitor. Keep the panel window off the board: the board is unreadable under
it, and moves are refused while it overlaps.

Headless loop without the GUI: `python main.py --cli` (`--no-click` to only print moves).

## Modules

| File | Job |
|---|---|
| `calibrate.py` | grid from two hover points, piece templates, `config.json` |
| `vision.py` | screen → board, FEN, sanity checks, game-over panel |
| `engine.py` | Fairy-Stockfish over UCI: timed search, MultiPV, live analysis |
| `controller.py` | board squares → mouse clicks/drags |
| `game.py` | the play loop (wait for turn, think, move, confirm) |
| `gui.py` | PySide6 control panel |
| `notation.py` | engine moves → Chinese notation (炮二平五) |
| `sim_game.py` | offline simulation of the play loop against a second engine |

Logs and screenshots of misreads go to `logs/`.
