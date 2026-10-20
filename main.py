"""Entry point.

    python main.py                  open the control panel (GUI)
    python main.py --cli            headless loop: play; my colour = the colour at the bottom
    python main.py --cli --no-click only print/log the engine's moves, you play them
    python main.py --cli --movetime 2000 --color black

Hotkeys (global, work while the browser has focus): F9 pause/resume,
F10 stop (configurable in config.json).
Logs: logs/game_<time>.log, plus a screenshot of the board for every misread.
"""
import argparse
import os
import threading
import time

import common  # must be imported first: sets DPI awareness

import controller
import engine
import vision
from game import Game, Hotkeys, Quit, log, setup_logging


def run_cli(args):
    cfg = common.load_config()
    if args.movetime:
        cfg["movetime_ms"] = args.movetime
    log_path = setup_logging()

    def hard_quit():
        # Give the loop a moment to stop on its own, then exit no matter what.
        def kill():
            time.sleep(1.5)
            controller.pyautogui.mouseUp()
            os._exit(0)
        threading.Thread(target=kill, daemon=True).start()

    keys = Hotkeys(cfg, on_quit=hard_quit)
    log.info("Log: %s", log_path)
    log.info("Hotkeys: %s pause/resume, %s quit. Mouse fail-safe: top-right corner of the primary monitor.",
             cfg["hotkeys"]["pause"].upper(), cfg["hotkeys"]["quit"].upper())
    with engine.Engine.from_config(cfg) as eng:
        eng.new_game()
        game = Game(cfg, vision.BoardReader(cfg), controller.Controller(cfg), eng, keys,
                    colour=args.color, click=not args.no_click)
        try:
            game.run()
        except Quit:
            log.info("Quit.")
        except controller.pyautogui.FailSafeException:
            log.warning("Mouse fail-safe triggered - stopped.")
    log.info("Stopped.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cli", action="store_true", help="run the game loop without the GUI")
    ap.add_argument("--color", choices=["red", "black"], help="(cli) my colour (default: whoever is at the bottom)")
    ap.add_argument("--movetime", type=int, help="(cli) engine ms per move (default from config.json)")
    ap.add_argument("--no-click", action="store_true", help="(cli) don't touch the mouse, just show moves")
    args = ap.parse_args()
    if args.cli:
        run_cli(args)
    else:
        import gui
        raise SystemExit(gui.main())


if __name__ == "__main__":
    main()
