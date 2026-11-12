"""Play Redactle in the terminal (uses the same RedactleEnv as the agents).

Usage:
  python play.py                  # random article
  python play.py --split test     # random article from one split
  python play.py --max-level 3    # only the ~1,000 most famous topics
  python play.py --title "Fay Peak"
  python play.py --show           # just print the redacted article and exit

Hidden words show their length in the middle of the blocks: ██7██ = 7 letters.
Commands while playing:  :q quit   :giveup reveal answer   :history list guesses
"""

from __future__ import annotations

import argparse
import os
import sys

from env import DEFAULT_DATA, Observation, RedactleEnv, load_articles
from redact import hidden

USE_COLOR = sys.stdout.isatty()


def color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def render(seq: list, keys: list, highlight: str | None) -> str:
    """Hidden words dimmed with their length; the latest guess highlighted."""
    out = []
    for x, key in zip(seq, keys):
        if isinstance(x, int):
            out.append(color(hidden("x" * x, show_length=True), "90"))
        elif highlight and key == highlight:
            out.append(color(x, "1;30;43"))  # black on yellow
        else:
            out.append(x)
    return "".join(out)


def screen(env: RedactleEnv, obs: Observation, highlight: str | None, msg: str):
    keys = [t.key for t in env.puzzle.tokens]
    n = env.puzzle.n_title
    if USE_COLOR:
        os.system("clear")
    print(color(render(obs.title, keys[:n], highlight), "1"))
    print("=" * 60)
    print(render(obs.body, keys[n:], highlight))
    print("=" * 60)
    recent = ", ".join(f"{g.word} ({g.count})" for g in obs.history if g.valid)
    print(f"guesses: {obs.n_guesses}   recent: {recent[-200:]}")
    if msg:
        print(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--max-level", type=int, choices=[1, 2, 3, 4],
                    help="only Vital Articles at this level or lower (3 = top 1,000)")
    ap.add_argument("--title", help="play a specific article by title")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--max-guesses", type=int, default=500)
    ap.add_argument("--reveal-numbers", action="store_true",
                    help="show numbers from the start (modern redactle.net)")
    ap.add_argument("--show", action="store_true",
                    help="print the redacted article and exit")
    args = ap.parse_args()

    try:
        arts = load_articles(args.data, args.split, args.max_level)
    except FileNotFoundError as e:
        sys.exit(str(e))
    if not arts:
        sys.exit("No articles match those options. If you used --max-level, "
                 "rebuild the dataset with `python data.py` (vital articles).")

    env = RedactleEnv(arts, max_guesses=args.max_guesses, count_invalid=False,
                      reveal_numbers=args.reveal_numbers, seed=args.seed)
    try:
        obs = env.reset(args.title)
    except KeyError as e:
        sys.exit(str(e))

    if args.show:
        title, body = obs.render()
        print(title)
        print(body)
        return

    highlight, msg = None, "Type a word to guess."
    while True:
        screen(env, obs, highlight, msg)
        try:
            guess = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not guess:
            continue
        if guess == ":q":
            return
        if guess == ":history":
            msg = "\n".join(f"{i+1:3}. {g.word} ({g.count})"
                            for i, g in enumerate(g for g in obs.history if g.valid))
            continue
        if guess == ":giveup":
            art = env.puzzle.article
            print(f"\nThe answer was: {color(art['title'], '1;32')}  ({art['url']})")
            return

        obs, reward, done, info = env.step(guess)
        msg = {
            "not_a_word": "Guess one word at a time.",
            "always_visible": f"'{guess}' is always visible.",
            "already_guessed": f"Already revealed '{guess}' (same word family "
                               "as an earlier guess).",
        }.get(info["reason"], f"'{guess}': {info['count']} "
                              f"match{'es' if info['count'] != 1 else ''}.")
        highlight = env.history[-1].key if info["valid"] else None

        if info["solved"]:
            obs = env.reveal_all()
            screen(env, obs, None, "")
            print(color(f"Solved '{info['title']}' in {obs.n_guesses} guesses!", "1;32"))
            return
        if info["truncated"]:
            screen(env, obs, None, "")
            print(f"Out of guesses. The answer was: {info['title']}")
            return


if __name__ == "__main__":
    main()
