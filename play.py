"""Play Redactle in the terminal on an article from data/articles.jsonl.

Usage:
  python play.py                  # random article
  python play.py --split test     # random article from one split
  python play.py --title "Fay Peak"
  python play.py --show           # just print the redacted article and exit

Hidden words show their length in the middle of the blocks: ██7██ = 7 letters.
Commands while playing:  :q quit   :giveup reveal answer   :history list guesses
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from redact import (count_matches, is_always_visible, normalize, redact,
                    tokenize)

USE_COLOR = sys.stdout.isatty()


def color(s: str, code: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def render(text: str, revealed: set[str], highlight: str | None,
           reveal_numbers: bool) -> str:
    """Redacted text with hidden words dimmed and the latest guess highlighted."""
    out = []
    for piece, is_word in tokenize(text):
        key = normalize(piece) if is_word else None
        if is_word and key == highlight:
            out.append(color(piece, "1;30;43"))  # black on yellow
        elif not is_word or key in revealed or is_always_visible(piece, reveal_numbers):
            out.append(piece)
        else:
            out.append(color(redact(piece, show_lengths=True), "90"))
    return "".join(out)


def load_articles(path: Path, split: str | None) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found - run `python data.py` first.")
    with path.open() as f:
        arts = [json.loads(line) for line in f]
    return [a for a in arts if split is None or a["split"] == split]


def screen(art: dict, revealed: set[str], highlight: str | None,
           history: list[tuple[str, int]], msg: str, reveal_numbers: bool) -> None:
    if USE_COLOR:
        os.system("clear")
    print(color(render(art["title"], revealed, highlight, reveal_numbers), "1"))
    print("=" * 60)
    print(render(art["text"], revealed, highlight, reveal_numbers))
    print("=" * 60)
    last = ", ".join(f"{w} ({n})" for w, n in history[-8:])
    print(f"guesses: {len(history)}   recent: {last}")
    if msg:
        print(msg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/articles.jsonl")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--title", help="play a specific article by title")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--reveal-numbers", action="store_true",
                    help="show numbers from the start (modern redactle.net)")
    ap.add_argument("--show", action="store_true",
                    help="print the redacted article and exit")
    args = ap.parse_args()

    arts = load_articles(Path(args.data), args.split)
    if args.title:
        arts = [a for a in arts if a["title"].lower() == args.title.lower()]
        if not arts:
            sys.exit(f"no article titled {args.title!r}")
    art = random.Random(args.seed).choice(arts)

    if args.show:
        print(render(art["title"], set(), None, args.reveal_numbers))
        print(render(art["text"], set(), None, args.reveal_numbers))
        return

    targets = set(art["target_keys"])
    revealed: set[str] = set()
    history: list[tuple[str, int]] = []
    highlight, msg = None, "Type a word to guess."

    while True:
        screen(art, revealed, highlight, history, msg, args.reveal_numbers)
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
            msg = "\n".join(f"{i+1:3}. {w} ({n})" for i, (w, n) in enumerate(history))
            continue
        if guess == ":giveup":
            print(f"\nThe answer was: {color(art['title'], '1;32')}  ({art['url']})")
            return

        key = normalize(guess)
        if " " in guess or not key:
            msg = "Guess one word at a time."
        elif is_always_visible(guess, args.reveal_numbers):
            msg = f"'{guess}' is always visible."
        elif key in revealed or any(w == key for w, _ in history):
            msg = f"Already guessed '{guess}'."
        else:
            n = count_matches(art["title"], key) + count_matches(art["text"], key)
            history.append((key, n))
            revealed.add(key)
            highlight = key
            msg = f"'{guess}': {n} match{'es' if n != 1 else ''}."
            if targets <= revealed:
                screen(art, revealed | {normalize(p) for p, w in tokenize(art["text"]) if w},
                       None, history, "", args.reveal_numbers)
                print(color(f"Solved '{art['title']}' in {len(history)} guesses!", "1;32"))
                return


if __name__ == "__main__":
    main()
