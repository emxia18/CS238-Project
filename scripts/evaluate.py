"""Evaluate an agent on a split of the Redactle dataset.

Usage (from the repo root):
  python -m scripts.evaluate --agent frequency --split val
  python -m scripts.evaluate --agent random --split val --n 100
  python -m scripts.evaluate --agent belief --n 100 --set endgame_conf=0.5 \
      --tag endgame0.5 --out-dir results/sweeps

Prints summary metrics and writes one row per episode to
results/<agent>[_<tag>]_<split>.csv. --set passes keyword arguments to the
agent's constructor (values are parsed as Python literals when possible).
"""

from __future__ import annotations

import argparse
import ast
import csv
import statistics
import time
from pathlib import Path

from redactle.agents import AGENTS, Agent, load_vocab, make_agent
from redactle.env import RedactleEnv, load_articles
from redactle.paths import RESULTS_DIR

BUDGETS = (50, 100, 200, 500)


def run_episode(env: RedactleEnv, agent: Agent, article: dict) -> dict:
    obs = env.reset(article)
    agent.reset()
    total, done, info = 0.0, False, {}
    while not done:
        obs, reward, done, info = env.step(agent.act(obs))
        total += reward
    return {
        "title": article["title"],
        "level": article.get("level"),
        "n_title_words": len(env.puzzle.targets),
        "solved": info["solved"],
        "guesses": obs.n_guesses,
        "return": total,
        "title_words_found": len(env.puzzle.targets & obs.revealed),
        "zero_hit_guesses": sum(g.valid and g.count == 0 for g in obs.history),
    }


def summarize(rows: list[dict], max_guesses: int) -> None:
    n = len(rows)
    solved = [r for r in rows if r["solved"]]
    capped = [r["guesses"] if r["solved"] else max_guesses for r in rows]
    print(f"episodes:            {n}")
    print(f"success rate:        {len(solved) / n:.1%}")
    for b in BUDGETS:
        if b <= max_guesses:
            rate = sum(r["solved"] and r["guesses"] <= b for r in rows) / n
            print(f"  solved within {b:3}: {rate:.1%}")
    print(f"mean capped cost:    {statistics.mean(capped):.1f}  (failures = {max_guesses})")
    if solved:
        print(f"median guesses (solved): {statistics.median(r['guesses'] for r in solved):.0f}")
    print(f"mean return:         {statistics.mean(r['return'] for r in rows):.1f}")
    zero = sum(r["zero_hit_guesses"] for r in rows) / sum(r["guesses"] for r in rows)
    print(f"guesses revealing 0: {zero:.1%}")
    frac = statistics.mean(r["title_words_found"] / r["n_title_words"] for r in rows)
    print(f"title words found:   {frac:.1%} on average")
    levels = sorted({r["level"] for r in rows if r["level"]})
    for lv in levels:
        sub = [r for r in rows if r["level"] == lv]
        print(f"  level {lv}: {sum(r['solved'] for r in sub) / len(sub):.1%} solved "
              f"({len(sub)} articles)")


def parse_kwargs(pairs: list[str]) -> dict:
    """['alpha=20', 'endgame_conf=None'] -> {'alpha': 20, 'endgame_conf': None}"""
    out = {}
    for pair in pairs:
        key, _, value = pair.partition("=")
        try:
            out[key] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            out[key] = value
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=list(AGENTS), required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n", type=int, help="evaluate only the first n articles")
    ap.add_argument("--max-level", type=int, choices=[1, 2, 3, 4])
    ap.add_argument("--max-guesses", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=RESULTS_DIR)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="agent constructor arguments, e.g. alpha=10")
    ap.add_argument("--tag", help="label added to the output filename")
    args = ap.parse_args()

    articles = load_articles(split=args.split, max_level=args.max_level)[:args.n]
    agent = make_agent(args.agent, seed=args.seed, **parse_kwargs(args.set))
    env = RedactleEnv(articles, max_guesses=args.max_guesses, seed=args.seed)

    # Upper bound for vocabulary-based agents: titles whose words are all in
    # the agent's training vocabulary.
    vocab_keys = {v["key"] for v in load_vocab(min_df=2)}
    reachable = sum(env._get_puzzle(a).targets <= vocab_keys for a in articles)

    start = time.time()
    rows = [run_episode(env, agent, a) for a in articles]
    print(f"\n== {args.agent} on {args.split} ({time.time() - start:.0f}s) ==")
    summarize(rows, args.max_guesses)
    print(f"reachable ceiling:   {reachable / len(articles):.1%} of titles have all "
          f"words in the vocabulary")

    name = f"{args.agent}_{args.tag}" if args.tag else args.agent
    out = Path(args.out_dir) / f"{name}_{args.split}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-episode results -> {out}")


if __name__ == "__main__":
    main()
