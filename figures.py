"""Figures for the write-up. Run after evaluate.py has produced results/*.csv.

  python figures.py                     # all figures -> figures/
  python figures.py --example Telescope # a different example game

Outputs:
  figures/example_<title>_history.png   guesses + match counts for one game
  figures/example_<title>_steps.png     belief and top next-word scores at
                                         a few steps of that game
  figures/example_<title>_text.txt      redacted text before / mid / end
  figures/success_vs_budget.png         % solved vs guess budget, per agent
  figures/guesses_to_solve.png          distribution of guesses (belief agent)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from agents import make_agent  # noqa: E402
from env import RedactleEnv, load_articles  # noqa: E402

# Palette (validated: categorical slots 1-3 of the reference palette).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#c9c8c2"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
OUT = Path("figures")

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Example game
# ---------------------------------------------------------------------------

def trace_game(title: str, split: str = "val") -> dict:
    """Play one game with the belief agent, recording its internals."""
    train = load_articles(split="train")
    env = RedactleEnv(load_articles(split=split))
    agent = make_agent("belief")
    obs = env.reset(title)
    agent.reset()
    steps, snapshots, done = [], [obs], False
    while not done:
        word = agent.act(obs)
        score, top, w = agent.last_score, agent.last_top, agent.last_w
        cand = [(agent.vocab[k]["word"], agent.vocab[k]["key"], float(score[k]))
                for k in np.argsort(-score)[:8]]
        belief = [(train[top[i]]["title"], float(w[i])) for i in np.argsort(-w)[:5]]
        obs, _, done, info = env.step(word)
        g = env.history[-1]
        steps.append({"word": word, "count": info["count"],
                      "title_hit": g.key in env.puzzle.targets,
                      "candidates": cand, "belief": belief})
        snapshots.append(obs)
    return {"title": title, "targets": env.puzzle.targets, "steps": steps,
            "snapshots": snapshots}


def plot_history(game: dict, path: Path) -> None:
    steps = game["steps"]
    x = np.arange(1, len(steps) + 1)
    counts = [s["count"] for s in steps]
    colors = [ORANGE if s["title_hit"] else BLUE for s in steps]
    fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(steps) + 1.5), 3.2))
    ax.bar(x, counts, width=0.7, color=colors, edgecolor="white", linewidth=1)
    ax.set_xticks(x, [s["word"] for s in steps], rotation=60, ha="right")
    ax.set_ylabel("words revealed")
    ax.set_xlim(0.4, len(steps) + 0.6)
    ax.grid(axis="x", visible=False)
    for xi, c in zip(x, counts):
        if c == 0:
            ax.text(xi, max(counts) * 0.01, "0", ha="center", va="bottom",
                    color=INK2, fontsize=8)
    ax.set_title(f"Belief agent on “{game['title']}”: solved in "
                 f"{len(steps)} guesses", loc="left")
    ax.text(0, -0.42, "Bars: matches per guess, in order.  Orange = title word.",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def pick_steps(game: dict, n: int = 4) -> list[int]:
    """Steps (0-based) to show: early, where the belief first concentrates,
    the step after, and the final guess."""
    steps = game["steps"]
    peak = next((i for i, s in enumerate(steps) if s["belief"][0][1] > 0.1),
                len(steps) // 2)
    chosen = sorted({min(5, len(steps) - 1), peak, min(peak + 1, len(steps) - 1),
                     len(steps) - 1})
    return chosen[-n:]


def plot_steps(game: dict, path: Path) -> None:
    idx = pick_steps(game)
    fig, axes = plt.subplots(2, len(idx), figsize=(3.3 * len(idx), 5.6),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.75})
    for col, i in enumerate(idx):
        s = game["steps"][i]
        # Row 1: belief over training articles.
        ax = axes[0, col]
        names = [b[0] for b in s["belief"]][::-1]
        vals = [b[1] for b in s["belief"]][::-1]
        ax.barh(names, vals, color=AQUA, height=0.65, edgecolor="white")
        ax.set_xlim(0, 1)
        ax.set_title(f"Before guess {i + 1}", loc="left", fontsize=10)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
        for y, v in enumerate(vals):
            ax.text(min(v + 0.03, 0.80), y, f"{v:.2f}", va="center",
                    fontsize=8, color=INK2)
        # Row 2: top candidate next words.
        ax = axes[1, col]
        cand = s["candidates"][::-1]
        colors = [ORANGE if c[1] in game["targets"]
                  else BLUE if c[0] == s["word"] else GRAY for c in cand]
        ax.barh([c[0] for c in cand], [c[2] for c in cand], color=colors,
                height=0.65, edgecolor="white")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.set_xlabel("score (expected matches)", fontsize=8.5)
        ax.set_title(f"guessed “{s['word']}” → {s['count']}",
                     loc="left", fontsize=10)
    axes[0, 0].set_ylabel("most similar\ntraining articles", fontsize=9)
    axes[1, 0].set_ylabel("top-scoring\nnext words", fontsize=9)
    fig.suptitle(f"What the belief agent thinks while solving "
                 f"“{game['title']}”", x=0.05, ha="left", fontsize=12)
    fig.text(0.05, -0.02, "Top: belief weight on the most similar training "
             "articles.  Bottom: highest-scoring next words; blue = word "
             "guessed, orange = title word, gray = other candidates.",
             color=INK2, fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def write_text(game: dict, path: Path, words: int = 90) -> None:
    from redact import hidden

    def render(obs, n):
        out, k = [], 0
        for x in obs.title + ["\n"] + obs.body:
            out.append(x if isinstance(x, str) else hidden("x" * x, True))
            k += isinstance(x, int) or (isinstance(x, str) and x[:1].isalnum())
            if k > n:
                break
        return "".join(out) + " ..."

    snaps, steps = game["snapshots"], game["steps"]
    mid = pick_steps(game)[1] + 1
    parts = [("Start (no guesses)", snaps[0]),
             (f"After {mid} guesses: " + ", ".join(s["word"] for s in steps[:mid]),
              snaps[mid]),
             (f"Solved after {len(steps)} guesses", snaps[-1])]
    path.write_text("\n\n".join(f"== {h} ==\n{render(o, words)}" for h, o in parts) + "\n")


# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------

AGENT_STYLE = {"belief": ("Belief agent", BLUE),
               "frequency": ("Fixed frequency order", ORANGE),
               "random": ("Random words", AQUA)}


def load_results(split: str) -> dict[str, list[dict]]:
    res = {}
    for name in AGENT_STYLE:
        p = Path("results") / f"{name}_{split}.csv"
        if p.exists():
            res[name] = list(csv.DictReader(p.open()))
    return res


def plot_success_curve(results: dict, path: Path, max_guesses: int = 500) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    budgets = np.arange(0, max_guesses + 1)
    ends = []
    for name, rows in results.items():
        label, color = AGENT_STYLE[name]
        g = np.array([int(r["guesses"]) for r in rows if r["solved"] == "True"])
        rate = [(g <= b).sum() / len(rows) * 100 for b in budgets]
        ax.plot(budgets, rate, color=color, linewidth=2, label=label)
        ends.append((rate[-1], f"{label} ({rate[-1]:.1f}%)"))
    # Direct labels at the right edge, nudged apart so they never overlap.
    last = -np.inf
    for y, text in sorted(ends):
        y_lab = max(y, last + 6)
        ax.text(max_guesses + 8, y_lab, text, color=INK, va="center", fontsize=9)
        last = y_lab
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    for b in (100, 200):
        ax.axvline(b, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlim(0, max_guesses)
    ax.set_ylim(0, 100)
    ax.set_xlabel("guess budget")
    ax.set_ylabel("% of validation articles solved")
    n = len(next(iter(results.values())))
    ax.set_title(f"Success rate vs. guess budget ({n} validation articles)",
                 loc="left")
    fig.savefig(path)
    plt.close(fig)


def plot_guess_hist(rows: list[dict], path: Path, max_guesses: int = 500) -> None:
    g = np.array([int(r["guesses"]) for r in rows if r["solved"] == "True"])
    n_fail = sum(r["solved"] != "True" for r in rows)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    bins = np.arange(0, max_guesses + 25, 25)
    ax.hist(g, bins=bins, color=BLUE, edgecolor="white", linewidth=1.5)
    med = np.median(g)
    ax.axvline(med, color=INK, linewidth=1, linestyle="--")
    ax.text(med + 6, ax.get_ylim()[1] * 0.92, f"median {med:.0f}", fontsize=9,
            color=INK)
    ax.set_xlim(0, max_guesses)
    ax.set_xlabel("guesses to solve")
    ax.set_ylabel("articles")
    ax.grid(axis="x", visible=False)
    ax.set_title("Belief agent: guesses needed on solved articles", loc="left")
    ax.text(0, -0.25, f"{len(g)} of {len(rows)} validation articles solved; "
            f"{n_fail} unsolved after {max_guesses} guesses (not shown).",
            transform=ax.transAxes, color=INK2, fontsize=8.5)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="Neptune")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    slug = args.example.lower().replace(" ", "_")
    game = trace_game(args.example, args.split)
    plot_history(game, OUT / f"example_{slug}_history.png")
    plot_steps(game, OUT / f"example_{slug}_steps.png")
    write_text(game, OUT / f"example_{slug}_text.txt")

    results = load_results(args.split)
    if results:
        plot_success_curve(results, OUT / "success_vs_budget.png")
    if "belief" in results:
        plot_guess_hist(results["belief"], OUT / "guesses_to_solve.png")
    print(f"figures -> {OUT}/")


if __name__ == "__main__":
    main()
