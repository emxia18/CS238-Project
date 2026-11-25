"""Figures for the write-up. Run after scripts/evaluate.py has produced results/*.csv.

  python -m scripts.figures                       # all figures -> figures/
  python -m scripts.figures --example Telescope   # a different example game
  python -m scripts.figures --agent belief        # example game with the belief agent
  python -m scripts.figures --split val           # aggregate plots on val instead of test

Aggregate plots use the best configuration of each agent (see AGENTS below,
matching scripts/run_test.sh); the example game uses the same configuration.

Outputs:
  figures/example_<title>_history.png   guesses + match counts for one game
  figures/example_<title>_steps.png     belief and top next-word scores at
                                         a few steps of that game
  figures/example_<title>_text.txt      redacted text before / mid / end
  figures/success_vs_budget.png         % solved vs guess budget, per agent
  figures/guesses_to_solve.png          distribution of guesses (best agent)
  figures/split_success.png             % solved per agent, train vs val vs test
  figures/split_cost.png                mean capped guesses, train vs val vs test
  figures/split_success_curves.png      success vs budget, one panel per split
  figures/split_*_main.png              the three above without the baselines
  figures/level_success.png             % solved per vital-article level (val, test)
  figures/level_cost.png                mean capped guesses per level (val, test)

The train split is evaluated on its first 647 articles (same size as val /
test), e.g. python -m scripts.evaluate --split train --n 647 ...
"""

from __future__ import annotations

import argparse
import csv
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
import numpy as np  # noqa: E402

from redactle.agents import make_agent  # noqa: E402
from redactle.env import RedactleEnv, load_articles  # noqa: E402
from redactle.paths import FIGURES_DIR, RESULTS_DIR  # noqa: E402

# Palette (validated: categorical slots 1-4 of the reference palette).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#c9c8c2"
# Splits: ordinal one-hue ramp (validated --ordinal), distinct from agent hues.
SPLIT_COLOR = {"train": "#86b6ef", "val": "#2a78d6", "test": "#104281"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
OUT = FIGURES_DIR

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
})


# ---------------------------------------------------------------------------
# Example game
# ---------------------------------------------------------------------------

def trace_game(title: str, split: str = "val", agent_name: str = "belief") -> dict:
    """Play one game with a belief-based agent, recording its internals."""
    train = load_articles(split="train")
    env = RedactleEnv(load_articles(split=split))
    agent = make_agent(agent_name, **AGENTS[agent_name][3])
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
                      "candidates": cand, "belief": belief,
                      "kind": agent.last_kind})
        snapshots.append(obs)
    return {"title": title, "targets": env.puzzle.targets, "steps": steps,
            "snapshots": snapshots, "label": AGENTS[agent_name][0]}


def plot_history(game: dict, path: Path) -> None:
    steps = game["steps"]
    x = np.arange(1, len(steps) + 1)
    counts = [s["count"] for s in steps]
    colors = [ORANGE if s["title_hit"] else BLUE for s in steps]
    fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(steps) + 1.5), 3.2))
    ax.bar(x, counts, width=0.7, color=colors, edgecolor="white", linewidth=1)
    ax.set_xticks(x, [s["word"] for s in steps], rotation=60, ha="right")
    ax.set_ylabel("words revealed by guess")
    ax.set_xlim(0.4, len(steps) + 0.6)
    ax.grid(axis="x", visible=False)
    for xi, c in zip(x, counts):
        if c == 0:
            ax.text(xi, max(counts) * 0.01, "0", ha="center", va="bottom",
                    color=INK2, fontsize=9.5)
    ax.set_title(f"{game['label']} on “{game['title']}”: solved in "
                 f"{len(steps)} guesses", loc="left")
    ax.legend(handles=[Patch(color=BLUE, label="other guess"),
                       Patch(color=ORANGE, label="title word")],
              loc="upper left", frameon=False, fontsize=9.5)
    fig.savefig(path)
    plt.close(fig)


def pick_steps(game: dict, n: int = 4) -> list[int]:
    """Steps (0-based) to show: early, where the belief first concentrates,
    the step after, and the final guess."""
    steps = game["steps"]
    peak = next((i for i, s in enumerate(steps) if s["belief"][0][1] > 0.1),
                len(steps) // 2)
    last = len(steps) - 1
    chosen = {min(5, last), peak, min(peak + 1, last), last}
    # Short games: fill with the guesses just before the final one.
    for i in range(last - 1, -1, -1):
        if len(chosen) >= min(n, len(steps)):
            break
        chosen.add(i)
    return sorted(chosen)[-n:]


def plot_steps(game: dict, path: Path) -> None:
    idx = pick_steps(game)
    fig, axes = plt.subplots(2, len(idx), figsize=(3.4 * len(idx), 6.0),
                             gridspec_kw={"hspace": 0.6, "wspace": 0.8,
                                          "top": 0.86})
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
        ax.tick_params(axis="y", labelsize=9)
        for y, v in enumerate(vals):
            if v > 0.7:   # long bar: label inside, so it never hits the edge
                ax.text(v - 0.03, y, f"{v:.2f}", va="center", ha="right",
                        fontsize=8.5, color="white")
            else:
                ax.text(v + 0.03, y, f"{v:.2f}", va="center",
                        fontsize=8.5, color=INK2)
        # Row 2: top candidate next words.
        ax = axes[1, col]
        cand = s["candidates"][::-1]
        colors = [ORANGE if c[1] in game["targets"]
                  else BLUE if c[0] == s["word"] else GRAY for c in cand]
        ax.barh([c[0] for c in cand], [c[2] for c in cand], color=colors,
                height=0.65, edgecolor="white")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=9)
        hunt = s["kind"] == "hunt"
        ax.set_xlabel("title-hunt score" if hunt else "score (expected matches)",
                      fontsize=8.5)
        ax.set_title(f"{'title hunt: ' if hunt else ''}guessed “{s['word']}” "
                     f"→ {s['count']}", loc="left", fontsize=10)
    axes[0, 0].set_ylabel("belief: most similar\ntraining articles",
                          fontsize=10)
    axes[1, 0].set_ylabel("top-scoring\nnext words", fontsize=10)
    fig.suptitle(f"What the agent thinks while solving “{game['title']}” "
                 f"({game['label']})", x=0.05, y=1.0, ha="left", fontsize=13)
    fig.legend(handles=[Patch(color=BLUE, label="word guessed"),
                        Patch(color=ORANGE, label="title word"),
                        Patch(color=GRAY, label="other candidate")],
               loc="upper left", bbox_to_anchor=(0.05, 0.965), ncol=3,
               frameon=False, fontsize=9.5)
    fig.savefig(path)
    plt.close(fig)


def write_text(game: dict, path: Path, words: int = 90) -> None:
    from redactle.redact import hidden

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

# name -> (label, color, results tag, make_agent kwargs). Best configuration
# of each agent type; keep in sync with scripts/run_test.sh.
AGENTS = {
    "lm": ("Belief + DistilRoBERTa", BLUE, "w3_end0.5_hunt3",
           {"hunt_every": 3}),
    "belief": ("Belief agent", AQUA, "end0.5_hunt3",
               {"endgame_conf": 0.5, "hunt_every": 3}),
    "ppo": ("PPO policy", YELLOW, "hunt_best",
            {"checkpoint": "checkpoints/ppo_hunt/best.pt"}),
    "frequency": ("Fixed frequency order", ORANGE, None, {}),
    "random": ("Random words", GRAY, None, {}),
}
SPLIT_NAME = {"val": "validation", "test": "test", "train": "training"}


def load_results(split: str) -> dict[str, list[dict]]:
    res = {}
    for name, (_, _, tag, _) in AGENTS.items():
        stem = f"{name}_{tag}" if tag else name
        p = RESULTS_DIR / f"{stem}_{split}.csv"
        if p.exists():
            res[name] = list(csv.DictReader(p.open()))
    return res


def draw_success_curves(ax, results: dict, max_guesses: int = 500,
                        full_labels: bool = True, fontsize: float = 9) -> None:
    budgets = np.arange(0, max_guesses + 1)
    ends = []
    for name, rows in results.items():
        label, color = AGENTS[name][:2]
        g = np.array([int(r["guesses"]) for r in rows if r["solved"] == "True"])
        rate = [(g <= b).sum() / len(rows) * 100 for b in budgets]
        ax.plot(budgets, rate, color=color, linewidth=2, label=label)
        ends.append((rate[-1], f"{label} ({rate[-1]:.1f}%)" if full_labels
                     else f"{rate[-1]:.1f}%"))
    # Direct labels at the right edge, nudged apart so they never overlap.
    # Short labels: curves ending on the same value share one label.
    if not full_labels:
        ends = sorted({(round(y, 1), t) for y, t in ends})
    last = -np.inf
    for y, text in sorted(ends):
        y_lab = max(y, last + 6)
        ax.text(max_guesses + 8, y_lab, text, color=INK, va="center",
                fontsize=fontsize)
        last = y_lab
    for b in (100, 200):
        ax.axvline(b, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_xlim(0, max_guesses)
    ax.set_ylim(0, 100)
    ax.set_xlabel("guess budget")


def plot_success_curve(results: dict, path: Path, split: str,
                       max_guesses: int = 500) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    draw_success_curves(ax, results, max_guesses)
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.38), frameon=False,
              fontsize=9.5)
    ax.set_ylabel(f"% of {SPLIT_NAME[split]} articles solved")
    n = len(next(iter(results.values())))
    ax.set_title(f"Success rate vs. guess budget ({n} {SPLIT_NAME[split]} articles)",
                 loc="left")
    fig.savefig(path)
    plt.close(fig)


def plot_guess_hist(rows: list[dict], path: Path, label: str, split: str,
                    max_guesses: int = 500) -> None:
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
    ax.set_title(f"{label}: guesses needed on solved "
                 f"{SPLIT_NAME[split]} articles", loc="left")
    ax.text(0.98, 0.95, f"{len(g)} of {len(rows)} solved\n"
            f"{n_fail} unsolved (not shown)", transform=ax.transAxes,
            ha="right", va="top", color=INK2, fontsize=10)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Train vs val vs test
# ---------------------------------------------------------------------------

SPLITS = ("train", "val", "test")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a binomial proportion, in percent."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * max(centre - half, 0), 100 * min(centre + half, 1)


def capped_cost(rows: list[dict], max_guesses: int = 500) -> np.ndarray:
    return np.array([int(r["guesses"]) if r["solved"] == "True" else max_guesses
                     for r in rows])


def _grouped_bars(by_split: dict, path: Path, metric, ylabel: str, title: str,
                  ylim: float, fmt: str,
                  legend_loc: str = "upper right") -> None:
    """One group of bars per agent, one bar per split. metric(rows) ->
    (value, lo, hi). legend_loc="outside" puts the legend right of the plot."""
    agents = [a for a in AGENTS if any(a in r for r in by_split.values())]
    splits = [s for s in SPLITS if s in by_split]
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(7.6 if len(agents) > 3 else 6.4, 3.8))
    x = np.arange(len(agents))
    for j, split in enumerate(splits):
        xs, vals, errs = [], [], []
        for i, a in enumerate(agents):
            if a not in by_split[split]:
                continue
            v, lo, hi = metric(by_split[split][a])
            xs.append(x[i] + (j - (len(splits) - 1) / 2) * width)
            vals.append(v)
            errs.append((max(v - lo, 0), max(hi - v, 0)))
        n = len(next(iter(by_split[split].values())))
        ax.bar(xs, vals, width=width * 0.92, color=SPLIT_COLOR[split],
               edgecolor="white", linewidth=1,
               label=f"{SPLIT_NAME[split]} ({n})")
        ax.errorbar(xs, vals, yerr=np.array(errs).T, fmt="none", ecolor=INK2,
                    elinewidth=0.9, capsize=2)
        for xi, v, (_, up) in zip(xs, vals, errs):
            ax.text(xi, v + up + ylim * 0.012, format(v, fmt), ha="center",
                    va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [textwrap.fill(AGENTS[a][0], 14) for a in agents],
                  fontsize=10.5)
    ax.set_xlim(-0.5, len(agents) - 0.5)
    ax.set_ylim(0, ylim)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    where = ({"loc": "upper left", "bbox_to_anchor": (1.01, 1)}
             if legend_loc == "outside" else {"loc": legend_loc})
    ax.legend(title="split (articles)", frameon=False, fontsize=9.5,
              title_fontsize=9.5, alignment="left", **where)
    ax.set_title(title, loc="left")
    fig.savefig(path)
    plt.close(fig)


BASELINES = ("frequency", "random")


def main_agents(by_split: dict) -> dict:
    """by_split without the two baselines."""
    return {s: {a: r for a, r in res.items() if a not in BASELINES}
            for s, res in by_split.items()}


def plot_split_success(by_split: dict, path: Path, main: bool = False) -> None:
    def metric(rows):
        k = sum(r["solved"] == "True" for r in rows)
        return (100 * k / len(rows), *wilson(k, len(rows)))
    if main:
        by_split = main_agents(by_split)
    _grouped_bars(by_split, path, metric, "% of articles solved",
                  "Success rate within 500 guesses, by split", 110,
                  ".1f", legend_loc="outside" if main else "center right")


def plot_split_cost(by_split: dict, path: Path, max_guesses: int = 500,
                    n_boot: int = 2000, main: bool = False) -> None:
    rng = np.random.default_rng(0)
    if main:
        by_split = main_agents(by_split)

    def metric(rows):
        c = capped_cost(rows, max_guesses)
        boot = rng.choice(c, (n_boot, len(c))).mean(axis=1)
        return c.mean(), *np.percentile(boot, [2.5, 97.5])
    _grouped_bars(by_split, path, metric,
                  f"mean guesses (failures = {max_guesses})",
                  "Mean guesses per article, by split (lower is better)",
                  250 if main else 575, ".0f",
                  legend_loc="outside" if main else "upper left")


def plot_split_curves(by_split: dict, path: Path,
                      max_guesses: int = 500, main: bool = False) -> None:
    if main:
        by_split = main_agents(by_split)
    splits = [s for s in SPLITS if s in by_split]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.1 * len(splits), 3.6),
                             sharey=True, gridspec_kw={"wspace": 0.42})
    axes = np.atleast_1d(axes)
    for ax, split in zip(axes, splits):
        draw_success_curves(ax, by_split[split], max_guesses,
                            full_labels=False, fontsize=9)
        n = len(next(iter(by_split[split].values())))
        ax.set_title(f"{SPLIT_NAME[split].capitalize()} ({n} articles)",
                     loc="left", fontsize=10)
    axes[0].set_ylabel("% of articles solved")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.06, 1.02))
    fig.suptitle("Success rate vs. guess budget on each split", x=0.06,
                 y=1.09, ha="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# By vital-article level
# ---------------------------------------------------------------------------

# Levels 1-2 have only a handful of articles per split, so they are pooled.
LEVEL_GROUPS = (("Levels 1–2", ("1", "2")), ("Level 3", ("3",)),
                ("Level 4", ("4",)))


def plot_by_level(by_split: dict, path: Path, metric, ylabel: str, title: str,
                  ylim: float, fmt: str,
                  splits: tuple = ("val", "test")) -> None:
    """One panel per split; groups = level, bars = main agents (agent colors).
    metric(rows) -> (value, lo, hi)."""
    by_split = main_agents(by_split)
    splits = [s for s in splits if s in by_split]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6 * len(splits), 3.9),
                             sharey=True, gridspec_kw={"wspace": 0.12})
    axes = np.atleast_1d(axes)
    for ax, split in zip(axes, splits):
        res = by_split[split]
        agents = [a for a in AGENTS if a in res]
        width = 0.8 / len(agents)
        x = np.arange(len(LEVEL_GROUPS))
        ticks = []
        for g, (name, lvs) in enumerate(LEVEL_GROUPS):
            n = sum(r["level"] in lvs for r in next(iter(res.values())))
            ticks.append(f"{name}\n({n} articles)")
        for j, a in enumerate(agents):
            xs, vals, errs = [], [], []
            for g, (_, lvs) in enumerate(LEVEL_GROUPS):
                rows = [r for r in res[a] if r["level"] in lvs]
                if not rows:
                    continue
                v, lo, hi = metric(rows)
                xs.append(x[g] + (j - (len(agents) - 1) / 2) * width)
                vals.append(v)
                errs.append((max(v - lo, 0), max(hi - v, 0)))
            ax.bar(xs, vals, width=width * 0.92, color=AGENTS[a][1],
                   edgecolor="white", linewidth=1, label=AGENTS[a][0])
            ax.errorbar(xs, vals, yerr=np.array(errs).T, fmt="none",
                        ecolor=INK2, elinewidth=0.9, capsize=2)
            for xi, v, (_, up) in zip(xs, vals, errs):
                ax.text(xi, v + up + ylim * 0.012, format(v, fmt),
                        ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(x, ticks, fontsize=9.5)
        ax.set_xlim(-0.5, len(LEVEL_GROUPS) - 0.5)
        ax.set_ylim(0, ylim)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{SPLIT_NAME[split].capitalize()}", loc="left",
                     fontsize=10)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.06, 1.03))
    fig.suptitle(title, x=0.06, y=1.1, ha="left", fontsize=12)
    fig.savefig(path)
    plt.close(fig)


def plot_level_success(by_split: dict, path: Path) -> None:
    def metric(rows):
        k = sum(r["solved"] == "True" for r in rows)
        return (100 * k / len(rows), *wilson(k, len(rows)))
    plot_by_level(by_split, path, metric, "% of articles solved",
                  "Success rate within 500 guesses, by article level",
                  112, ".0f")


def plot_level_cost(by_split: dict, path: Path, max_guesses: int = 500,
                    n_boot: int = 2000) -> None:
    rng = np.random.default_rng(0)

    def metric(rows):
        c = capped_cost(rows, max_guesses)
        boot = rng.choice(c, (n_boot, len(c))).mean(axis=1)
        return c.mean(), *np.percentile(boot, [2.5, 97.5])
    plot_by_level(by_split, path, metric,
                  f"mean guesses (failures = {max_guesses})",
                  "Mean guesses per article, by article level (lower is better)",
                  260, ".0f")


# Ablation: each step adds one component on top of the previous one.
ABLATION = (
    ("Belief agent\n(no title hunt)", AQUA, "belief_end0.5"),
    ("+ title hunt", AQUA, "belief_end0.5_hunt3"),
    ("+ DistilRoBERTa", BLUE, "lm_w3_end0.5_hunt3"),
)


def plot_ablation(path: Path, split: str = "val", max_guesses: int = 500,
                  n_boot: int = 2000) -> None:
    """Two panels (% solved, mean guesses), one bar per ablation step, with
    the change from the previous step written between bars."""
    steps = []
    for label, color, stem in ABLATION:
        p = RESULTS_DIR / f"{stem}_{split}.csv"
        if not p.exists():
            return
        steps.append((label, color, list(csv.DictReader(p.open()))))
    rng = np.random.default_rng(0)

    def solved(rows):
        k = sum(r["solved"] == "True" for r in rows)
        return 100 * k / len(rows), *wilson(k, len(rows))

    def cost(rows):
        c = capped_cost(rows, max_guesses)
        boot = rng.choice(c, (n_boot, len(c))).mean(axis=1)
        return c.mean(), *np.percentile(boot, [2.5, 97.5])

    panels = ((solved, "% of articles solved", "Puzzles solved (higher is better)",
               100, ".1f", "{:+.1f} pts"),
              (cost, f"mean guesses (failures = {max_guesses})",
               "Mean guesses (lower is better)", 280, ".0f", "{:+.0f}"))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    x = np.arange(len(steps))
    for ax, (metric, ylabel, title, ylim, fmt, dfmt) in zip(axes, panels):
        vals = [metric(rows) for _, _, rows in steps]
        v = np.array([m[0] for m in vals])
        err = np.array([(max(m[0] - m[1], 0), max(m[2] - m[0], 0))
                        for m in vals]).T
        ax.bar(x, v, width=0.62, color=[c for _, c, _ in steps],
               edgecolor="white", linewidth=1)
        ax.errorbar(x, v, yerr=err, fmt="none", ecolor=INK2, elinewidth=0.9,
                    capsize=2)
        for xi, vi, up in zip(x, v, err[1]):
            ax.text(xi, vi + up + ylim * 0.015, format(vi, fmt), ha="center",
                    va="bottom", fontsize=10, color=INK)
        # Change from the previous step, inside the top of each later bar.
        for i in range(1, len(steps)):
            ax.text(i, v[i] - err[0][i] - ylim * 0.03, dfmt.format(v[i] - v[i - 1]),
                    ha="center", va="top", fontsize=10, color="white",
                    fontweight="bold")
        ax.set_xticks(x, [s[0] for s in steps], fontsize=10)
        ax.set_xlim(-0.5, len(steps) - 0.5)
        ax.set_ylim(0, ylim)
        ax.set_ylabel(ylabel)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=11.5)
    n = len(steps[0][2])
    fig.suptitle("Ablation",
                 x=0.01, ha="left", fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--example", default="Neptune")
    ap.add_argument("--split", default="test",
                    help="split for the aggregate plots (results/*_<split>.csv)")
    ap.add_argument("--example-split", default="val",
                    help="split the example article comes from")
    ap.add_argument("--agent", default="lm", choices=["belief", "lm"],
                    help="agent for the example game")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    slug = args.example.lower().replace(" ", "_")
    if args.agent != "belief":
        slug += f"_{args.agent}"
    game = trace_game(args.example, args.example_split, args.agent)
    plot_history(game, OUT / f"example_{slug}_history.png")
    plot_steps(game, OUT / f"example_{slug}_steps.png")
    write_text(game, OUT / f"example_{slug}_text.txt")

    results = load_results(args.split)
    if results:
        plot_success_curve(results, OUT / "success_vs_budget.png", args.split)
    best = next((a for a in AGENTS if a in ("lm", "belief") and a in results), None)
    if best:
        plot_guess_hist(results[best], OUT / "guesses_to_solve.png",
                        AGENTS[best][0], args.split)

    by_split = {s: r for s in SPLITS if (r := load_results(s))}
    if len(by_split) > 1:
        plot_split_success(by_split, OUT / "split_success.png")
        plot_split_cost(by_split, OUT / "split_cost.png")
        plot_split_curves(by_split, OUT / "split_success_curves.png")
        # Same three without the baselines.
        plot_split_success(by_split, OUT / "split_success_main.png", main=True)
        plot_split_cost(by_split, OUT / "split_cost_main.png", main=True)
        plot_split_curves(by_split, OUT / "split_success_curves_main.png",
                          main=True)
    if by_split:
        plot_level_success(by_split, OUT / "level_success.png")
        plot_level_cost(by_split, OUT / "level_cost.png")
    plot_ablation(OUT / "ablation.png")
    print(f"figures -> {OUT}/")


if __name__ == "__main__":
    main()
