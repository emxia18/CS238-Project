"""Training-split statistics the agents are built from.

Everything here is computed from the *training* articles only and cached
under data/, so nothing about val/test articles leaks into the agents.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from redactle.env import Puzzle, load_articles
from redactle.paths import COUNTS, TRAIN_TITLES, VOCAB


def build_vocab(articles: list[dict]) -> list[dict]:
    """One entry per match key seen in the articles (always-visible words
    excluded): its most common lowercase surface form, document frequency
    and total count. Sorted by document frequency, then total count."""
    df, total, forms = Counter(), Counter(), {}
    for i, art in enumerate(articles):
        puzzle = Puzzle.from_article(art)
        for key, pos in puzzle.positions.items():
            df[key] += 1
            total[key] += len(pos)
            forms.setdefault(key, Counter()).update(
                puzzle.tokens[j].text.lower() for j in pos)
        if (i + 1) % 1000 == 0:
            print(f"  vocab: {i + 1}/{len(articles)} articles", flush=True)
    vocab = [{"key": k, "word": forms[k].most_common(1)[0][0],
              "df": df[k], "count": total[k]} for k in df]
    vocab.sort(key=lambda v: (-v["df"], -v["count"], v["key"]))
    return vocab


def load_vocab(path: str | Path = VOCAB, min_df: int = 1) -> list[dict]:
    """Training-split vocabulary, built and cached on first use."""
    p = Path(path)
    if not p.exists():
        print(f"Building vocabulary from training articles -> {p}")
        vocab = build_vocab(load_articles(split="train"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(vocab, ensure_ascii=False))
    vocab = json.loads(p.read_text())
    return [v for v in vocab if v["df"] >= min_df]


def load_counts(vocab: list[dict], path: str | Path = COUNTS):
    """Sparse (training article x vocab word) match-count matrix, cached."""
    import numpy as np
    import scipy.sparse as sp

    p = Path(path)
    if p.exists():
        m = sp.load_npz(p).tocsr()
        if m.shape[1] == len(vocab):
            return m
    print(f"Building article x word count matrix -> {p}")
    index = {v["key"]: i for i, v in enumerate(vocab)}
    rows, cols, vals = [], [], []
    arts = load_articles(split="train")
    for r, art in enumerate(arts):
        for key, pos in Puzzle.from_article(art).positions.items():
            if key in index:
                rows.append(r)
                cols.append(index[key])
                vals.append(len(pos))
        if (r + 1) % 1000 == 0:
            print(f"  counts: {r + 1}/{len(arts)} articles", flush=True)
    m = sp.csr_matrix((np.array(vals, dtype=np.float32), (rows, cols)),
                      shape=(len(arts), len(vocab)))
    p.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(p, m)
    return m


def load_train_title_keys(path: str | Path = TRAIN_TITLES) -> list[list[str]]:
    """Title match keys of each training article (row order of load_counts)."""
    p = Path(path)
    if not p.exists():
        keys = [Puzzle.from_article(a).targets for a in load_articles(split="train")]
        p.write_text(json.dumps([sorted(k) for k in keys]))
    return json.loads(p.read_text())
