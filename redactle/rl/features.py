"""Candidate-set state representation for RL agents.

The open vocabulary (~130k words) is far too large an action space to learn
over directly. Instead, each turn the belief model (BeliefAgent) proposes a
short list of K candidate words and describes each with a feature vector; a
policy then only has to choose among them. Candidates come from two pools:
the words with the highest expected match count under the belief and the
best belief words whose length matches a still-hidden title word. With
use_hunt=True a third pool holds the best title-hunt words
(BeliefAgent.hunt_scores: rare words common among the nearest training
articles), the hunt score becomes a feature, and a global feature says
whether the last guesses were hunts.

The masked LM is deliberately not used here: PPO and LMBeliefAgent are
separate models.

    feat = CandidateFeaturizer(vocab, use_hunt=True)
    feat.reset(exclude_row=None)     # row of a training article to leave out
    state = feat.featurize(obs)      # State(words, cand, mask, glob)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from redactle.agents.belief import BeliefAgent
from redactle.agents.data import load_train_title_keys
from redactle.env import Observation

K = 32                      # candidates per turn
N_CAND_FEATURES = 12
N_HUNT_FEATURES = 2         # added with use_hunt=True
N_GLOBAL_FEATURES = 10      # + 1 with use_hunt=True
N_HUNT_POOL = 6             # title-hunt candidates per turn
HUNT_EVERY = 3              # teacher hunts when neither of the last 2 guesses did


def n_cand_features(use_hunt: bool = False) -> int:
    return N_CAND_FEATURES + (N_HUNT_FEATURES if use_hunt else 0)


def n_global_features(use_hunt: bool = False) -> int:
    return N_GLOBAL_FEATURES + (1 if use_hunt else 0)


def top_desc(x: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n largest entries, largest first (no full sort)."""
    n = min(n, len(x))
    part = np.argpartition(-x, n - 1)[:n]
    return part[np.argsort(-x[part])]


@dataclass
class State:
    words: list[str]          # candidate surface forms ("" for padding)
    vocab_idx: np.ndarray     # (K,) vocab index, -1 pad
    cand: np.ndarray          # (K, N_CAND_FEATURES) float32
    mask: np.ndarray          # (K,) bool, True = real candidate
    glob: np.ndarray          # (N_GLOBAL_FEATURES,) float32


class CandidateFeaturizer(BeliefAgent):
    """Runs the belief update and turns the posterior into a candidate set.
    Inherits the posterior over training articles from BeliefAgent but never
    picks a word itself."""

    def __init__(self, vocab: list[dict], k: int = K, max_guesses: int = 500,
                 use_hunt: bool = False, **belief_kwargs):
        super().__init__(vocab, **belief_kwargs)
        self.k, self.max_guesses, self.use_hunt = k, max_guesses, use_hunt
        self.n_feat = n_cand_features(use_hunt)
        self.log_df = np.log1p(np.array([v["df"] for v in vocab], dtype=np.float32))
        self.log_df /= self.log_df.max()
        self.log_bg = np.log(self.background)
        self.max_len = int(self.word_len.max())
        # Title keys of each training article, as vocab-index sets.
        self.world_titles = [frozenset(self.index[k] for k in keys if k in self.index)
                             for keys in load_train_title_keys()]

    def reset(self, exclude_row: int | None = None) -> None:
        """exclude_row: a training article to remove from the possible worlds
        (leave-one-out, needed when the puzzle itself is a training article)."""
        super().reset()
        if exclude_row is not None:
            self.logw[exclude_row] = -np.inf
        self.hits: list[int] = []
        self.hunt_top: set[str] = set()   # keys of last turn's top hunt words
        self.was_hunt: list[bool] = []    # per guess: was it a top hunt word

    def featurize(self, obs: Observation) -> State:
        new = obs.history[self.n_seen:]
        self._observe(obs)
        self.hits.extend(g.count for g in new)
        self.was_hunt.extend(g.key in self.hunt_top for g in new)

        expected, top, w = self.expected_counts()
        title_lens = [x for x in obs.title if isinstance(x, int)]
        body_lens = [x for x in obs.body if isinstance(x, int)]
        # Hidden words per length, for lengths 0..max_len (a hidden word can be
        # longer than every vocabulary word; no candidate can fill it anyway).
        nl = self.max_len + 1
        n_title_of = np.bincount(title_lens, minlength=nl)[:nl]
        n_body_of = np.bincount(body_lens, minlength=nl)[:nl]

        # Candidate pool: valid words only.
        ok = ((n_title_of + n_body_of)[self.word_len] > 0) & self.allowed & ~self.guessed
        score = np.where(ok, expected, -np.inf)
        tl_ok = ok & (n_title_of[self.word_len] > 0)
        tl_score = np.where(tl_ok, expected, -np.inf)
        n_hunt_pool = N_HUNT_POOL if self.use_hunt else 0
        n_rest = self.k - n_hunt_pool
        n_belief = n_rest // 2 + n_rest % 2
        pool, source = [], []

        def add(order, valid, limit, src):
            for i in order:
                if len(pool) >= limit or not valid[i]:
                    break
                if int(i) not in seen:
                    pool.append(int(i)); source.append(src); seen.add(int(i))

        seen: set[int] = set()
        add(top_desc(score, n_belief), ok, n_belief, 0)
        add(top_desc(tl_score, self.k), tl_ok, n_belief + n_rest // 2, 1)
        hunt = None
        if self.use_hunt:
            hunt = self.hunt_scores()
            h_score = np.where(tl_ok, hunt, -np.inf)
            order = top_desc(h_score, self.k)
            add(order, tl_ok, len(pool) + n_hunt_pool, 2)
            best = int(order[0]) if len(order) and tl_ok[order[0]] else None
            self.hunt_top = {self.vocab[best]["key"]} if best is not None else set()
        else:
            self.hunt_top = set()
        idx = np.array(pool, dtype=int)
        source = np.array(source, dtype=int)

        # Per-candidate features.
        sub = self.counts[top][:, idx].toarray() if len(idx) else np.zeros((len(top), 0))
        presence = w @ (sub > 0)                        # P(word occurs at all)
        title_w = np.zeros(len(idx))
        for j, wj in zip(top, w):
            if wj > 0.01:
                t = self.world_titles[j]
                title_w += wj * np.array([i in t for i in idx])
        wlen = self.word_len[idx]
        n_title_len = n_title_of[wlen]
        n_body_len = n_body_of[wlen]
        exp = expected[idx]
        log_bg = self.log_bg[idx]
        cand = np.zeros((self.k, self.n_feat), dtype=np.float32)
        n = len(idx)
        cols = [
            np.log1p(exp),
            presence,
            n_title_len > 0,
            np.minimum(n_title_len, 4) / 4,
            np.log1p(n_body_len) / 6,
            wlen / 10,
            self.log_df[idx],
            title_w,
            np.arange(n) / self.k,
            source == 1,                                # from the title pool
            (log_bg + 15) / 10,
            np.clip(np.log(exp / self.length + 1e-12) - log_bg, -5, 10) / 5,
        ]
        if self.use_hunt:
            cols += [hunt[idx] / 5,                     # title-hunt score
                     source == 2]                       # from the hunt pool
        if n:
            cand[:n] = np.stack(cols, axis=1)
        mask = np.zeros(self.k, dtype=bool)
        mask[:n] = True
        vocab_idx = np.full(self.k, -1)
        vocab_idx[:n] = idx

        # Global features.
        n_tokens = len(obs.title) + len(obs.body)  # rough scale only
        hidden_now = len(title_lens) + len(body_lens)
        title_words = sum(isinstance(x, str) and x[:1].isalnum() or isinstance(x, int)
                          for x in obs.title)
        p = w[w > 0]
        recent = self.hits[-10:]
        glob = np.array([
            -(p * np.log(p)).sum() / np.log(len(w)),     # belief entropy
            w.max(),
            obs.n_guesses / self.max_guesses,
            1 - hidden_now / max(self.length, 1),        # share of words revealed
            len(title_lens) / max(title_words, 1),       # share of title hidden
            len(title_lens) / 4,
            np.mean([h > 0 for h in recent]) if recent else 0.0,
            np.log1p(recent[-1]) / 5 if recent else 0.0,
            np.log(self.length) / 10,
            np.log(n_tokens) / 10,
        ], dtype=np.float32)
        if self.use_hunt:   # hunts among the last HUNT_EVERY - 1 guesses
            last = self.was_hunt[-(HUNT_EVERY - 1):]
            glob = np.append(glob, np.float32(sum(last) / (HUNT_EVERY - 1)))

        words = [self.vocab[i]["word"] for i in idx] + [""] * (self.k - n)
        return State(words, vocab_idx, cand, mask, glob)
