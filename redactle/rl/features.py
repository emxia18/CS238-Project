"""Candidate-set state representation for RL agents.

The open vocabulary (~130k words) is far too large an action space to learn
over directly. Instead, each turn the belief model (BeliefAgent) proposes a
short list of K candidate words and describes each with a feature vector; a
policy then only has to choose among them. Candidates come from up to three
pools: the words with the highest expected match count under the belief, the
best belief words whose length matches a still-hidden title word, and (with
use_lm=True) the masked LM's best guesses for the hidden title-length words,
which may lie outside the training vocabulary.

    feat = CandidateFeaturizer(vocab, use_lm=False)
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
N_CAND_FEATURES = 12        # without the LM
N_LM_FEATURES = 3           # added with use_lm=True
N_GLOBAL_FEATURES = 10


def n_cand_features(use_lm: bool) -> int:
    return N_CAND_FEATURES + (N_LM_FEATURES if use_lm else 0)


def top_desc(x: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n largest entries, largest first (no full sort)."""
    n = min(n, len(x))
    part = np.argpartition(-x, n - 1)[:n]
    return part[np.argsort(-x[part])]


@dataclass
class State:
    words: list[str]          # candidate surface forms ("" for padding)
    vocab_idx: np.ndarray     # (K,) vocab index (>= len(vocab): LM extra word), -1 pad
    cand: np.ndarray          # (K, N_CAND_FEATURES) float32
    mask: np.ndarray          # (K,) bool, True = real candidate
    glob: np.ndarray          # (N_GLOBAL_FEATURES,) float32


class CandidateFeaturizer(BeliefAgent):
    """Runs the belief update and turns the posterior into a candidate set.
    Inherits the posterior over training articles from BeliefAgent but never
    picks a word itself."""

    def __init__(self, vocab: list[dict], k: int = K, max_guesses: int = 500,
                 use_lm: bool = False, **belief_kwargs):
        super().__init__(vocab, **belief_kwargs)
        self.k, self.max_guesses, self.use_lm = k, max_guesses, use_lm
        self.n_feat = n_cand_features(use_lm)
        self.scorer = None
        if use_lm:
            from redactle.agents.lm import MaskedLMScorer
            self.scorer = MaskedLMScorer(self.index)
            self.n_extra = len(self.scorer.extra_words)
            self.all_words = [v["word"] for v in vocab] + self.scorer.extra_words
            self.all_len = np.concatenate([self.word_len, self.scorer.extra_len])
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
        if self.use_lm:
            self.extra_guessed = np.zeros(self.n_extra, dtype=bool)
            self._lm_cache = (None, None)   # (hidden word count, lm counts)

    def _lm(self, obs: Observation, n_hidden: int) -> np.ndarray:
        """LM expected counts over vocab + extra words. The masked text only
        changes when a guess reveals something, so reuse the last result
        while the number of hidden words is unchanged."""
        if self._lm_cache[0] != n_hidden:
            lm = self.scorer.counts(obs, self.length)
            if lm is None:
                lm = np.zeros(len(self.vocab) + self.n_extra)
            self._lm_cache = (n_hidden, lm)
        return self._lm_cache[1]

    def featurize(self, obs: Observation) -> State:
        new = obs.history[self.n_seen:]
        self._observe(obs)
        self.hits.extend(g.count for g in new)
        if self.use_lm:
            for g in new:
                j = self.scorer.extra_pos.get(g.key)
                if j is not None:
                    self.extra_guessed[j] = True

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
        n_lm_pool = self.k * 5 // 16 if self.use_lm else 0     # 10 of 32
        n_belief = (self.k - n_lm_pool) // 2 + (self.k - n_lm_pool) % 2
        pool, source = [], []

        def add(order, valid, limit, src):
            for i in order:
                if len(pool) >= limit or not valid[i]:
                    break
                if int(i) not in seen:
                    pool.append(int(i)); source.append(src); seen.add(int(i))

        seen: set[int] = set()
        add(top_desc(score, n_belief), ok, n_belief, 0)
        add(top_desc(tl_score, self.k), tl_ok, self.k - n_lm_pool, 1)
        lm_all = None
        if self.use_lm:
            n_hidden = len(title_lens) + len(body_lens)
            lm_all = self._lm(obs, n_hidden)
            guessed_all = np.concatenate([self.guessed, self.extra_guessed])
            allowed_all = np.concatenate([self.allowed, np.ones(self.n_extra, bool)])
            lens = np.minimum(self.all_len, self.max_len)
            fits = (n_title_of[lens] > 0) & (self.all_len <= self.max_len)
            lm_ok = fits & allowed_all & ~guessed_all & (lm_all > 0)
            lm_score = np.where(lm_ok, lm_all, -np.inf)
            add(top_desc(lm_score, self.k), lm_ok, self.k, 2)
        all_idx = np.array(pool, dtype=int)
        source = np.array(source, dtype=int)
        in_vocab = all_idx < len(self.vocab)
        idx = np.where(in_vocab, all_idx, 0)     # vocab index (0 = placeholder)

        # Per-candidate features (belief statistics are 0 for extra words).
        sub = self.counts[top][:, idx].toarray() if len(idx) else np.zeros((len(top), 0))
        sub[:, ~in_vocab] = 0
        presence = w @ (sub > 0)                        # P(word occurs at all)
        title_w = np.zeros(len(idx))
        for j, wj in zip(top, w):
            if wj > 0.01:
                t = self.world_titles[j]
                title_w += wj * np.array([i in t for i in idx])
        title_w[~in_vocab] = 0
        wlen = (np.where(in_vocab, self.word_len[idx], self.all_len[all_idx])
                if self.use_lm else self.word_len[idx])
        wl = np.minimum(wlen, self.max_len)
        n_title_len = n_title_of[wl]
        n_body_len = n_body_of[wl]
        exp = np.where(in_vocab, expected[idx], 0.0)
        log_bg = np.where(in_vocab, self.log_bg[idx], self.log_bg.min())
        cand = np.zeros((self.k, self.n_feat), dtype=np.float32)
        n = len(idx)
        cols = [
            np.log1p(exp),
            presence,
            n_title_len > 0,
            np.minimum(n_title_len, 4) / 4,
            np.log1p(n_body_len) / 6,
            wlen / 10,
            np.where(in_vocab, self.log_df[idx], 0.0),
            title_w,
            np.arange(n) / self.k,
            source == 1,                                # from the title pool
            (log_bg + 15) / 10,
            np.clip(np.log(exp / self.length + 1e-12) - log_bg, -5, 10) / 5,
        ]
        if self.use_lm:
            cols += [np.log1p(lm_all[all_idx]),         # LM expected matches
                     ~in_vocab,                         # outside train vocab
                     source == 2]                       # from the LM pool
        if n:
            cand[:n] = np.stack(cols, axis=1)
        mask = np.zeros(self.k, dtype=bool)
        mask[:n] = True
        vocab_idx = np.full(self.k, -1)
        vocab_idx[:n] = all_idx

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

        names = self.all_words if self.use_lm else None
        words = ([names[i] for i in all_idx] if names else
                 [self.vocab[i]["word"] for i in all_idx]) + [""] * (self.k - n)
        return State(words, vocab_idx, cand, mask, glob)
