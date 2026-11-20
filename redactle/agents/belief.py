"""Belief agent: a posterior over training articles as possible worlds."""

from __future__ import annotations

from redactle.agents.base import Agent
from redactle.agents.data import load_counts, load_train_title_keys
from redactle.env import Observation


class BeliefAgent(Agent):
    """Belief over 'which training article is this most like'.

    The ~5k training articles act as possible worlds. After each guess, each
    world is reweighted by the Poisson likelihood of the observed match count
    given that world's (smoothed) word rate, scaled to this puzzle's length.
    The next guess is the unguessed word with the highest expected match
    count under the belief, restricted to lengths that are still hidden, with
    a bonus for lengths that still occur in the hidden title.

    mix:          share of the word rate taken from overall corpus frequency
                  (default 0: on 100 val articles, 0.1-0.5 all did worse),
                  so an uncertain belief falls back to broadly useful words
                  instead of one stray article's rare words.
    min_guess_df: only guess words seen in at least this many training
                  articles; words mixing letters and digits (e.g. hex colour
                  codes like "b0e0e6") are never guessed.
    """
    name = "belief"

    def __init__(self, vocab: list[dict], alpha: float = 20.0,
                 temperature: float = 0.5, top_k: int = 300,
                 title_bonus: float = 2.0, mix: float = 0.0,
                 min_guess_df: int = 2, endgame_conf: float | None = None,
                 endgame_bonus: float = 10.0, own_title_penalty: float = 1.0):
        import numpy as np

        self.np = np
        self.vocab = vocab
        self.index = {v["key"]: i for i, v in enumerate(vocab)}
        self.counts = load_counts(vocab)               # CSR, articles x words
        self.counts_csc = self.counts.tocsc()
        self.art_len = np.asarray(self.counts.sum(axis=1)).ravel()
        total = np.asarray(self.counts.sum(axis=0)).ravel()
        self.background = (total + 1) / (total.sum() + len(vocab))
        self.word_len = np.array([len(v["word"]) for v in vocab])
        self.alpha, self.temperature = alpha, temperature
        self.top_k, self.title_bonus, self.mix = top_k, title_bonus, mix
        self.endgame_conf, self.endgame_bonus = endgame_conf, endgame_bonus
        self.own_title_penalty = own_title_penalty
        self.allowed = np.array([
            v["df"] >= min_guess_df and (v["key"].isalpha() or v["key"].isdigit())
            for v in vocab])
        self.title_idx = None
        if own_title_penalty != 1.0:  # vocab indices of each training title
            self.title_idx = [[self.index[k] for k in keys if k in self.index]
                              for keys in load_train_title_keys()]

    def reset(self) -> None:
        np = self.np
        self.logw = np.zeros(self.counts.shape[0])
        self.guessed = np.zeros(len(self.vocab), dtype=bool)
        self.n_seen = 0
        self.length = None

    def _update(self, k: int, count: int) -> None:
        """Reweight worlds by the Poisson likelihood of `count` for word k."""
        np = self.np
        col = self.counts_csc[:, k].toarray().ravel()
        rate = (col + self.alpha * self.background[k]) / (self.art_len + self.alpha)
        lam = rate * self.length
        self.logw += self.temperature * (count * np.log(lam) - lam)
        self.logw -= self.logw.max()

    def _observe(self, obs: Observation) -> None:
        """Fold guesses made since the last call into the belief."""
        if self.length is None:  # total hideable words, visible to the player
            self.length = sum(isinstance(x, int) for x in obs.title + obs.body)
        for g in obs.history[self.n_seen:]:
            k = self.index.get(g.key)
            if k is not None:
                self.guessed[k] = True
                if g.valid:
                    self._update(k, g.count)
        self.n_seen = len(obs.history)

    def expected_counts(self):
        """Expected matches of every vocab word under the current belief.
        Returns (counts, top-k world indices, their normalised weights)."""
        np = self.np
        top = np.argpartition(self.logw, -self.top_k)[-self.top_k:]
        w = np.exp(self.logw[top])
        w /= w.sum()
        coef = w / (self.art_len[top] + self.alpha)
        rate = coef @ self.counts[top] + self.alpha * self.background * coef.sum()
        rate = (1 - self.mix) * np.asarray(rate).ravel() + self.mix * self.background
        return rate * self.length, top, w

    def shape_scores(self, score, obs: Observation, top, w):
        """Length constraints, title bonus and title-focused end-game."""
        np = self.np
        hidden = {x for x in obs.title + obs.body if isinstance(x, int)}
        title_hidden = {x for x in obs.title if isinstance(x, int)}
        bonus = self.title_bonus
        if self.endgame_conf is not None and w.max() >= self.endgame_conf:
            bonus = self.endgame_bonus      # topic found: go for the title
        score[~np.isin(self.word_len, list(hidden))] = -1
        score[np.isin(self.word_len, list(title_hidden))] *= 1 + bonus
        if self.title_idx is not None:
            # The puzzle is never a training article, so the titles of the
            # worlds we believe in are poor title guesses (e.g. "uranus"
            # while solving Neptune).
            for i in np.flatnonzero(w > 0.05):
                idx = self.title_idx[top[i]]
                score[idx] *= self.own_title_penalty
        score[self.guessed | ~self.allowed] = -np.inf
        return score

    def _pick(self, score, obs: Observation) -> str:
        np = self.np
        for k in np.argsort(-score)[:50]:
            if self.vocab[k]["key"] not in obs.revealed:
                self.guessed[k] = True
                return self.vocab[k]["word"]
            self.guessed[k] = True
        return self.vocab[int(np.argmax(score))]["word"]

    def act(self, obs: Observation) -> str:
        self._observe(obs)
        score, top, w = self.expected_counts()
        score = self.shape_scores(score, obs, top, w)
        # Kept for inspection / figures: candidate scores and belief this step.
        self.last_score, self.last_top, self.last_w = score.copy(), top, w
        return self._pick(score, obs)
