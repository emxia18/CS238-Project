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
    hunt_every:   title hunt. Once the belief is confident (endgame_conf),
                  every hunt_every-th guess goes to the title-length word
                  with the best hunt_scores() instead. The belief soon puts
                  all its weight on one training article, which usually lacks
                  the puzzle's own (rare, specific) title word; hunting scores
                  words by how many of the hunt_worlds nearest articles contain
                  them, times idf. Needs endgame_conf; None disables.
    """
    name = "belief"

    def __init__(self, vocab: list[dict], alpha: float = 20.0,
                 temperature: float = 0.5, top_k: int = 300,
                 title_bonus: float = 2.0, mix: float = 0.0,
                 min_guess_df: int = 2, endgame_conf: float | None = None,
                 endgame_bonus: float = 10.0, own_title_penalty: float = 1.0,
                 hunt_every: int | None = None, hunt_worlds: int = 50):
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
        if hunt_every is not None and endgame_conf is None:
            raise ValueError("hunt_every needs endgame_conf")
        self.hunt_every, self.hunt_worlds = hunt_every, hunt_worlds
        df = np.array([v["df"] for v in vocab], dtype=float)
        self.idf = np.log(self.counts.shape[0] / np.maximum(df, 1))
        self.presence = (self.counts > 0).astype(np.float32).tocsr()
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
        self.n_acts = 0
        self.last_hunt = -10**9      # act index of the last title-hunt guess

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

    def hunt_scores(self) -> "np.ndarray":
        """Title-hunt score of every vocab word: share of the hunt_worlds
        nearest training articles (rank-weighted 1/r) containing the word,
        times its idf. Unmasked: callers apply length/guessed filters."""
        np = self.np
        nb = np.argpartition(self.logw, -self.hunt_worlds)[-self.hunt_worlds:]
        nb = nb[np.argsort(-self.logw[nb])]
        nb = nb[np.isfinite(self.logw[nb])]
        wr = 1.0 / np.arange(1, len(nb) + 1)
        frac = np.asarray(wr @ self.presence[nb]).ravel() / wr.sum()
        return frac * self.idf

    def hunt_guess(self, obs: Observation, w) -> str | None:
        """The title-hunt guess if this turn is a hunt turn, else None.
        Call once per act(), after _observe()."""
        np = self.np
        self.n_acts += 1
        title_hidden = {x for x in obs.title if isinstance(x, int)}
        if (self.hunt_every is None or not title_hidden
                or w.max() < self.endgame_conf
                or self.n_acts - self.last_hunt < self.hunt_every):
            return None
        s = self.hunt_scores()
        s[~np.isin(self.word_len, list(title_hidden))] = -1
        s[self.guessed | ~self.allowed] = -np.inf
        if not np.isfinite(s.max()):
            return None
        self.last_hunt = self.n_acts
        self.hunt_score = s.copy()
        return self._pick(s, obs)

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
        hunt = self.hunt_guess(obs, w)
        if hunt is not None:
            self.record(self.hunt_score, top, w, "hunt")
            return hunt
        score = self.shape_scores(score, obs, top, w)
        self.record(score.copy(), top, w, "score")
        return self._pick(score, obs)

    def record(self, score, top, w, kind: str) -> None:
        """Kept for inspection / figures: the candidate scores this step
        ("score": expected matches, "hunt": title-hunt score) and belief."""
        self.last_score, self.last_top, self.last_w = score, top, w
        self.last_kind = kind
