"""Agents for the Redactle environment.

Every agent implements:
    reset()                 -- called at the start of each puzzle
    act(obs) -> str         -- the next word to guess

Baselines here use a vocabulary built from the *training* articles only
(see build_vocab), so nothing about val/test articles leaks in.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from env import Observation, Puzzle, load_articles

VOCAB_PATH = "data/vocab_train.json"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

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


def load_vocab(path: str = VOCAB_PATH, min_df: int = 1) -> list[dict]:
    """Training-split vocabulary, built and cached on first use."""
    p = Path(path)
    if not p.exists():
        print(f"Building vocabulary from training articles -> {p}")
        vocab = build_vocab(load_articles(split="train"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(vocab, ensure_ascii=False))
    vocab = json.loads(p.read_text())
    return [v for v in vocab if v["df"] >= min_df]


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class Agent:
    name = "agent"

    def reset(self) -> None:
        pass

    def act(self, obs: Observation) -> str:
        raise NotImplementedError


class RandomAgent(Agent):
    """Uniformly random unguessed word from the vocabulary (sanity check)."""
    name = "random"

    def __init__(self, vocab: list[dict], seed: int | None = None):
        self.vocab = vocab
        self.rng = random.Random(seed)

    def reset(self) -> None:
        self.order = self.rng.sample(range(len(self.vocab)), len(self.vocab))
        self.i = 0

    def act(self, obs: Observation) -> str:
        while self.vocab[self.order[self.i]]["key"] in obs.revealed:
            self.i += 1
        word = self.vocab[self.order[self.i]]["word"]
        self.i += 1
        return word


class FrequencyAgent(Agent):
    """Fixed order: words appearing in the most training articles first.
    Ignores all feedback."""
    name = "frequency"

    def __init__(self, vocab: list[dict]):
        self.vocab = vocab  # already sorted by document frequency

    def reset(self) -> None:
        self.i = 0

    def act(self, obs: Observation) -> str:
        while self.vocab[self.i]["key"] in obs.revealed:
            self.i += 1
        word = self.vocab[self.i]["word"]
        self.i += 1
        return word


# ---------------------------------------------------------------------------
# Belief agent
# ---------------------------------------------------------------------------

COUNTS_PATH = "data/belief_counts_train.npz"


def load_counts(vocab: list[dict], path: str = COUNTS_PATH):
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


TRAIN_TITLES_PATH = "data/train_title_keys.json"


def load_train_title_keys(path: str = TRAIN_TITLES_PATH) -> list[list[str]]:
    """Title match keys of each training article (row order of load_counts)."""
    p = Path(path)
    if not p.exists():
        keys = [Puzzle.from_article(a).targets for a in load_articles(split="train")]
        p.write_text(json.dumps([sorted(k) for k in keys]))
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Belief + masked language model
# ---------------------------------------------------------------------------

class LMBeliefAgent(BeliefAgent):
    """Belief agent plus DistilRoBERTa reading the redacted text.

    Every hidden word in the first `window` words is replaced by <mask>.
    For each mask whose length matches a still-hidden title word, the LM's
    probability for every whole-word token of exactly that length is added
    up, giving the LM's expected number of matches for each word in that
    window. This is rescaled to the full article and added to the belief
    agent's expected counts (weight `lm_weight`), and words the LM proposes
    that are outside the training vocabulary can be guessed too.
    Only single-token words can be proposed (e.g. "Neptune", not "Uranus").
    """
    name = "lm"

    def __init__(self, vocab: list[dict], lm_weight: float = 1.0,
                 window: int = 300, model: str = "distilroberta-base",
                 **kwargs):
        super().__init__(vocab, **kwargs)
        import sys
        sys.modules.setdefault("torchvision", None)  # broken install; unused
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForMaskedLM.from_pretrained(model).to(self.device).eval()
        self.lm_weight, self.window = lm_weight, window
        # Whole-word tokens: id -> (vocab index or -1, surface form, length).
        from redact import match_key
        self.extra_words: list[str] = []      # LM words outside the vocab
        extra_index: dict[str, int] = {}
        ids, cols, lens = [], [], []
        for tok, i in self.tok.get_vocab().items():
            if not tok.startswith("Ġ") or not tok[1:].isalpha():
                continue
            word = tok[1:]
            key = match_key(word)
            if key in self.index:
                col = self.index[key]
            else:
                if key not in extra_index:
                    extra_index[key] = len(self.extra_words)
                    self.extra_words.append(word.lower())
                col = len(self.vocab) + extra_index[key]
            ids.append(i); cols.append(col); lens.append(len(word))
        np = self.np
        self.lm_ids = torch.tensor(ids, device=self.device)
        self.lm_cols = np.array(cols)
        self.lm_lens = np.array(lens)
        self.extra_keys = list(extra_index)

    def reset(self) -> None:
        super().reset()
        self.extra_guessed = self.np.zeros(len(self.extra_words), dtype=bool)

    def lm_counts(self, obs: Observation):
        """LM expected matches per word (vocab, then extra words), or None."""
        np, torch = self.np, self.torch
        title_lens = {x for x in obs.title if isinstance(x, int)}
        text, lengths = obs.masked_text(mask=self.tok.mask_token,
                                        max_words=self.window)
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
        ids = enc["input_ids"][0]
        pos = (ids == self.tok.mask_token_id).nonzero().ravel()
        lengths = np.array(lengths[:len(pos)])
        keep = np.isin(lengths, list(title_lens))
        if not keep.any():
            return None
        with torch.no_grad():
            logits = self.lm(**{k: v.to(self.device) for k, v in enc.items()}).logits[0]
        probs = logits[pos[torch.from_numpy(keep)].to(self.device)].softmax(-1)
        probs = probs[:, self.lm_ids].cpu().numpy()          # masks x words
        match = self.lm_lens[None, :] == lengths[keep][:, None]
        per_word = (probs * match).sum(axis=0)
        out = np.zeros(len(self.vocab) + len(self.extra_words))
        np.add.at(out, self.lm_cols, per_word)
        return out * self.length / max(len(lengths), 1)

    def act(self, obs: Observation) -> str:
        np = self.np
        self._observe(obs)
        for g in obs.history:
            if g.key in self._extra_pos:
                self.extra_guessed[self._extra_pos[g.key]] = True
        score, top, w = self.expected_counts()
        lm = self.lm_counts(obs)
        extra = np.full(len(self.extra_words), -np.inf)
        if lm is not None:
            score = score + self.lm_weight * lm[:len(self.vocab)]
            extra = self.lm_weight * lm[len(self.vocab):]
        score = self.shape_scores(score, obs, top, w)
        # Extra (out-of-vocab) words: same length rules and title bonus.
        hidden = {x for x in obs.title + obs.body if isinstance(x, int)}
        title_hidden = {x for x in obs.title if isinstance(x, int)}
        elen = self._extra_len
        extra = np.where(np.isin(elen, list(hidden)), extra, -np.inf)
        extra = np.where(np.isin(elen, list(title_hidden)),
                         extra * (1 + self.title_bonus), extra)
        extra[self.extra_guessed] = -np.inf
        self.last_score, self.last_top, self.last_w = score.copy(), top, w
        j = int(np.argmax(extra)) if len(extra) else -1
        if j >= 0 and extra[j] > score.max():
            self.extra_guessed[j] = True
            return self.extra_words[j]
        return self._pick(score, obs)

    @property
    def _extra_pos(self):
        if not hasattr(self, "_extra_pos_cache"):
            self._extra_pos_cache = {k: i for i, k in enumerate(self.extra_keys)}
        return self._extra_pos_cache

    @property
    def _extra_len(self):
        if not hasattr(self, "_extra_len_cache"):
            self._extra_len_cache = self.np.array([len(w) for w in self.extra_words])
        return self._extra_len_cache


AGENTS = {"random": RandomAgent, "frequency": FrequencyAgent,
          "belief": BeliefAgent, "lm": LMBeliefAgent}


def make_agent(name: str, seed: int | None = None, min_df: int = 2,
               **kwargs) -> Agent:
    vocab = load_vocab(min_df=min_df)
    if name == "random":
        return RandomAgent(vocab, seed)
    if name == "frequency":
        return FrequencyAgent(vocab)
    if name == "belief":
        return BeliefAgent(vocab, **kwargs)
    if name == "lm":
        return LMBeliefAgent(vocab, **kwargs)
    raise ValueError(f"unknown agent {name!r}; choose from {list(AGENTS)}")
