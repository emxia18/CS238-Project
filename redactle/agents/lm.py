"""Belief agent plus a masked language model reading the redacted text."""

from __future__ import annotations

import numpy as np

from redactle.agents.belief import BeliefAgent
from redactle.env import Observation
from redactle.redact import REDACTLE_COMMON_WORDS, match_key


class MaskedLMScorer:
    """DistilRoBERTa's guess at the hidden title-length words.

    Every hidden word in the first `window` words is replaced by <mask>.
    For each mask whose length matches a still-hidden title word, the LM's
    probability for every whole-word token of exactly that length is added
    up, giving the LM's expected number of matches for each word in that
    window, rescaled to the full article.

    Words are indexed like the training vocabulary, followed by `extra_words`:
    LM tokens outside the vocabulary (always-visible words excluded).
    Only single-token words can be proposed (e.g. "Neptune", not "Uranus").
    """

    def __init__(self, vocab_index: dict[str, int], window: int = 300,
                 model: str = "distilroberta-base"):
        import sys
        sys.modules.setdefault("torchvision", None)  # broken install; unused
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model)
        self.lm = AutoModelForMaskedLM.from_pretrained(model).to(self.device).eval()
        self.window = window
        self.n_vocab = len(vocab_index)
        self.extra_words: list[str] = []
        extra_index: dict[str, int] = {}
        ids, cols, lens = [], [], []
        for tok, i in self.tok.get_vocab().items():
            if not tok.startswith("Ġ") or not tok[1:].isalpha():
                continue
            word = tok[1:]
            if word.lower() in REDACTLE_COMMON_WORDS:
                continue                      # never hidden, so never a match
            key = match_key(word)
            if key in vocab_index:
                col = vocab_index[key]
            else:
                if key not in extra_index:
                    extra_index[key] = len(self.extra_words)
                    self.extra_words.append(word.lower())
                col = self.n_vocab + extra_index[key]
            ids.append(i); cols.append(col); lens.append(len(word))
        self.lm_ids = torch.tensor(ids, device=self.device)
        self.lm_cols = np.array(cols)
        self.lm_lens = np.array(lens)
        self.extra_pos = extra_index                       # key -> extra index
        self.extra_len = np.array([len(w) for w in self.extra_words])

    def counts(self, obs: Observation, length: int) -> np.ndarray | None:
        """Expected matches per word (vocab, then extra words), or None when
        no masked word in the window has a hidden title length."""
        torch = self.torch
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
        out = np.zeros(self.n_vocab + len(self.extra_words))
        np.add.at(out, self.lm_cols, per_word)
        return out * length / max(len(lengths), 1)


class LMBeliefAgent(BeliefAgent):
    """Belief agent plus MaskedLMScorer: the LM's expected counts are added to
    the belief agent's (weight `lm_weight`), and words the LM proposes that
    are outside the training vocabulary can be guessed too."""
    name = "lm"

    def __init__(self, vocab: list[dict], lm_weight: float = 3.0,
                 window: int = 300, model: str = "distilroberta-base",
                 endgame_conf: float | None = 0.5, **kwargs):
        # Defaults tuned on the first 100 val articles: 79% solved vs 69% for
        # the belief agent alone (lm_weight 0.3/1/10 and no end-game were worse).
        super().__init__(vocab, endgame_conf=endgame_conf, **kwargs)
        self.scorer = MaskedLMScorer(self.index, window, model)
        self.lm_weight = lm_weight
        self.extra_words = self.scorer.extra_words

    def reset(self) -> None:
        super().reset()
        self.extra_guessed = np.zeros(len(self.extra_words), dtype=bool)

    def lm_counts(self, obs: Observation):
        return self.scorer.counts(obs, self.length)

    def act(self, obs: Observation) -> str:
        self._observe(obs)
        for g in obs.history:
            if g.key in self.scorer.extra_pos:
                self.extra_guessed[self.scorer.extra_pos[g.key]] = True
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
        elen = self.scorer.extra_len
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
