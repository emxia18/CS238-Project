"""Redactle environment.

Open vocabulary: an action is any word string, exactly like the real game.
Agents decide which words to propose (e.g. from a masked language model).

    env = RedactleEnv(load_articles(split="train"))
    obs = env.reset()
    obs, reward, done, info = env.step("he")

Rules: matching uses redact.match_key (case/accents/apostrophes ignored,
related forms like he/him/his and run/ran share a key). A guess reveals every
matching word in the title and body. The puzzle is solved when every
non-common title word is revealed.

Reward: -1 per counted guess, +success_reward when solved. Episodes end on
success or after max_guesses counted guesses. Guesses that reveal nothing new
(repeat / same word family, always-visible word, not a single word) are
"invalid": they cost a guess unless count_invalid=False (used for human play).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from redact import (REDACTLE_COMMON_WORDS, hidden, match_key, normalize,
                    tokenize)

DEFAULT_DATA = "data/articles.jsonl"


def load_articles(path: str = DEFAULT_DATA, split: str | None = None,
                  max_level: int | None = None) -> list[dict]:
    """Articles from data.py's output, optionally filtered by split/level."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found - run `python data.py` first.")
    arts = []
    with p.open() as f:
        for line in f:
            a = json.loads(line)
            if split and a["split"] != split:
                continue
            if max_level and (a.get("level") or 99) > max_level:
                continue
            arts.append(a)
    return arts


# ---------------------------------------------------------------------------
# Precomputed puzzle
# ---------------------------------------------------------------------------

@dataclass
class Token:
    text: str
    is_word: bool
    key: str | None = None      # match_key for hideable words, else None
    in_title: bool = False


@dataclass
class Puzzle:
    """An article tokenized once, with an index from match key -> tokens."""
    article: dict
    tokens: list[Token]               # title tokens, then body tokens
    n_title: int                      # tokens[:n_title] are the title
    positions: dict[str, list[int]]   # key -> indices of hideable tokens
    targets: frozenset[str]           # title keys needed to win

    @classmethod
    def from_article(cls, article: dict, reveal_numbers: bool = False) -> "Puzzle":
        tokens: list[Token] = []
        positions: dict[str, list[int]] = {}
        targets = set()
        for part, in_title in ((article["title"], True), (article["text"], False)):
            for piece, is_word in tokenize(part):
                tok = Token(piece, is_word, in_title=in_title)
                norm = normalize(piece) if is_word else ""
                visible = (not is_word or norm in REDACTLE_COMMON_WORDS
                           or (reveal_numbers and norm.isdigit()))
                if not visible:
                    tok.key = match_key(piece)
                    positions.setdefault(tok.key, []).append(len(tokens))
                    if in_title:
                        targets.add(tok.key)
                tokens.append(tok)
            if in_title:
                n_title = len(tokens)
        return cls(article, tokens, n_title, positions, frozenset(targets))


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class Guess:
    word: str        # as typed
    key: str         # match key
    count: int       # number of words it revealed (title + body)
    valid: bool      # False if it revealed nothing new / wasn't allowed


@dataclass
class Observation:
    """What the player can see. Hidden words expose only their length."""
    title: list[str | int]       # revealed text, or int length if hidden
    body: list[str | int]
    history: list[Guess]
    revealed: frozenset[str]     # match keys guessed so far
    n_guesses: int               # counted guesses
    solved: bool

    def render(self, show_lengths: bool = True) -> tuple[str, str]:
        def r(seq):
            return "".join(x if isinstance(x, str) else hidden("x" * x, show_lengths)
                           for x in seq)
        return r(self.title), r(self.body)

    def masked_text(self, mask: str = "<mask>", max_words: int | None = None
                    ) -> tuple[str, list[int]]:
        """Body with each hidden word replaced by `mask` (for masked LMs).
        Returns the text and the hidden-word lengths in order."""
        out, lengths, words = [], [], 0
        for x in self.body:
            if isinstance(x, int):
                out.append(mask)
                lengths.append(x)
                words += 1
            else:
                out.append(x)
                words += x[:1].isalnum()
            if max_words and words >= max_words:
                break
        return "".join(out), lengths


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

@dataclass
class RedactleEnv:
    articles: list[dict]
    max_guesses: int = 500
    success_reward: float = 500.0
    step_penalty: float = 1.0
    count_invalid: bool = True
    reveal_numbers: bool = False
    seed: int | None = None

    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.puzzle: Puzzle | None = None

    # -- core API -----------------------------------------------------------
    def reset(self, article: dict | int | str | None = None) -> Observation:
        """Start a puzzle: a random article, an index, a title, or a dict."""
        if article is None:
            article = self.rng.choice(self.articles)
        elif isinstance(article, int):
            article = self.articles[article]
        elif isinstance(article, str):
            matches = [a for a in self.articles
                       if a["title"].lower() == article.lower()]
            if not matches:
                raise KeyError(f"no article titled {article!r}")
            article = matches[0]
        self.puzzle = self._get_puzzle(article)
        self.visible = [tok.key is None for tok in self.puzzle.tokens]
        self.revealed: set[str] = set()
        self.history: list[Guess] = []
        self.n_guesses = 0
        self.done = False
        return self.observation()

    def step(self, word: str) -> tuple[Observation, float, bool, dict]:
        if self.puzzle is None or self.done:
            raise RuntimeError("call reset() before step()")
        word = word.strip()
        norm = normalize(word)
        key = match_key(word) if norm else ""

        if not norm or " " in word:
            reason = "not_a_word"
        elif norm in REDACTLE_COMMON_WORDS or (self.reveal_numbers and norm.isdigit()):
            reason = "always_visible"
        elif key in self.revealed:
            reason = "already_guessed"
        else:
            reason = None

        count = 0
        if reason is None:
            self.revealed.add(key)
            for i in self.puzzle.positions.get(key, []):
                self.visible[i] = True
            count = len(self.puzzle.positions.get(key, []))

        valid = reason is None
        self.history.append(Guess(word, key, count, valid))
        counted = valid or self.count_invalid
        reward = -self.step_penalty if counted else 0.0
        self.n_guesses += counted

        solved = self.solved
        if solved:
            reward += self.success_reward
        truncated = not solved and self.n_guesses >= self.max_guesses
        self.done = solved or truncated
        info = {"count": count, "valid": valid, "reason": reason,
                "solved": solved, "truncated": truncated,
                "title": self.puzzle.article["title"] if self.done else None}
        return self.observation(), reward, self.done, info

    # -- helpers ------------------------------------------------------------
    @property
    def solved(self) -> bool:
        return self.puzzle is not None and self.puzzle.targets <= self.revealed

    def observation(self) -> Observation:
        p = self.puzzle
        seq = [t.text if v else len(t.text) for t, v in zip(p.tokens, self.visible)]
        return Observation(title=seq[:p.n_title], body=seq[p.n_title:],
                           history=list(self.history),
                           revealed=frozenset(self.revealed),
                           n_guesses=self.n_guesses, solved=self.solved)

    def reveal_all(self) -> Observation:
        """Show the whole article (e.g. after giving up)."""
        self.visible = [True] * len(self.puzzle.tokens)
        return self.observation()

    def _get_puzzle(self, article: dict) -> Puzzle:
        aid = article["id"]
        if aid not in self._cache:
            self._cache[aid] = Puzzle.from_article(article, self.reveal_numbers)
        return self._cache[aid]
