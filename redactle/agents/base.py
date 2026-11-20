"""Agent interface and simple baselines."""

from __future__ import annotations

import random

from redactle.env import Observation


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
