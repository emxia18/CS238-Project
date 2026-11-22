"""Agents for the Redactle environment.

Every agent implements:
    reset()                 -- called at the start of each puzzle
    act(obs) -> str         -- the next word to guess

Modules:
    base    Agent interface, RandomAgent, FrequencyAgent
    data    training-split vocabulary / count matrix (cached under data/)
    belief  BeliefAgent
    lm      LMBeliefAgent (imported lazily: needs torch + transformers)
    ppo     PPOAgent: trained policy over belief candidates (redactle/rl/)
"""

from __future__ import annotations

from redactle.agents.base import Agent, FrequencyAgent, RandomAgent
from redactle.agents.data import load_vocab

AGENTS = ("random", "frequency", "belief", "lm", "ppo")


def make_agent(name: str, seed: int | None = None, min_df: int = 2,
               **kwargs) -> Agent:
    vocab = load_vocab(min_df=min_df)
    if name == "random":
        return RandomAgent(vocab, seed)
    if name == "frequency":
        return FrequencyAgent(vocab)
    if name == "belief":
        from redactle.agents.belief import BeliefAgent
        return BeliefAgent(vocab, **kwargs)
    if name == "lm":
        from redactle.agents.lm import LMBeliefAgent
        return LMBeliefAgent(vocab, **kwargs)
    if name == "ppo":
        from redactle.agents.ppo import PPOAgent
        return PPOAgent(vocab, **kwargs)
    raise ValueError(f"unknown agent {name!r}; choose from {list(AGENTS)}")


__all__ = ["AGENTS", "Agent", "FrequencyAgent", "RandomAgent", "load_vocab",
           "make_agent"]
