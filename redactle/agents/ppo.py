"""PPO agent: a trained candidate-set policy (see redactle/rl/)."""

from __future__ import annotations

import torch

from redactle.agents.base import Agent
from redactle.env import Observation
from redactle.paths import ROOT
from redactle.rl.features import CandidateFeaturizer
from redactle.rl.policy import CandidatePolicy

DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "ppo" / "best.pt"


class PPOAgent(Agent):
    """Picks among the belief model's candidate words with a policy trained by
    PPO (scripts/train_ppo.py). Greedy (argmax) unless sample=True."""
    name = "ppo"

    def __init__(self, vocab: list[dict], checkpoint: str = DEFAULT_CHECKPOINT,
                 sample: bool = False, seed: int = 0):
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = ckpt["config"]
        if cfg.get("use_lm"):
            raise ValueError(f"{checkpoint} was trained with LM candidates, "
                             "which PPO no longer supports")
        use_hunt = cfg.get("use_hunt", False)
        self.policy = CandidatePolicy(cfg["hidden"], use_hunt)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.policy.eval()
        self.feat = CandidateFeaturizer(vocab, max_guesses=cfg["max_guesses"],
                                        use_hunt=use_hunt)
        self.sample = sample
        self.gen = torch.Generator().manual_seed(seed)

    def reset(self) -> None:
        self.feat.reset()

    def act(self, obs: Observation) -> str:
        s = self.feat.featurize(obs)
        if not s.mask.any():
            return f"zz{obs.n_guesses}"
        with torch.no_grad():
            logits, _ = self.policy(torch.from_numpy(s.cand)[None],
                                    torch.from_numpy(s.mask)[None],
                                    torch.from_numpy(s.glob)[None])
        if self.sample:
            a = int(torch.multinomial(logits[0].softmax(-1), 1, generator=self.gen))
        else:
            a = int(logits[0].argmax())
        self.last_state, self.last_logits = s, logits[0]
        return s.words[a]
