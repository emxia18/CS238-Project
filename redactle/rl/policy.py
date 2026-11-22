"""Actor-critic network over a candidate set.

Actor: each candidate is scored by an MLP on [candidate features, global
features]; a masked softmax over the K scores is the policy. The network is
permutation-equivariant in the candidates, so it learns *what makes a good
guess* rather than slot positions.
Critic: an MLP on the global features plus mean- and max-pooled candidates.
"""

from __future__ import annotations

import torch
from torch import nn

from redactle.rl.features import n_cand_features, n_global_features


def mlp(n_in: int, hidden: int, n_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                         nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


class CandidatePolicy(nn.Module):
    def __init__(self, hidden: int = 128, use_lm: bool = False,
                 use_hunt: bool = False):
        super().__init__()
        f, g = n_cand_features(use_lm, use_hunt), n_global_features(use_hunt)
        self.actor = mlp(f + g, hidden, 1)
        self.critic = mlp(g + 2 * f, hidden, 1)

    def forward(self, cand: torch.Tensor, mask: torch.Tensor, glob: torch.Tensor):
        """cand (B,K,F), mask (B,K) bool, glob (B,G) -> logits (B,K), value (B,)"""
        B, K, _ = cand.shape
        x = torch.cat([cand, glob[:, None, :].expand(B, K, -1)], dim=-1)
        logits = self.actor(x).squeeze(-1).masked_fill(~mask, -1e9)
        m = mask[..., None].float()
        mean = (cand * m).sum(1) / m.sum(1).clamp(min=1)
        mx = cand.masked_fill(~mask[..., None], -1e9).max(1).values
        mx = torch.where(mask.any(1, keepdim=True), mx, torch.zeros_like(mx))
        value = self.critic(torch.cat([glob, mean, mx], dim=-1)).squeeze(-1)
        return logits, value
