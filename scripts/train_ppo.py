"""Train the PPO candidate-set agent.

Usage (from the repo root):
  python -m scripts.train_ppo                          # defaults -> checkpoints/ppo/
  python -m scripts.train_ppo --use-lm --workers 3 --out checkpoints/ppo_lm
  python -m scripts.train_ppo --iterations 5 --episodes-per-iter 8 \\
      --bc-episodes 16 --eval-n 20 --out checkpoints/ppo_smoke   # quick check

Writes config.json, log.csv, init.pt (after behaviour cloning), best.pt (best
eval success on the first --eval-n val articles) and last.pt. Evaluate with:
  python -m scripts.evaluate --agent ppo --set checkpoint=checkpoints/ppo/best.pt
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from redactle.paths import ROOT
from redactle.rl.ppo import PPOConfig, Trainer


def main() -> None:
    ap = argparse.ArgumentParser()
    for f in fields(PPOConfig):
        flag = f"--{f.name.replace('_', '-')}"
        if isinstance(f.default, bool):     # --use-lm / --no-use-lm
            ap.add_argument(flag, action=argparse.BooleanOptionalAction,
                            default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    ap.add_argument("--out", default=ROOT / "checkpoints" / "ppo")
    args = vars(ap.parse_args())
    out = args.pop("out")
    Trainer(PPOConfig(**args), out).run()


if __name__ == "__main__":
    main()
