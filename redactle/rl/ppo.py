"""PPO over candidate sets (see features.py and policy.py).

Training episodes use *training* articles with the puzzle's own article
removed from the belief's possible worlds (leave-one-out); evaluation uses
val articles, which are never worlds.

Pipeline:
  1. Behaviour cloning: imitate the belief agent's choice (expected count with
     a title-length bonus) so PPO starts from a competent policy.
  2. PPO with GAE, rollouts collected in parallel worker processes.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from redactle.env import RedactleEnv, load_articles
from redactle.rl.features import N_CAND_FEATURES
from redactle.rl.policy import CandidatePolicy


@dataclass
class PPOConfig:
    iterations: int = 200
    episodes_per_iter: int = 32
    max_guesses: int = 500
    success_reward: float = 500.0
    title_word_reward: float = 0.0   # optional shaping per title word found
    reward_scale: float = 0.01
    gamma: float = 0.999
    gae_lambda: float = 0.95
    lr: float = 3e-4
    epochs: int = 4
    minibatch: int = 1024
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 128
    bc_episodes: int = 128
    bc_epochs: int = 10
    bc_minibatch: int = 256
    eval_every: int = 10
    eval_n: int = 100
    workers: int = 6
    use_hunt: bool = False    # add title-hunt candidates and features
    seed: int = 0


# ---------------------------------------------------------------------------
# Rollout workers
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(hidden: int, max_guesses: int, use_hunt: bool = False) -> None:
    torch.set_num_threads(1)
    from redactle.agents.data import load_vocab
    from redactle.rl.features import CandidateFeaturizer
    _W["feat"] = CandidateFeaturizer(load_vocab(min_df=2), max_guesses=max_guesses,
                                     use_hunt=use_hunt)
    _W["policy"] = CandidatePolicy(hidden, use_hunt)
    _W["use_hunt"] = use_hunt
    _W["max_guesses"] = max_guesses


def teacher_action(cand: np.ndarray, mask: np.ndarray, glob: np.ndarray,
                   use_hunt: bool = False) -> int:
    """The BeliefAgent rule: belief expected count x (1 + bonus * title-length
    match), with the end-game bonus 10 once the top world has weight >= 0.5,
    else 2 (endgame_conf=0.5). With use_hunt, in the end-game and when
    neither of the last 2 guesses was a hunt, the title-length candidate with
    the best hunt score instead (hunt_every=3)."""
    endgame = glob[1] >= 0.5
    if use_hunt and endgame and glob[-1] == 0:
        h = cand[:, N_CAND_FEATURES]
        ok = mask & (cand[:, 2] > 0) & (h > 0)
        if ok.any():
            return int(np.argmax(np.where(ok, h, -np.inf)))
    score = np.expm1(cand[:, 0])
    bonus = 10.0 if endgame else 2.0
    score = score * (1 + bonus * cand[:, 2])
    return int(np.argmax(np.where(mask, score, -np.inf)))


def _run_episode(task: dict) -> dict:
    """task: article, exclude_row, mode ('sample' | 'greedy' | 'teacher'),
    state_dict (numpy), seed, success_reward, title_word_reward."""
    feat, policy = _W["feat"], _W["policy"]
    if task.get("state_dict") is not None:
        policy.load_state_dict({k: torch.from_numpy(v)
                                for k, v in task["state_dict"].items()})
    gen = torch.Generator().manual_seed(task["seed"])
    env = RedactleEnv([task["article"]], max_guesses=_W["max_guesses"],
                      success_reward=task["success_reward"])
    obs = env.reset(0)
    feat.reset(exclude_row=task["exclude_row"])
    buf = {k: [] for k in ("cand", "mask", "glob", "action", "logp", "value", "reward")}
    done, info, found = False, {}, 0
    while not done:
        s = feat.featurize(obs)
        if not s.mask.any():           # nothing sensible left: any unguessed word
            obs, _, done, info = env.step(f"zz{obs.n_guesses}")
            continue
        if task["mode"] == "teacher":
            a = teacher_action(s.cand, s.mask, s.glob, use_hunt=_W["use_hunt"])
            logp, value = 0.0, 0.0
        else:
            with torch.no_grad():
                logits, v = policy(torch.from_numpy(s.cand)[None],
                                   torch.from_numpy(s.mask)[None],
                                   torch.from_numpy(s.glob)[None])
            dist = torch.distributions.Categorical(logits=logits[0])
            a = (int(logits[0].argmax()) if task["mode"] == "greedy"
                 else int(torch.multinomial(dist.probs, 1, generator=gen)))
            logp, value = float(dist.log_prob(torch.tensor(a))), float(v[0])
        obs, r, done, info = env.step(s.words[a])
        now = len(env.puzzle.targets & obs.revealed)
        r += task["title_word_reward"] * (now - found)
        found = now
        for k, x in zip(buf, (s.cand, s.mask, s.glob, a, logp, value, r)):
            buf[k].append(x)
    out = {k: np.array(v) for k, v in buf.items()}
    out.update(title=task["article"]["title"], solved=bool(info.get("solved")),
               guesses=obs.n_guesses, title_found=found,
               n_title=len(env.puzzle.targets))
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def gae(rewards, values, gamma, lam):
    adv = np.zeros_like(values)
    last = 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0   # terminal at end
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


class Trainer:
    def __init__(self, cfg: PPOConfig, out_dir: Path):
        self.cfg, self.out = cfg, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
        torch.manual_seed(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.train = load_articles(split="train")   # row order = belief worlds
        self.val = load_articles(split="val")[:cfg.eval_n]
        self.policy = CandidatePolicy(cfg.hidden, cfg.use_hunt)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        # One BLAS thread per worker; otherwise workers oversubscribe the CPU.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
            os.environ[var] = "1"
        self.pool = mp.get_context("spawn").Pool(
            cfg.workers, _init_worker,
            (cfg.hidden, cfg.max_guesses, cfg.use_hunt))
        self.log_path = self.out / "log.csv"
        self.best = -1.0

    # -- rollouts -----------------------------------------------------------
    def _sd(self):
        return {k: v.detach().numpy().copy() for k, v in self.policy.state_dict().items()}

    def rollouts(self, n: int, mode: str, articles=None) -> list[dict]:
        sd = None if mode == "teacher" else self._sd()
        tasks = []
        for i in range(n):
            if articles is None:
                row = self.rng.randrange(len(self.train))
                art, excl = self.train[row], row
            else:
                art, excl = articles[i], None
            tasks.append({"article": art, "exclude_row": excl, "mode": mode,
                          "state_dict": sd, "seed": self.rng.randrange(2**31),
                          "success_reward": self.cfg.success_reward,
                          "title_word_reward": self.cfg.title_word_reward})
        return list(self.pool.imap_unordered(_run_episode, tasks))

    @staticmethod
    def _stack(eps, key):
        return torch.from_numpy(np.concatenate([e[key] for e in eps if len(e["action"])]))

    # -- behaviour cloning ----------------------------------------------------
    def behaviour_clone(self) -> None:
        cfg = self.cfg
        t0 = time.time()
        eps = self.rollouts(cfg.bc_episodes, "teacher")
        cand, mask, glob = (self._stack(eps, k) for k in ("cand", "mask", "glob"))
        act = self._stack(eps, "action").long()
        print(f"BC: {len(act)} steps from {len(eps)} teacher episodes "
              f"({np.mean([e['solved'] for e in eps]):.0%} solved, "
              f"{time.time() - t0:.0f}s)", flush=True)
        for epoch in range(cfg.bc_epochs):
            perm, tot, acc = torch.randperm(len(act)), 0.0, 0.0
            for i in range(0, len(act), cfg.bc_minibatch):
                b = perm[i:i + cfg.bc_minibatch]
                logits, _ = self.policy(cand[b], mask[b], glob[b])
                loss = F.cross_entropy(logits, act[b])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                tot += float(loss) * len(b)
                acc += float((logits.argmax(1) == act[b]).sum())
            print(f"  BC epoch {epoch + 1}: loss {tot / len(act):.3f}, "
                  f"agreement {acc / len(act):.1%}", flush=True)

    # -- PPO ----------------------------------------------------------------
    def update(self, eps: list[dict]) -> dict:
        cfg = self.cfg
        advs, rets = [], []
        for e in eps:
            if len(e["action"]):
                a, r = gae(e["reward"] * cfg.reward_scale, e["value"],
                           cfg.gamma, cfg.gae_lambda)
                advs.append(a); rets.append(r)
        adv = torch.from_numpy(np.concatenate(advs)).float()
        ret = torch.from_numpy(np.concatenate(rets)).float()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        cand, mask, glob = (self._stack(eps, k) for k in ("cand", "mask", "glob"))
        act = self._stack(eps, "action").long()
        old_logp = self._stack(eps, "logp").float()
        stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "kl": 0.0, "n": 0}
        for _ in range(cfg.epochs):
            perm = torch.randperm(len(act))
            for i in range(0, len(act), cfg.minibatch):
                b = perm[i:i + cfg.minibatch]
                logits, value = self.policy(cand[b], mask[b], glob[b])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(act[b])
                ratio = (logp - old_logp[b]).exp()
                pi_loss = -torch.min(ratio * adv[b], ratio.clamp(
                    1 - cfg.clip, 1 + cfg.clip) * adv[b]).mean()
                v_loss = F.mse_loss(value, ret[b])
                ent = dist.entropy().mean()
                loss = pi_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                for k, x in (("pi_loss", pi_loss), ("v_loss", v_loss), ("entropy", ent),
                             ("kl", (old_logp[b] - logp).mean())):
                    stats[k] += float(x)
                stats["n"] += 1
        n = stats.pop("n")
        return {k: v / n for k, v in stats.items()} | {"steps": len(act)}

    def evaluate(self) -> dict:
        eps = self.rollouts(len(self.val), "greedy", self.val)
        g = [e["guesses"] if e["solved"] else self.cfg.max_guesses for e in eps]
        return {"eval_solved": np.mean([e["solved"] for e in eps]),
                "eval_capped": float(np.mean(g))}

    def save(self, name: str) -> None:
        torch.save({"state_dict": self.policy.state_dict(),
                    "config": asdict(self.cfg)}, self.out / name)

    LOG_FIELDS = ["iter", "train_solved", "train_guesses", "train_title_frac",
                  "pi_loss", "v_loss", "entropy", "kl", "steps", "secs",
                  "eval_solved", "eval_capped"]

    def log(self, row: dict) -> None:
        # Fixed columns: eval_* are only present every eval_every iterations.
        new = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.LOG_FIELDS, restval="")
            if new:
                w.writeheader()
            w.writerow(row)

    def run(self) -> None:
        cfg = self.cfg
        if cfg.bc_episodes:
            self.behaviour_clone()
        ev = self.evaluate()
        print(f"iter 0: eval {ev['eval_solved']:.0%} solved, "
              f"capped {ev['eval_capped']:.0f}", flush=True)
        self.save("init.pt")
        for it in range(1, cfg.iterations + 1):
            t0 = time.time()
            eps = self.rollouts(cfg.episodes_per_iter, "sample")
            st = self.update(eps)
            row = {"iter": it,
                   "train_solved": np.mean([e["solved"] for e in eps]),
                   "train_guesses": np.mean([e["guesses"] for e in eps]),
                   "train_title_frac": np.mean([e["title_found"] / e["n_title"] for e in eps]),
                   **st, "secs": time.time() - t0}
            if it % cfg.eval_every == 0 or it == cfg.iterations:
                row |= self.evaluate()
                if row["eval_solved"] > self.best:
                    self.best = row["eval_solved"]
                    self.save("best.pt")
            self.save("last.pt")
            self.log(row)
            print(" ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in row.items()), flush=True)
        self.pool.close()
