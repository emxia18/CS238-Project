"""Tests for the RL candidate featurizer (needs data/ built: vocab + counts)."""

import numpy as np
import pytest

from redactle.agents.data import VOCAB, load_vocab
from redactle.env import RedactleEnv
from redactle.rl.features import K, CandidateFeaturizer

pytestmark = pytest.mark.skipif(not VOCAB.exists(), reason="data/ not built")

ART = {
    "id": "t", "url": "", "level": 4, "split": "val",
    "title": "Neptune",
    "text": "Neptune is the eighth planet from the Sun. "
            "Pneumonoultramicroscopicsilicovolcanoconiosislike words exist. " * 40,
}


@pytest.fixture(scope="module")
def feat():
    return CandidateFeaturizer(load_vocab(min_df=2))


def test_featurize_shapes_and_word_longer_than_vocab(feat):
    env = RedactleEnv([ART])
    obs = env.reset(0)
    feat.reset()
    for _ in range(5):
        s = feat.featurize(obs)
        assert s.cand.shape[0] == K and s.mask.any()
        assert np.isfinite(s.cand).all() and np.isfinite(s.glob).all()
        obs, *_ = env.step(s.words[int(np.argmax(s.mask))])


def test_leave_one_out_excludes_world(feat):
    feat.reset(exclude_row=0)
    feat.length = 1000  # normally set from the first observation
    _, top, w = feat.expected_counts()
    assert 0 not in top[w > 0]


def test_teacher_matches_belief_rule():
    from redactle.rl.ppo import teacher_action
    cand = np.zeros((K, 12), dtype=np.float32)
    cand[0, 0], cand[1, 0], cand[1, 2] = np.log1p(10), np.log1p(4), 1  # 10 vs 4*3
    mask = np.zeros(K, bool); mask[:2] = True
    glob = np.zeros(10, np.float32)
    assert teacher_action(cand, mask, glob) == 1
    glob[1] = 0.9      # end-game: title bonus 10
    cand[1, 0] = np.log1p(1)
    assert teacher_action(cand, mask, glob) == 1


def test_hunt_features_and_teacher():
    from redactle.rl.features import n_cand_features, n_global_features
    from redactle.rl.ppo import teacher_action
    f = CandidateFeaturizer(load_vocab(min_df=2), use_hunt=True)
    env = RedactleEnv([ART])
    obs = env.reset(0)
    f.reset()
    for _ in range(3):
        s = f.featurize(obs)
        assert s.cand.shape == (K, n_cand_features(True))
        assert s.glob.shape == (n_global_features(True),)
        assert np.isfinite(s.cand).all() and np.isfinite(s.glob).all()
        obs, *_ = env.step(s.words[teacher_action(s.cand, s.mask, s.glob,
                                                  use_hunt=True)])
    # End-game and no recent hunt: the teacher takes the best hunt candidate.
    cand = np.zeros((K, n_cand_features(True)), dtype=np.float32)
    cand[0, 0] = np.log1p(100)                     # belief favourite
    cand[1, 2], cand[1, 12] = 1, 0.5               # title-length hunt word
    mask = np.zeros(K, bool); mask[:2] = True
    glob = np.zeros(n_global_features(True), np.float32)
    glob[1] = 0.9
    assert teacher_action(cand, mask, glob, use_hunt=True) == 1
    glob[-1] = 0.5                                 # hunted last turn
    assert teacher_action(cand, mask, glob, use_hunt=True) == 0
