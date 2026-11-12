"""Tests for the Redactle environment. Run: python tests/test_env.py (or pytest)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from env import RedactleEnv  # noqa: E402

ART = {
    "id": "1", "url": "", "level": 3, "split": "train",
    "title": "Charlie Chaplin",
    "text": "Charlie Chaplin was an English comic actor. He made his films "
            "in the United States; Chaplin's films were popular. McDonald's "
            "is unrelated. He ran, and running was his habit.",
}


def make_env(**kw):
    return RedactleEnv([ART], **kw)


def test_initial_observation_hides_words_but_not_common_ones():
    obs = make_env().reset()
    assert obs.title == [7, " ", 7]
    assert obs.body[:4] == [7, " ", 7, " "]
    assert "was" in obs.body and "in" in obs.body and "the" in obs.body
    seps = "".join(x for x in obs.body if isinstance(x, str))
    assert ";" in seps and "." in seps


def test_pronoun_family_and_case():
    env = make_env()
    env.reset()
    obs, _, _, info = env.step("he")
    assert info["count"] == 4  # He, his, He, his
    assert obs.body.count("He") == 2 and obs.body.count("his") == 2


def test_inflections_and_apostrophes():
    env = make_env()
    env.reset()
    _, _, _, info = env.step("run")
    assert info["count"] == 2  # ran, running
    _, _, _, info = env.step("McDonald")
    assert info["count"] == 1
    _, _, _, info = env.step("chaplin")  # title, body Chaplin, Chaplin's
    assert info["count"] == 3


def test_invalid_guesses():
    env = make_env()
    env.reset()
    env.step("he")
    _, r, _, info = env.step("him")
    assert info["reason"] == "already_guessed" and not info["valid"] and r == -1
    _, r, _, info = env.step("the")
    assert info["reason"] == "always_visible"
    _, _, _, info = env.step("are")  # every form of "be" is always visible
    assert info["reason"] == "always_visible"
    _, _, _, info = env.step("two words")
    assert info["reason"] == "not_a_word"
    assert env.n_guesses == 5


def test_invalid_guesses_free_when_not_counted():
    env = make_env(count_invalid=False)
    env.reset()
    env.step("he")
    _, r, _, _ = env.step("him")
    assert r == 0 and env.n_guesses == 1


def test_solve_reward():
    env = make_env(success_reward=500)
    env.reset()
    total = 0
    for w in ["comic", "charlie", "chaplin"]:
        _, r, done, info = env.step(w)
        total += r
    assert done and info["solved"] and info["title"] == "Charlie Chaplin"
    assert total == 500 - 3


def test_truncation():
    env = make_env(max_guesses=3)
    env.reset()
    for w in ["a1", "b1", "c1"]:
        _, r, done, info = env.step(w)
    assert done and info["truncated"] and not info["solved"]


def test_masked_text():
    obs = make_env().reset()
    text, lengths = obs.masked_text()
    assert text.startswith("<mask> <mask> was an <mask>")
    assert lengths[:3] == [7, 7, 7]


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} passed")
