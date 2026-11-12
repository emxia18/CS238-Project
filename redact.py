"""Redactle-style tokenization and redaction.

Rules (mirroring the original Redactle):
  * Punctuation and whitespace are always visible.
  * A fixed list of common words (articles, prepositions, a few conjunctions /
    verbs) is always visible. List taken from
    https://github.com/foobuzz/auto-redactle/blob/master/redactle-common-words.txt
  * Every other word is hidden as a block of the same length.
  * Matching is case-insensitive, ignores diacritics (é == e) and apostrophes
    (McDonald's == mcdonalds), and reveals related forms: he -> He/him/his,
    ring -> rings, run -> ran/running (see match_key). Hyphenated words are
    separate tokens.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import simplemma

REDACTLE_COMMON_WORDS = frozenset("""
a aboard about above across after against along amid among an and around as at
because before behind below beneath beside between beyond but by concerning
considering despite down during except following for from if in inside into is
it like minus near next of off on onto opposite or out outside over past per
plus regarding round save since than the through till to toward under
underneath unlike until up upon versus via was with within without
""".split())

BLOCK = "█"  # █

# A word is a run of letters/digits, optionally joined by internal apostrophes.
WORD_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")


def normalize(word: str) -> str:
    """Key used to compare a guess against article words."""
    word = unicodedata.normalize("NFKD", word)
    word = "".join(c for c in word if not unicodedata.combining(c))
    return word.replace("'", "").replace("’", "").lower()


# Pronoun families: guessing any member reveals all of them (he -> him, his).
PRONOUN_FAMILIES = [
    "he him his himself",
    "she her hers herself",
    "they them their theirs themselves themself",
    "i me my mine myself",
    "we us our ours ourselves",
    "you your yours yourself yourselves",
    "it its itself",
]
_PRONOUN_KEY = {w: fam.split()[0] for fam in PRONOUN_FAMILIES for w in fam.split()}


@lru_cache(maxsize=200_000)
def match_key(word: str) -> str:
    """Key that decides which words a guess reveals. Case, accents and
    apostrophes are ignored, and related forms share a key: pronoun families
    (he/him/his), plurals (ring/rings), verb forms (run/ran/running) and
    possessives (Chaplin's/Chaplin)."""
    word = re.sub(r"['’][sS]$", "", word)  # possessive 's
    key = normalize(word)
    if key in _PRONOUN_KEY:
        return _PRONOUN_KEY[key]
    return simplemma.lemmatize(key, lang="en").lower()


def tokenize(text: str) -> list[tuple[str, bool]]:
    """Split text into (piece, is_word) pairs that concatenate back to text."""
    pieces, pos = [], 0
    for m in WORD_RE.finditer(text):
        if m.start() > pos:
            pieces.append((text[pos:m.start()], False))
        pieces.append((m.group(), True))
        pos = m.end()
    if pos < len(text):
        pieces.append((text[pos:], False))
    return pieces


def is_always_visible(word: str, reveal_numbers: bool = False) -> bool:
    key = normalize(word)
    return key in REDACTLE_COMMON_WORDS or (reveal_numbers and key.isdigit())


def hidden(word: str, show_length: bool = False) -> str:
    """Blocks for a hidden word; with show_length the length is written in the
    middle (e.g. '██7██') so the rendered width still equals the word length."""
    n = len(word)
    label = str(n)
    if not show_length or n < len(label) + 2:
        return BLOCK * n
    left = (n - len(label)) // 2
    return BLOCK * left + label + BLOCK * (n - len(label) - left)


def redact(text: str, revealed: set[str] = frozenset(),
           reveal_numbers: bool = False, show_lengths: bool = False) -> str:
    """Render text with every word hidden unless common or in `revealed`
    (a set of match_key keys)."""
    out = []
    for piece, is_word in tokenize(text):
        if (not is_word or match_key(piece) in revealed
                or is_always_visible(piece, reveal_numbers)):
            out.append(piece)
        else:
            out.append(hidden(piece, show_lengths))
    return "".join(out)


def count_matches(text: str, key: str) -> int:
    """How many words in text a guess with match_key `key` reveals."""
    return sum(is_word and match_key(p) == key for p, is_word in tokenize(text))


def target_keys(title: str) -> list[str]:
    """match_key keys of the title words the player must reveal to win."""
    keys = []
    for piece, is_word in tokenize(title):
        if not is_word or normalize(piece) in REDACTLE_COMMON_WORDS:
            continue
        key = match_key(piece)
        if key not in keys:
            keys.append(key)
    return keys
