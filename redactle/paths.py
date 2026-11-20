"""Project paths, anchored at the repository root so scripts work from any cwd."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
VITAL_LIST = DATA_DIR / "vital_articles.json"     # tracked: vital titles -> level
VITAL_RAW = DATA_DIR / "vital_raw.jsonl"          # cache of the raw dump matches
ARTICLES = DATA_DIR / "articles.jsonl"            # filtered puzzles with splits
VOCAB = DATA_DIR / "vocab_train.json"             # training-split vocabulary
COUNTS = DATA_DIR / "belief_counts_train.npz"     # train article x word counts
TRAIN_TITLES = DATA_DIR / "train_title_keys.json"  # title keys per train article

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
