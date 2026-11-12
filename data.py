"""Load Wikipedia articles, filter them into playable Redactle puzzles, and save.

By default the articles are Wikipedia's Vital Articles (~10k well-known topics,
listed in vital_articles.json; refresh that file with `python vital.py`).
Their text comes from the Hugging Face Wikipedia dump. The first run scans the
whole dump once (~11 GB download, files deleted as it goes) and caches the matches to data/vital_raw.jsonl.

Usage:
  python data.py                          # vital articles
  python data.py --source random --n 10000  # random sample of all Wikipedia
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset

from redact import redact, target_keys, tokenize

# ---- filter thresholds ------------------------------------------------------
MIN_WORDS = 300            # text too short
MAX_TITLE_WORDS = 4        # title too long (counts words the player must find)
MAX_TITLE_CHARS = 40
MIN_LATIN_FRACTION = 0.95  # share of letters in Latin script
# Share of words that are English stopwords. English prose is ~0.30-0.45;
# foreign-language text scores near 0, table/list-heavy pages ~0.05-0.20.
MIN_ENGLISH_STOPWORD_FRACTION = 0.10  # below: treated as not English
MIN_PROSE_STOPWORD_FRACTION = 0.20    # below: mostly tables/lists, not prose

# Trailing sections that aren't prose; everything from the first of these
# headings onward is dropped.
TRAILING_SECTIONS = {
    "see also", "references", "external links", "further reading", "notes",
    "bibliography", "sources", "citations", "footnotes", "notes and references",
    "works cited", "selected bibliography", "discography", "filmography",
}

# Common English words, used only to detect English text (not for redaction).
ENGLISH_STOPWORDS = frozenset("""
the of and to in a is was for on as by with that at from his her its it an
are were be been has had have which this or but not their they he she also
who after first one two new into other more when there than all can about
""".split())


def clean_text(text: str) -> str:
    """Drop reference/link sections and collapse excess blank lines."""
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.strip().lower() in TRAILING_SECTIONS:
            lines = lines[:i]
            break
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_title(title: str) -> str:
    """'Mercury (planet)' -> 'Mercury'."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", title).strip()


def latin_fraction(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    latin = sum(unicodedata.name(c, "").startswith("LATIN") for c in letters)
    return latin / len(letters)


def english_stopword_fraction(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(w.lower() in ENGLISH_STOPWORDS for w in words) / len(words)


def check(title: str, text: str) -> str | None:
    """Return the rejection reason, or None if the article is playable."""
    if title.startswith(("List of", "Lists of")) or "may refer to" in text[:500]:
        return "list_or_disambiguation"

    words = [p for p, is_word in tokenize(text) if is_word]
    if len(words) < MIN_WORDS:
        return "text_too_short"

    stop_frac = english_stopword_fraction(words)
    if (latin_fraction(title) < MIN_LATIN_FRACTION
            or latin_fraction(text) < MIN_LATIN_FRACTION
            or stop_frac < MIN_ENGLISH_STOPWORD_FRACTION):
        return "not_english"
    if stop_frac < MIN_PROSE_STOPWORD_FRACTION:
        return "not_prose"

    keys = target_keys(title)
    if not keys:
        return "title_has_no_target_words"
    if len(keys) > MAX_TITLE_WORDS or len(title) > MAX_TITLE_CHARS:
        return "title_too_long"
    return None


DUMP_REPO, DUMP_CONFIG = "wikimedia/wikipedia", "20231101.en"


def wikipedia_stream():
    return load_dataset(DUMP_REPO, DUMP_CONFIG, split="train", streaming=True)


def _scan_file(filename: str, titles: list[str], tmpdir: str) -> list[dict]:
    """Download one dump parquet file, keep rows whose title is vital, delete it."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(DUMP_REPO, filename, repo_type="dataset",
                           local_dir=tmpdir)
    try:
        table = pq.read_table(path)
        table = table.filter(pc.is_in(table["title"], value_set=pa_array(titles)))
        return table.to_pylist()
    finally:
        os.remove(path)


def pa_array(values):
    import pyarrow as pa
    return pa.array(values)


def load_vital_raw(levels: dict[str, int], cache: Path,
                   workers: int = 4) -> list[dict]:
    """Raw dump entries for the vital titles. The first call downloads the dump
    one parquet file at a time (deleting each after use) and caches matches."""
    if not cache.exists():
        from huggingface_hub import HfFileSystem
        files = sorted(Path(f).name for f in HfFileSystem().glob(
            f"datasets/{DUMP_REPO}/{DUMP_CONFIG}/*.parquet"))
        print(f"One-time scan of {len(files)} dump files for {len(levels)} "
              f"vital articles...", flush=True)
        titles = list(levels)
        rows = []
        cache.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache.parent) as tmpdir, \
                ThreadPoolExecutor(workers) as pool:
            jobs = [pool.submit(_scan_file, f"{DUMP_CONFIG}/{name}", titles, tmpdir)
                    for name in files]
            for i, job in enumerate(as_completed(jobs), 1):
                rows.extend(job.result())
                print(f"  {i}/{len(files)} files, {len(rows)} found", flush=True)
        tmp = cache.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.rename(cache)
    with cache.open() as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["vital", "random"], default="vital")
    ap.add_argument("--vital-list", default="vital_articles.json")
    ap.add_argument("--vital-cache", default="data/vital_raw.jsonl")
    ap.add_argument("--n", type=int, default=10_000,
                    help="articles to sample (--source random)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/articles.jsonl")
    args = ap.parse_args()

    if args.source == "vital":
        levels = json.loads(Path(args.vital_list).read_text())
        sampled = load_vital_raw(levels, Path(args.vital_cache))
        print(f"found {len(sampled)} / {len(levels)} vital articles in the dump")
    else:
        levels = {}
        sampled = wikipedia_stream().shuffle(
            seed=args.seed, buffer_size=10_000).take(args.n)

    kept, rejected = [], {}
    n_total = 0
    for art in sampled:
        n_total += 1
        title, text = clean_title(art["title"]), clean_text(art["text"])
        reason = check(title, text)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        kept.append({
            "id": art["id"],
            "url": art["url"],
            "level": levels.get(art["title"]),
            "title": title,
            "text": text,
            "target_keys": target_keys(title),
            "redacted_title": redact(title),
            "redacted_text": redact(text),
        })

    # Disjoint train / val / test split (80 / 10 / 10).
    random.Random(args.seed).shuffle(kept)
    n_val = n_test = len(kept) // 10
    for i, art in enumerate(kept):
        art["split"] = ("test" if i < n_test
                        else "val" if i < n_test + n_val else "train")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for art in kept:
            f.write(json.dumps(art, ensure_ascii=False) + "\n")

    print(f"kept {len(kept)} / {n_total} -> {out}")
    for reason, count in sorted(rejected.items(), key=lambda kv: -kv[1]):
        print(f"  rejected {reason}: {count}")
    if kept:
        ex = kept[0]
        print(f"\nexample: {ex['title']}  ->  {ex['redacted_title']}")
        print(ex["redacted_text"][:400])


if __name__ == "__main__":
    main()
