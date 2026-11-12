"""Fetch Wikipedia's Vital Articles list (Levels 1-4, ~10k well-known topics).

The original Redactle drew its puzzles from this list. Each title gets the
lowest (most important) level it appears in: Level 1 = 10 articles,
Level 2 = 100, Level 3 = 1,000, Level 4 = 10,000 (levels are nested).

Usage: python vital.py [--out vital_articles.json]
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://en.wikipedia.org/w/api.php"
REQUEST_DELAY = 1.0  # seconds between requests (Wikimedia rate limits)
HEADERS = {"User-Agent": "cs238-redactle-project/0.1 (student course project)"}

LEVEL_PAGES = {
    1: ["Wikipedia:Vital articles/Level/1"],
    2: ["Wikipedia:Vital articles/Level/2"],
    3: ["Wikipedia:Vital articles/Level 3"],
}
LEVEL4_PREFIX = "Vital articles/Level 4/"
SKIP_WORDS = ("alert", "archive", "draft", "removed", "candidates")


def api(**params) -> dict:
    params |= {"format": "json", "formatversion": "2"}
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}",
                                 headers=HEADERS)
    for attempt in range(8):
        time.sleep(REQUEST_DELAY)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:  # 429 = rate limited
            wait = max(int(e.headers.get("Retry-After") or 0), 5 * 2 ** attempt)
            print(f"    HTTP {e.code}, waiting {wait}s")
            time.sleep(wait)
        except OSError:
            time.sleep(5 * 2 ** attempt)
    raise RuntimeError(f"API request failed: {params}")


def query_all(**params):
    """Yield every result page of a query, following continuation."""
    cont: dict = {}
    while True:
        data = api(action="query", **params, **cont)
        yield data
        if "continue" not in data:
            return
        cont = data["continue"]


def level4_pages() -> list[str]:
    pages = []
    for data in query_all(list="allpages", apprefix=LEVEL4_PREFIX, apnamespace=4,
                          apfilterredir="nonredirects", aplimit="max"):
        for p in data["query"]["allpages"]:
            title = p["title"]
            sub = title.split(LEVEL4_PREFIX, 1)[1]
            if sub and not any(w in title.lower() for w in SKIP_WORDS):
                pages.append(title)
    return pages


def article_links(page: str) -> set[str]:
    """All main-namespace articles linked from a Vital Articles list page."""
    titles = set()
    for data in query_all(titles=page, prop="links", plnamespace=0,
                          pllimit="max", redirects=1):
        for p in data["query"]["pages"]:
            titles.update(link["title"] for link in p.get("links", []))
    return titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="vital_articles.json")
    args = ap.parse_args()

    pages = dict(LEVEL_PAGES)
    pages[4] = level4_pages()

    levels: dict[str, int] = {}
    for level in sorted(pages):
        found = set()
        for page in pages[level]:
            links = article_links(page)
            print(f"  L{level} {page}: {len(links)} links")
            found |= links
        for title in found:
            levels.setdefault(title, level)
        print(f"Level {level}: {len(found)} articles")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(levels, indent=0, ensure_ascii=False, sort_keys=True))
    counts = {lv: sum(v == lv for v in levels.values()) for lv in sorted(pages)}
    print(f"{len(levels)} titles -> {out}  (new per level: {counts})")


if __name__ == "__main__":
    main()
