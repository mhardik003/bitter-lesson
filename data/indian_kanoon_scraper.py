#!/usr/bin/env python3
"""
indian_kanoon_scraper.py
Search Indian Kanoon for keywords, paginate results, and save judgment links.

Usage:
  python indian_kanoon_scraper.py --keywords "Section 10 Indian Contract Act" \
      --max-pages 50 --delay 1.5 --out links.txt

Notes:
- Respects site structure by using the public search interface.
- Adds a delay between requests; increase if you scale up.
- Doesn’t fetch full judgments (heavy). It saves links; you can later fetch selectively.
"""

import argparse
import time
import sys
from urllib.parse import urljoin, urlencode

import requests
from bs4 import BeautifulSoup

BASE = "https://indiankanoon.org/search/"
HEADERS = {
    "User-Agent": "ResearchBot/1.0 (+contact: your-email@example.com)"
}

def build_search_url(query: str, page: int) -> str:
    params = {"formInput": query}
    if page > 1:
        params["pagenum"] = page
    return f"{BASE}?{urlencode(params)}"

def extract_doc_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    # Primary: canonical judgment pages live under /doc/<id>/
    for a in soup.select('a[href^="/doc/"], a[href^="https://indiankanoon.org/doc/"]'):
        href = a.get("href", "")
        # Normalize and ignore fragments / anchors
        if href.startswith("/doc/") or href.startswith("https://indiankanoon.org/doc/"):
            # Some links lack trailing slash; both are fine
            links.add(href)
    # Make them absolute
    return [urljoin("https://indiankanoon.org", h) for h in links]

def page_has_results(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    # Heuristic: search result list divs; if none, we likely reached the end
    results = soup.select("div.result, div[id^=p]")
    return bool(results)

def scrape(query: str, max_pages: int, delay: float) -> list[str]:
    session = requests.Session()
    session.headers.update(HEADERS)

    all_links = set()
    for page in range(1, max_pages + 1):
        url = build_search_url(query, page)
        # print(url)
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"[warn] page {page} failed: {e}", file=sys.stderr)
            break

        links = extract_doc_links(r.text)
        if not links and not page_has_results(r.text):
            # No more results
            break

        before = len(all_links)
        all_links.update(links)
        added = len(all_links) - before
        print(f"[info] page {page}: found {added} new links (total {len(all_links)})")

        # Simple, polite throttle
        time.sleep(max(delay, 0.5))
    return sorted(all_links)

def main():
    ap = argparse.ArgumentParser(description="Search Indian Kanoon and save judgment links.")
    ap.add_argument("--keywords", required=True, help="Search query (quotes recommended).")
    ap.add_argument("--max-pages", type=int, default=30, help="Maximum pages to fetch.")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds).")
    ap.add_argument("--out", default="indiankanoon_links.txt", help="Output text file.")
    args = ap.parse_args()

    links = scrape(args.keywords, args.max_pages, args.delay)
    if not links:
        print("[info] no links found. Try broadening the query or increasing --max-pages.")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        for url in links:
            f.write(url + "\n")

    print(f"[done] wrote {len(links)} links to {args.out}")

if __name__ == "__main__":
    main()
