#!/usr/bin/env python3
# indian_kanoon_scraper.py (robust)
import argparse, sys, time, unicodedata
from pathlib import Path
from urllib.parse import urlencode, urljoin
import requests
from bs4 import BeautifulSoup

SEARCH_BASE = "https://indiankanoon.org/search/"
HEADERS = {
    "User-Agent": "IK-ResearchBot/1.1 (+contact: your-email@example.com)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-GB,en;q=0.9",
    "Connection": "keep-alive",
}

def normalize_query(q: str) -> str:
    # Replace curly quotes and weird spaces with simple ASCII
    q = unicodedata.normalize("NFKC", q)
    q = q.replace("“", '"').replace("”", '"').replace("’", "'").strip()
    return q

def build_urls(query: str, page: int):
    # Try both param styles; some setups prefer one or the other.
    params1 = {"formInput": query}
    params2 = {"q": query}
    if page > 1:
        params1["pagenum"] = page
        params2["pagenum"] = page
    return (
        f"{SEARCH_BASE}?{urlencode(params1)}",
        f"{SEARCH_BASE}?{urlencode(params2)}",
    )

def fetch_html(session: requests.Session, url: str) -> str | None:
    try:
        r = session.get(url, headers=HEADERS, timeout=40)
        r.raise_for_status()
        return r.text
    except requests.RequestException as e:
        print(f"[warn] GET failed {url}: {e}", file=sys.stderr)
        return None

def extract_doc_links(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    # Catch /doc/<id>/ and absolute variants (with extra query strings allowed)
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/doc/") or href.startswith("https://indiankanoon.org/doc/"):
            links.add(href.split("#")[0])  # drop fragments
    return [urljoin("https://indiankanoon.org", h) for h in links]

def page_has_any_results(html: str) -> bool:
    soup = BeautifulSoup(html, "html.parser")
    if soup.select("div.result, div#results, div[id^=p], a[href^='/doc/']"):
        return True
    t = soup.get_text(" ", strip=True).lower()
    if "no results found" in t or "did not match any documents" in t:
        return False
    return True

def read_keywords(path: Path) -> list[str]:
    out = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(normalize_query(s))
    return out

def load_existing(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    try:
        return set(u.strip() for u in out_path.read_text(encoding="utf-8").splitlines() if u.strip())
    except Exception:
        return set()

def append(out_path: Path, links: list[str]):
    if not links: return
    with out_path.open("a", encoding="utf-8") as f:
        for u in links:
            f.write(u + "\n")

def scrape_keyword(session, keyword, pages, delay, seen, out_path, debug=False):
    print(f"\n=== Query: {keyword!r} ===")
    total = 0
    saved_debug = False

    for page in range(1, pages + 1):
        url1, url2 = build_urls(keyword, page)

        html = fetch_html(session, url1)
        if html is None:
            html = fetch_html(session, url2)  # fallback
        if html is None:
            print(f"[warn] page {page} failed for {keyword!r}")
            time.sleep(max(delay, 0.5)); continue

        if debug and not saved_debug and page == 1:
            Path("debug_page1.html").write_text(html, encoding="utf-8")
            print("[debug] wrote debug_page1.html")
            saved_debug = True

        links = extract_doc_links(html)
        if not links and not page_has_any_results(html):
            print(f"[info] no more results after page {page-1} for {keyword!r}")
            break

        new_links = [u for u in links if u not in seen]
        if new_links:
            append(out_path, new_links)
            seen.update(new_links)
            total += len(new_links)
            print(f"[info] page {page}: +{len(new_links)} (total for query: {total})")
        else:
            print(f"[info] page {page}: 0 new links")
            break  # no new links, likely done

        time.sleep(max(delay, 0.8))  # be polite

    print(f"[done] {keyword!r}: added {total} links")

def main():
    ap = argparse.ArgumentParser(description="Append Indian Kanoon judgment links for multiple keywords.")
    ap.add_argument("--keywords-file", default="keywords.txt")
    ap.add_argument("--pages", type=int, default=40)
    ap.add_argument("--delay", type=float, default=1.2)
    ap.add_argument("--out", default="links.txt")
    ap.add_argument("--debug", action="store_true", help="Save first page HTML to debug_page1.html")
    args = ap.parse_args()

    kw_path, out_path = Path(args.keywords_file), Path(args.out)
    if not kw_path.exists():
        print(f"[error] missing {kw_path}", file=sys.stderr); sys.exit(1)

    keywords = read_keywords(kw_path)
    if not keywords:
        print(f"[error] no keywords in {kw_path}", file=sys.stderr); sys.exit(1)

    seen = load_existing(out_path)
    if seen: print(f"[info] loaded {len(seen)} existing links from {args.out}")

    session = requests.Session()
    session.headers.update(HEADERS)

    for kw in keywords:
        scrape_keyword(session, kw, args.pages, args.delay, seen, out_path, args.debug)

    print(f"\n[summary] unique links recorded: {len(seen)}")
    print(f"[where] appended to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
