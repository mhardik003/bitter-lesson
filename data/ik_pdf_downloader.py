#!/usr/bin/env python3
"""
ik_pdf_downloader.py

Reads Indian Kanoon /doc/ URLs from links.txt, downloads their PDFs, and:
- Skips URLs already processed (from a processed log, default: downloaded.txt)
- Appends each successfully saved URL to the processed log
- Optionally records failures in a separate failures log

Usage:
  python ik_pdf_downloader.py \
      --links links.txt \
      --outdir ./pdfs \
      --processed downloaded.txt \
      --failures failures.txt \
      --delay 1.2 \
      --max 0

Notes:
- Only marks a URL as processed after a successful PDF save.
- “Polite” by default; increase --delay if scaling up.
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE = "https://indiankanoon.org"
HEADERS = {
    "User-Agent": "Mozilla Firefox/1.2 (+contact: your-email@example.com)"
}

DOC_ID_RE = re.compile(r"/doc/(\d+)/?")

def sanitize_filename(s: str, maxlen: int = 140) -> str:
    s = re.sub(r"[^\w\-\.\(\)\[\] ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > maxlen:
        s = s[:maxlen].rstrip()
    return s or "document"

def guess_pdf_url(doc_url: str) -> list[str]:
    # Fallback guesses if the explicit PDF link isn't found on the page
    base = doc_url.rstrip("/")
    return [
        # base + "/?format=pdf",
        base + "/?type=pdf",
        # base + "/?download=1",
        # base + "/?pdf=1",
    ]

def extract_title_and_pdf_link(html: str, page_url: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    title = None
    h2 = soup.find("h2")
    if h2 and h2.get_text(strip=True):
        title = h2.get_text(strip=True)
    if not title and soup.title and soup.title.string:
        title = soup.title.string.strip()

    pdf_url = None
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True).lower()
        href = a["href"]
        if "get this document in pdf" in text or "download pdf" in text or href.lower().endswith(".pdf"):
            pdf_url = urljoin(page_url, href)
            break
    if not pdf_url:
        # Any anchor that mentions 'pdf' in href
        a = soup.select_one('a[href*="pdf"]')
        if a:
            pdf_url = urljoin(page_url, a["href"])

    return title, pdf_url

def fetch(session: requests.Session, url: str) -> requests.Response | None:
    try:
        r = session.get(url, timeout=40)
        r.raise_for_status()
        return r
    except requests.RequestException as e:
        print(f"[warn] GET failed {url}: {e}", file=sys.stderr)
        return None

def download_pdf(session: requests.Session, pdf_url: str, out_path: Path) -> bool:
    try:
        with session.get(pdf_url, timeout=120, stream=True) as r:
            r.raise_for_status()
            # Content-type sanity; allow mislabelled PDFs if bytes begin with %PDF-
            ctype = (r.headers.get("Content-Type") or "").lower()
            # print(f"[info] downloading PDF from {pdf_url} (Content-Type: {ctype})")
            if "pdf" not in ctype:
                # Peek first 5 bytes
                first = r.raw.read(5, decode_content=True)
                if first != b"%PDF-":
                    print(f"[warn] Not a PDF (ctype={ctype}) from {pdf_url}", file=sys.stderr)
                    return False
                chunk_iter = iter([first] + list(r.iter_content(chunk_size=8192)))
            else:
                chunk_iter = r.iter_content(chunk_size=8192)

            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(out_path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in chunk_iter:
                    if not chunk:
                        continue
                    f.write(chunk)
            os.replace(tmp, out_path)
            return True
    except requests.RequestException as e:
        print(f"[warn] PDF download failed {pdf_url}: {e}", file=sys.stderr)
        return False

def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []

def load_set(path: Path) -> set[str]:
    return set(load_lines(path))

def append_line(path: Path, text: str):
    with path.open("a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def normalize_doc_url(u: str) -> str:
    # Make absolute, strip fragments, collapse dup slashes
    if not u.startswith("http"):
        u = urljoin(BASE, u)
    parts = urlparse(u)
    clean = parts._replace(fragment="").geturl()
    return clean

def main():
    ap = argparse.ArgumentParser(description="Download PDFs from Indian Kanoon /doc/ pages with resume support.")
    ap.add_argument("--links", default="links.txt", help="Input file with /doc/ URLs (one per line).")
    ap.add_argument("--outdir", default="sources", help="Directory to save PDFs.")
    ap.add_argument("--processed", default="downloaded.txt", help="Log file storing URLs already downloaded.")
    ap.add_argument("--failures", default="failures.txt", help="Optional log file for failed URLs.")
    ap.add_argument("--delay", type=float, default=4.0, help="Delay between requests (seconds).")
    ap.add_argument("--max", type=int, default=0, help="Max URLs to process from the input (0 = no limit).")
    args = ap.parse_args()

    links_path = Path(args.links)
    outdir = Path(args.outdir)
    processed_path = Path(args.processed)
    failures_path = Path(args.failures) if args.failures else None

    if not links_path.exists():
        print(f"[error] links file not found: {links_path}", file=sys.stderr)
        sys.exit(1)

    all_urls = load_lines(links_path)
    if args.max > 0:
        all_urls = all_urls[:args.max]

    # Normalize input URLs and keep order
    all_urls = [normalize_doc_url(u) for u in all_urls if "/doc/" in u]

    # Load processed set to skip
    processed = load_set(processed_path)
    if processed:
        print(f"[info] loaded {len(processed)} processed URLs from {processed_path}")

    session = requests.Session()
    session.headers.update(HEADERS)

    count_total = 0
    count_skipped = 0
    count_saved = 0

    for i, doc_url in enumerate(all_urls, 1):
        
        if doc_url in processed:
            count_skipped += 1
            if i % 50 == 0:
                print(f"[info] skipped so far: {count_skipped}")
            continue

        if "/doc/" not in urlparse(doc_url).path:
            print(f"[skip] not a /doc/ URL: {doc_url}", file=sys.stderr)
            time.sleep(max(args.delay, 0.5))
            continue

        m = DOC_ID_RE.search(doc_url)
        doc_id = m.group(1) if m else f"doc_{i}"

        # Fetch judgment page to find the PDF link + title
        r = fetch(session, doc_url)
        if not r:
            if failures_path:
                append_line(failures_path, doc_url)
            time.sleep(max(args.delay, 0.5))
            continue

        title, pdf_link = extract_title_and_pdf_link(r.text, doc_url)
        base = f"{doc_id}"
        if title:
            base += "_" + sanitize_filename(title)
        filename = base + ".pdf"
        out_path = outdir / filename

        # If file already exists, consider it processed and log (idempotent)
        if out_path.exists():
            append_line(processed_path, doc_url)
            processed.add(doc_url)
            count_saved += 1
            print(f"[ok] exists (logged): {out_path.name}")
            time.sleep(max(args.delay, 0.5))
            continue

        # Try explicit PDF link first, then guesses
        candidates = [pdf_link] if pdf_link else []
        candidates += guess_pdf_url(doc_url)

        saved = False
        for cand in candidates:
            if not cand:
                continue
            if download_pdf(session, cand, out_path):
                saved = True
                break

        if saved:
            append_line(processed_path, doc_url)  # record success
            processed.add(doc_url)
            count_saved += 1
            print(f"[ok] saved: {out_path.name}")
        else:
            print(f"[fail] could not download PDF for {doc_url}", file=sys.stderr)
            if failures_path:
                append_line(failures_path, doc_url)

        count_total += 1
        time.sleep(max(args.delay, 0.5))

    print(f"\n[done] total considered: {len(all_urls)}")
    print(f"[done] skipped (already processed): {count_skipped}")
    print(f"[done] newly saved or confirmed existing: {count_saved}")
    print(f"[where] PDFs: {outdir.resolve()}")
    print(f"[where] processed log: {processed_path.resolve()}")
    if failures_path:
        print(f"[where] failures log: {failures_path.resolve()}")

if __name__ == "__main__":
    main()
