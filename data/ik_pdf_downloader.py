#!/usr/bin/env python3
"""
ik_pdf_downloader.py

Given a list of Indian Kanoon /doc/<id>/ URLs, fetch the "Get this document in PDF"
link for each and download the PDF.

Usage:
  python ik_pdf_downloader.py --in links.txt --outdir ./pdfs --delay 1.5 --max 200

Notes / ethics:
- Be gentle: keep delays >= 1s; don't parallelize without permission.
- Use for research; respect robots/ToS and any takedown requests.
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
    "User-Agent": "IK-ResearchBot/1.0 (+contact: your-email@example.com)"
}

DOC_ID_RE = re.compile(r"/doc/(\d+)/?")

def sanitize_filename(s: str, maxlen: int = 140) -> str:
    s = re.sub(r"[^\w\-\.\(\)\[\] ]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > maxlen:
        s = s[:maxlen].rstrip()
    return s or "document"

def guess_pdf_url(doc_url: str) -> list[str]:
    # Some pages accept query params for PDF; keep as last resort.
    # We'll try a few reasonable guesses.
    candidates = [
        doc_url.rstrip("/") + "/?format=pdf",
        doc_url.rstrip("/") + "/?type=pdf",
        doc_url.rstrip("/") + "/?download=1",
        doc_url.rstrip("/") + "/?pdf=1",
    ]
    return candidates

def extract_title_and_pdf_link(html: str, page_url: str) -> tuple[str | None, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    # Try to get a human-friendly title
    title = None
    # Case title often in the first <h2> or centered bold text near the top
    h2 = soup.find("h2")
    if h2 and h2.get_text(strip=True):
        title = h2.get_text(strip=True)
    if not title:
        # Fallback to document title
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

    # Find the "Get this document in PDF" anchor by text
    pdf_url = None
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True).lower()
        href = a["href"]
        if "get this document in pdf" in text or "download pdf" in text or href.lower().endswith(".pdf"):
            pdf_url = urljoin(page_url, href)
            break

    # If still not found, try anchors that mention 'pdf' in href
    if not pdf_url:
        for a in soup.select('a[href*="pdf"]'):
            pdf_url = urljoin(page_url, a["href"])
            break

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
            # Basic content-type sanity check (not all servers set this consistently)
            ctype = r.headers.get("Content-Type", "")
            if "pdf" not in ctype.lower():
                # Sometimes the endpoint returns a PDF with text/html; allow if bytes start with %PDF
                peek = r.raw.read(5, decode_content=True)
                if peek != b"%PDF-":
                    print(f"[warn] Not a PDF (ctype={ctype}) from {pdf_url}", file=sys.stderr)
                    return False
                # Put the peek back into the stream for writing
                chunk_iter = iter([peek] + list(r.iter_content(chunk_size=8192)))
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

def main():
    ap = argparse.ArgumentParser(description="Download PDFs from Indian Kanoon /doc/ pages.")
    ap.add_argument("--in", dest="infile", required=True, help="Text file with /doc/ URLs (one per line).")
    ap.add_argument("--outdir", default="pdfs", help="Directory to save PDFs.")
    ap.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds).")
    ap.add_argument("--max", type=int, default=0, help="Max documents to process (0 = no limit).")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    with open(args.infile, "r", encoding="utf-8") as f:
        urls = [ln.strip() for ln in f if ln.strip()]

    if args.max > 0:
        urls = urls[:args.max]

    session = requests.Session()
    session.headers.update(HEADERS)

    done = 0
    for i, doc_url in enumerate(urls, 1):
        # Normalize to absolute and ensure it looks like a /doc/ page
        if not doc_url.startswith("http"):
            doc_url = urljoin(BASE, doc_url)
        if "/doc/" not in urlparse(doc_url).path:
            print(f"[skip] Not a /doc/ URL: {doc_url}", file=sys.stderr)
            continue

        # Derive ID for filename fallback
        m = DOC_ID_RE.search(doc_url)
        doc_id = m.group(1) if m else f"doc_{i}"

        # Fetch page
        r = fetch(session, doc_url)
        if not r:
            time.sleep(max(args.delay, 0.5))
            continue

        title, pdf_url = extract_title_and_pdf_link(r.text, doc_url)
        # Build filename
        base = f"{doc_id}"
        if title:
            base += "_" + sanitize_filename(title)
        filename = base + ".pdf"
        out_path = outdir / filename

        if out_path.exists():
            print(f"[skip] exists: {out_path.name}")
            time.sleep(max(args.delay, 0.5))
            continue

        # If page didn’t expose a PDF link, try guesses
        candidates = [pdf_url] if pdf_url else []
        candidates += guess_pdf_url(doc_url)

        got = False
        for cand in candidates:
            if not cand:
                continue
            if download_pdf(session, cand, out_path):
                print(f"[ok] {out_path.name}")
                got = True
                break

        if not got:
            print(f"[fail] could not find/download PDF for {doc_url}", file=sys.stderr)

        done += 1
        time.sleep(max(args.delay, 0.5))

    print(f"[done] processed {done} URLs; PDFs saved to {outdir.resolve()}")

if __name__ == "__main__":
    main()
