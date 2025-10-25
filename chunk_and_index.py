
import os
import re
import json
import math
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import regex  # better regex engine
from tqdm import tqdm

# Tokenizer and embeddings
from transformers import AutoTokenizer, AutoModel
import torch

# BM25
from rank_bm25 import BM25Okapi

# FAISS
import faiss
import numpy as np

# -------------------------
# Helpers
# -------------------------

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")

def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-").lower()
    return s or "doc"

def sha256_text(s: str) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()

# -------------------------
# DeepSeek-specific cleanup
# -------------------------

REF_TAG_RE = re.compile(r"<\|ref\|\>.*?<\|/ref\|\>", re.S)
DET_TAG_RE = re.compile(r"<\|det\|\>.*?<\|/det\|\>", re.S)
EMPTY_TD_RE = re.compile(r"\s*<\s*td\s*>\s*</\s*td\s*>\s*", re.I)
TD_BLOCK_RE = re.compile(r"(?:\s*<\s*td\s*>\s*</\s*td\s*>\s*){2,}", re.I)

def remove_deepseek_tags(md: str) -> str:
    # Strip grounding and detection tags entirely
    md = REF_TAG_RE.sub("", md)
    md = DET_TAG_RE.sub("", md)
    return md

def remove_empty_tds(md: str) -> str:
    # Remove runs of empty table cells first, then any isolated empty cell
    md = TD_BLOCK_RE.sub("", md)
    md = EMPTY_TD_RE.sub("", md)
    return md

def collapse_punctuation(md: str) -> str:
    # Replace sequences like ". ." ".  ." "… .." with a single period and space
    md = re.sub(r"(?:\s*\.\s*){2,}", ". ", md)
    # Collapse repeated punctuation (commas, semicolons) conservatively
    md = re.sub(r"([,;:])\s*(\1\s*){1,}", r"\1 ", md)
    # Normalize whitespace-only lines
    md = re.sub(r"[ \t]+\n", "\n", md)
    # Collapse >2 blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md

def deepseek_clean(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = remove_deepseek_tags(md)
    md = remove_empty_tds(md)
    md = collapse_punctuation(md)
    return md

# -------------------------
# Markdown parsing for legal docs
# -------------------------

HEADING_RE = re.compile(r'^(#{1,6})\s+(.*)$')
SECTION_LINE_RE = re.compile(r'^\s*(?:section|sec\.?)\s+([0-9A-Za-z]+)\b[.:]?\s*(.*)$', re.I)
SECTION_NUM_TITLE_RE = re.compile(r'^\s*([0-9A-Za-z]+)\.\s+(.+)$')
PARA_SPLIT_RE = re.compile(r'\n\s*\n+', re.M)

def extract_sections_from_markdown(md_text: str) -> List[Dict]:
    """
    Extract sections from markdown.
    We first track chapter headings via '#' or '##' but sections are detected by:
      - 'Section 74 ...' style lines, OR
      - '74. Title ...' style lines
    Everything until the next section line or chapter heading belongs to that section.
    Returns a list of dicts: {section_id, heading, text_original, chapter}
    """
    lines = md_text.splitlines()
    chapters = []  # (idx, level, title)
    for i, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            chapters.append((i, level, title))

    chapter_starts = set(i for i, _, _ in chapters)

    section_starts = []
    for i, line in enumerate(lines):
        if i in chapter_starts:
            continue
        m1 = SECTION_LINE_RE.match(line)
        m2 = SECTION_NUM_TITLE_RE.match(line)
        if m1:
            sec_id = m1.group(1)
            title = m1.group(2).strip() or f"Section {sec_id}"
            section_starts.append((i, f"s{sec_id}", title))
        elif m2:
            sec_id = m2.group(1)
            title = m2.group(2).strip()
            section_starts.append((i, f"s{sec_id}", title))

    section_starts.sort(key=lambda x: x[0])
    sections = []
    for idx, (start_i, sec_id, title) in enumerate(section_starts):
        end_i = len(lines)
        if idx + 1 < len(section_starts):
            end_i = section_starts[idx + 1][0]
        later_chapters = [ci for ci, _, _ in chapters if ci > start_i and ci < end_i]
        if later_chapters:
            end_i = min(end_i, later_chapters[0])

        chapter_title = None
        prev_chaps = [t for (ci, _, t) in chapters if ci < start_i]
        if prev_chaps:
            chapter_title = prev_chaps[-1]

        body = "\n".join(lines[start_i:end_i]).strip()
        sections.append({
            "section_id": sec_id,
            "heading": title,
            "chapter": chapter_title,
            "text_original": body
        })
    return sections

def normalize_for_index(text: str) -> str:
    # Conservative normalization for indexing
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = regex.sub(r"\n{3,}", "\n\n", t)
    t = regex.sub(r"[ \t]+\n", "\n", t)
    t = t.lower()
    # Normalize dash variants to a simple hyphen
    t = t.replace("—", "-").replace("–", "-")
    # Remove empty td residues if any slipped through
    t = EMPTY_TD_RE.sub(" ", t)
    # Collapse stray ".." again defensively
    t = re.sub(r"(?:\s*\.\s*){2,}", ". ", t)
    return t

# -------------------------
# Sentence segmentation and packing
# -------------------------

ABBR = set(["e.g.", "i.e.", "mr.", "mrs.", "dr.", "vs.", "v.", "no.", "art.", "sec.", "secs.", "s.", "u.s.", "etc."])

def split_sentences(text: str) -> List[str]:
    parts = regex.split(r"(?<=[.?!])\s+(?=[A-Z(])", text)
    sents = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if sents:
            prev = sents[-1]
            tail = prev.split()[-1].lower() if prev.split() else ""
            if tail in ABBR:
                sents[-1] = prev + " " + p
                continue
        sents.append(p)
    return sents

def pack_sentences_to_windows(sents: List[str], tokenizer, target_tokens: int) -> List[str]:
    windows = []
    buf = []
    buf_tok = 0
    for s in sents:
        toks = tokenizer.encode(s, add_special_tokens=False)
        if buf_tok + len(toks) > target_tokens and buf:
            windows.append(" ".join(buf).strip())
            buf = [s]
            buf_tok = len(toks)
        else:
            buf.append(s)
            buf_tok += len(toks)
    if buf:
        windows.append(" ".join(buf).strip())
    return windows

# -------------------------
# Chunking
# -------------------------

def chunk_sections(sections: List[Dict], doc_id: str, tokenizer, granularities=(2048,512,128)) -> List[Dict]:
    chunks = []
    sec_ids = [s["section_id"] for s in sections]
    for si, sec in enumerate(sections):
        sec_id = sec["section_id"]
        # Apply DeepSeek cleanup at section level too, to be safe
        text_orig = deepseek_clean(sec["text_original"])
        sents = split_sentences(text_orig)
        for g in granularities:
            windows = pack_sentences_to_windows(sents, tokenizer, g)
            if not windows:
                continue
            for wi, wtxt in enumerate(windows):
                offset = wi
                chunk_id = f"{doc_id}::sec/{sec_id}::g={g}::o={offset}"
                neighbors = []
                if si > 0:
                    neighbors.append(f"{doc_id}::sec/{sec_ids[si-1]}::g={g}::o=0")
                if si + 1 < len(sec_ids):
                    neighbors.append(f"{doc_id}::sec/{sec_ids[si+1]}::g={g}::o=0")
                chunk = {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "scope": {"type": "statute_section", "section_id": sec_id},
                    "parents": [f"{doc_id}::sec/{sec_id}"],
                    "neighbors": neighbors,
                    "text_original": wtxt,  # cleaned original
                    "text_norm": normalize_for_index(wtxt),
                    "topics": [],
                    "sha256_chunk": sha256_text(wtxt)
                }
                # Drop chunks that became empty after cleanup
                if chunk["text_norm"].strip():
                    chunks.append(chunk)
    return chunks

# -------------------------
# Indexing
# -------------------------

def build_bm25(chunks: List[Dict], out_dir: Path, granularity: int = 512):
    docs = []
    ids = []
    for c in chunks:
        if f"g={granularity}" in c["chunk_id"]:
            tokens = c["text_norm"].split()
            if tokens:
                docs.append(tokens)
                ids.append(c["chunk_id"])
    bm25 = BM25Okapi(docs)
    import pickle
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"bm25_g{granularity}.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "ids": ids}, f)
    write_text(out_dir / f"bm25_g{granularity}.ids", "\n".join(ids))

def build_faiss(chunks: List[Dict], model_name: str, out_dir: Path, granularity: int = 512, device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model = model.eval().to(device)
    def encode(texts: List[str], batch_size: int = 16) -> np.ndarray:
        embs = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="encode"):
                batch = texts[i:i+batch_size]
                tok = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                tok = {k: v.to(device) for k,v in tok.items()}
                out = model(**tok)
                if hasattr(out, "last_hidden_state"):
                    cls = out.last_hidden_state[:, 0, :]
                else:
                    cls = out[0][:, 0, :]
                embs.append(cls.detach().cpu().numpy())
        return np.vstack(embs).astype("float32")

    texts = []
    ids = []
    for c in chunks:
        if f"g={granularity}" in c["chunk_id"]:
            texts.append(c["text_norm"])
            ids.append(c["chunk_id"])
    if not texts:
        print(f"No chunks found for g={granularity}")
        return

    vecs = encode(texts, batch_size=16)
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(vecs)
    index.add(vecs)

    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / f"faiss_{slugify(model_name)}_g{granularity}.index"))
    write_text(out_dir / f"faiss_{slugify(model_name)}_g{granularity}.ids", "\n".join(ids))

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md_dir", default="data_md/md", help="Directory with Markdown files from OCR (.md or .mmd)")
    ap.add_argument("--out_dir", default="data_out", help="Output directory for chunks and indexes")
    ap.add_argument("--granularities", default="2048,512,128", help="Comma-separated token sizes")
    ap.add_argument("--bm25_granularity", type=int, default=512, help="Granularity to BM25 index")
    ap.add_argument("--faiss_granularity", type=int, default=512, help="Granularity to vector index")
    ap.add_argument("--vector_model", default="sentence-transformers/all-mpnet-base-v2", help="HF model id for vector embeddings")
    ap.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    ap.add_argument("--write_cleaned_md", action="store_true", help="Also save cleaned markdown next to source for auditing")
    args = ap.parse_args()

    md_dir = Path(args.md_dir)
    out_dir = Path(args.out_dir)
    chunks_path = out_dir / "chunks.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    granularities = tuple(int(x) for x in args.granularities.split(","))

    # Tokenizer for chunking
    chunk_tokenizer = AutoTokenizer.from_pretrained(args.vector_model, trust_remote_code=True)

    # Collect markdown files
    md_files = sorted([p for p in md_dir.glob("**/*") if p.suffix.lower() in (".md", ".mmd")])
    if not md_files:
        print(f"No markdown files found under {md_dir}")
        return

    all_chunks = []
    for md_path in tqdm(md_files, desc="parse+chunk"):
        raw_md = read_text(md_path)
        cleaned_md = deepseek_clean(raw_md)
        if args.write_cleaned_md:
            audit_path = md_path.with_suffix(md_path.suffix + ".clean.md")
            write_text(audit_path, cleaned_md)

        # doc_id based on cleaned content hash and name
        doc_hash = sha256_text(cleaned_md)[:8]
        doc_id = f"{slugify(md_path.stem)}_{doc_hash}@v1"

        sections = extract_sections_from_markdown(cleaned_md)
        if not sections:
            sections = [{
                "section_id": "s1",
                "heading": "Document",
                "chapter": None,
                "text_original": cleaned_md
            }]
        chunks = chunk_sections(sections, doc_id, chunk_tokenizer, granularities=granularities)
        all_chunks.extend(chunks)

    # Write chunks.jsonl
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Wrote {len(all_chunks)} chunks to {chunks_path}")

    # Build BM25
    build_bm25(all_chunks, out_dir / "index/bm25", granularity=args.bm25_granularity)
    print("BM25 index built")

    # Build FAISS for vector_model
    build_faiss(all_chunks, args.vector_model, out_dir / "index/vector", granularity=args.faiss_granularity, device=args.device)
    print("FAISS index built")

if __name__ == "__main__":
    main()
