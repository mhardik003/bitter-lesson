import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from PIL import Image
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

from pdf2image import convert_from_path
import hashlib

# ------------------------ utils ------------------------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-").lower()
    return s or "doc"

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def normalize_markdown(md: str) -> str:
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"[ \t]+\n", "\n", md)
    return md

def list_inputs(input_root: Path, exts=(".pdf", ".png", ".jpg", ".jpeg", ".tiff")) -> List[Path]:
    files = []
    for p in sorted(input_root.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files

def load_done_hashes(corpus_path: Path) -> Dict[str, dict]:
    done = {}
    if corpus_path.exists():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except Exception:
                continue
            h = row.get("sha256_file")
            sp = row.get("source_path")
            if h:
                done[h] = row
            if sp and sp not in done:
                # also mark by absolute path string
                done[sp] = row
    return done

# ------------------------ pdf to images ------------------------

def pdf_to_images(pdf_path: Path, tmp_dir: Path, dpi: int = 300) -> List[Path]:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        paths = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt="png",
            output_folder=str(tmp_dir),
            output_file=slugify(pdf_path.stem),
            paths_only=True,
            thread_count=2,
            use_pdftocairo=True
        )
        return [Path(p) for p in paths]
    except Exception as e:
        print(f"[pdf2image] Failed for {pdf_path}: {e}", file=sys.stderr)
        return []

# ------------------------ batching ------------------------

def chunked(it, n):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf

# ------------------------ main pipeline ------------------------

def process_docs_vllm(
    model_id: str,
    input_root: Path,
    out_root: Path,
    tmp_root: Path,
    dpi: int = 300,
    batch_pages: int = 16,
    prompt: str = "<image>\n<|grounding|>Convert the document to markdown.",
    ngram_size: int = 30,
    window_size: int = 90,
    whitelist_token_ids: Optional[str] = None,
    max_tokens: int = 8192,
    temperature: float = 0.0,
    skip_existing: bool = True,
):
    md_dir = out_root / "md"
    work_dir = out_root / "work"
    md_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_root / "corpus.jsonl"

    # Load already processed file hashes/paths to skip
    done_map = load_done_hashes(corpus_path) if skip_existing else {}

    sampling_param = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        extra_args=dict(
            ngram_size=ngram_size,
            window_size=window_size,
            whitelist_token_ids=set(int(x) for x in whitelist_token_ids.split(",")) if whitelist_token_ids else None,
        ),
        skip_special_tokens=False,
    )
    llm = LLM(
        model=model_id,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor]
    )

    files = list_inputs(input_root)
    if not files:
        print(f"[info] No inputs under {input_root}")
        return

    for fidx, src_path in enumerate(files, start=1):
        # Compute file sha BEFORE any rasterization, so we can skip early
        try:
            src_bytes = src_path.read_bytes()
        except Exception as e:
            print(f"[warn] cannot read {src_path}: {e}", file=sys.stderr)
            continue
        file_sha = sha256_bytes(src_bytes)

        if skip_existing and (file_sha in done_map or str(src_path) in done_map):
            print(f"[skip] already processed: {src_path}")
            continue

        print(f"[{fidx}/{len(files)}] {src_path}")
        doc_slug = slugify(src_path.stem)
        doc_work = work_dir / doc_slug
        page_dir_root = doc_work / "pages"
        page_dir_root.mkdir(parents=True, exist_ok=True)

        # Gather pages
        page_images = []
        if src_path.suffix.lower() == ".pdf":
            page_pngs = pdf_to_images(src_path, tmp_root / doc_slug, dpi=dpi)
            for i, png in enumerate(page_pngs, start=1):
                try:
                    im = Image.open(png).convert("RGB")
                except Exception as e:
                    print(f"[warn] cannot open {png}: {e}", file=sys.stderr)
                    continue
                page_images.append((i, png, im))
        else:
            try:
                im = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"[warn] cannot open {src_path}: {e}", file=sys.stderr)
                continue
            page_images.append((1, src_path, im))

        if not page_images:
            print(f"[skip] no pages for {src_path}")
            continue

        # vLLM batched OCR
        md_parts = {}
        for batch in chunked(page_images, batch_pages):
            model_input = [{"prompt": prompt, "multi_modal_data": {"image": im}} for (_i, _p, im) in batch]
            outputs = llm.generate(model_input, sampling_param)
            for (i, png_path, _im), out in zip(batch, outputs):
                text = out.outputs[0].text if out.outputs else ""
                text = normalize_markdown(text)
                page_dir = page_dir_root / f"page_{i:04d}"
                write_text(page_dir / "page.mmd", text)
                md_parts[i] = text

        # Merge
        merged = [f"# {src_path.name}", ""]
        for i in sorted(md_parts.keys()):
            merged.append(f"<<<PAGE {i}>>>")
            merged.append(md_parts[i])
            merged.append("")
        merged_md = "\n".join(merged).strip() + "\n"

        md_path = md_dir / f"{doc_slug}.md"
        write_text(md_path, merged_md)

        # Append to corpus.jsonl
        entry = {
            "doc_id": f"{doc_slug}_{sha256_text(merged_md)[:8]}@v1",
            "title": src_path.stem,
            "doc_type": "unknown",
            "jurisdiction": "IN",
            "source_path": str(src_path),
            "output_markdown": str(md_path),
            "ocr_engine": "deepseek-ai/DeepSeek-OCR (vLLM)",
            "dpi": dpi,
            "sha256_file": file_sha,
            "sha256_markdown": sha256_text(merged_md),
        }
        with (out_root / "corpus.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Batched DeepSeek-OCR via vLLM over PDFs/images")
    ap.add_argument("--input", default="../data", help="Input dir (run from ocr/ so this points outside)")
    ap.add_argument("--out", default="../data_md", help="Output dir (outside ocr/)")
    ap.add_argument("--tmp", default="../data_tmp", help="Temp dir for rasterized pages")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--batch_pages", type=int, default=16)
    ap.add_argument("--prompt", default="<image>\\n<|grounding|>Convert the document to markdown.")
    ap.add_argument("--model_id", default="deepseek-ai/DeepSeek-OCR")
    ap.add_argument("--ngram_size", type=int, default=30)
    ap.add_argument("--window_size", type=int, default=90)
    ap.add_argument("--whitelist_token_ids", type=str, default="")
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no_skip_existing", action="store_true", help="Process even if file is in corpus.jsonl")
    args = ap.parse_args()

    process_docs_vllm(
        model_id=args.model_id,
        input_root=Path(args.input).resolve(),
        out_root=Path(args.out).resolve(),
        tmp_root=Path(args.tmp).resolve(),
        dpi=args.dpi,
        batch_pages=args.batch_pages,
        prompt=args.prompt,
        ngram_size=args.ngram_size,
        window_size=args.window_size,
        whitelist_token_ids=args.whitelist_token_ids if args.whitelist_token_ids else None,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        skip_existing=not args.no_skip_existing,
    )

if __name__ == "__main__":
    main()
