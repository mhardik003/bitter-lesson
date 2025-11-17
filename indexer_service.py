import dotenv
import time
from typing import List, Dict, Any, Optional

import subprocess
import shutil
from pathlib import Path
import tempfile
import logging

from flask import Flask, request, jsonify, abort, send_from_directory
import os

from index import LegalIndexer

DB_PATH = "/scratch/akshit.kumar/indices/legal_index.db"
FAISS_PATH = "/scratch/akshit.kumar/indices/faiss.index"

app = Flask(__name__)
app.logger.setLevel(logging.INFO)


dotenv.load_dotenv()

indexer = LegalIndexer.load_from_existing(DB_PATH, FAISS_PATH)
print("FAISS vectors:", indexer.faiss_index.ntotal)
print("Chunks:", len(indexer.chunk_metadata))
print("BM25 docs:", len(indexer.bm25_doc_ids))

PDF_ROOT = "/scratch/akshit.kumar/pdfs/sources"

GENERATED_OUTPUT_ROOT = "/scratch/akshit.kumar/generated"
os.makedirs(GENERATED_OUTPUT_ROOT, exist_ok=True)

PDFLATEX_BIN = os.getenv("PDFLATEX_BIN")
PANDOC_BIN = os.getenv("PANDOC_BIN")


def _safe_basename(basename: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in basename)
    if not safe:
        safe = f"doc_{int(time.time())}"
    return safe[:64]


def run_pandoc(markdown_text: str, basename: str, fmt: str) -> str:
    """
    Run pandoc to convert markdown_text to the given format.
    Returns absolute path to the generated file.

    fmt can be: 'pdf', 'latex', 'docx', 'pptx', 'html', 'beamer-pdf', etc.
    """
    safe = _safe_basename(basename)
    tempdir = tempfile.mkdtemp(prefix="pandoc_", dir="/tmp")
    md_path = Path(tempdir) / f"{safe}.md"
    md_path.write_text(markdown_text)

    # Decide target format and output extension
    if fmt == "pdf":
        target = "pdf"
        ext = "pdf"
        args = ["-t", "pdf", f"--pdf-engine={PDFLATEX_BIN}"]
    elif fmt == "latex":
        target = "latex"
        ext = "tex"
        args = ["-t", "latex"]
    elif fmt == "docx":
        target = "docx"
        ext = "docx"
        args = ["-t", "docx"]
    elif fmt == "pptx":
        target = "pptx"
        ext = "pptx"
        args = ["-t", "pptx"]
    elif fmt == "html":
        target = "html"
        ext = "html"
        args = ["-t", "html"]
    elif fmt == "beamer-pdf":
        # pandoc markdown -> beamer -> pdf
        target = "beamer"
        ext = "pdf"
        args = ["-t", "beamer", f"--pdf-engine={PDFLATEX_BIN}"]
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    out_path = Path(GENERATED_OUTPUT_ROOT) / f"{safe}.{ext}"

    cmd = [PANDOC_BIN, str(md_path), "-o", str(out_path)] + args

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

    return str(out_path)


@app.route("/generated/<path:filename>")
def serve_generated(filename: str):
    return send_from_directory(GENERATED_OUTPUT_ROOT, filename)


@app.route("/render", methods=["POST"])
def render():
    """
    JSON:
      {
        "markdown": "...",
        "format": "pdf" | "latex" | "docx" | "pptx" | "html" | "beamer-pdf",
        "basename": "optional_name"
      }
    """
    data = request.get_json(force=True) or {}
    markdown_text = data.get("markdown")
    fmt = data.get("format")
    basename = data.get("basename") or f"case_study_{int(time.time())}"

    if not markdown_text:
        return jsonify({"error": "Missing 'markdown'"}), 400
    if not fmt:
        return jsonify({"error": "Missing 'format'"}), 400

    try:
        out_path = run_pandoc(markdown_text, basename, fmt)
    except Exception as e:
        return jsonify(
            {
                "error": "render_failed",
                "detail": str(e),
            }
        ), 500

    filename = os.path.basename(out_path)
    base = request.url_root.rstrip("/")
    url = f"{base}/generated/{filename}"

    return jsonify(
        {
            "path": out_path,
            "url": url,
            "format": fmt,
        }
    )


@app.route("/pdf")
def serve_pdf():
    pdf_path = request.args.get("path")  # full path string from DB or metadata
    if not pdf_path:
        abort(400)

    # Safety: ensure it starts with the known prefix
    if not pdf_path.startswith(PDF_ROOT):
        abort(403)

    filename = os.path.basename(pdf_path)  # e.g. "129235580_Delhi High Court.pdf"
    return send_from_directory(PDF_ROOT, filename)


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "faiss_vectors": indexer.faiss_index.ntotal,
            "chunks": len(indexer.chunk_metadata),
            "bm25_docs": len(indexer.bm25_doc_ids),
        }
    )


@app.route("/search/bm25", methods=["POST"])
def search_bm25() -> Any:
    """Keyword search over doc level metadata."""
    data = request.get_json(force=True) or {}
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    top_k = int(data.get("top_k", 20))
    hits = indexer.search_bm25(query, top_k=top_k)

    # Shape: { "hits": [ {doc_id, score, md_path, pdf_path, metadata: {...}}, ... ] }
    return jsonify({"hits": hits})


@app.route("/search/faiss", methods=["POST"])
def search_faiss() -> Any:
    """Semantic search over chunks."""
    data = request.get_json(force=True) or {}
    query = data.get("query")
    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    top_k = int(data.get("top_k", 20))
    hits = indexer.search_faiss(query, top_k=top_k)

    # hits already include doc_id, pages, text, score, etc
    return jsonify({"hits": hits})


@app.route("/doc/metadata", methods=["POST"])
def doc_metadata() -> Any:
    """Get document metadata by doc_id."""
    data = request.get_json(force=True) or {}
    doc_id = data.get("doc_id")
    if not doc_id:
        return jsonify({"error": "Missing 'doc_id'"}), 400

    meta = indexer.get_doc_metadata(doc_id)
    return jsonify(meta)


@app.route("/doc/content", methods=["POST"])
def doc_content() -> Any:
    """Get markdown content by doc_id and optional pages."""
    data = request.get_json(force=True) or {}
    doc_id = data.get("doc_id")
    if not doc_id:
        return jsonify({"error": "Missing 'doc_id'"}), 400

    pages = data.get("pages")  # optional list of ints
    content = indexer.get_doc_content(doc_id, pages=pages)
    return jsonify({"doc_id": doc_id, "pages": pages, "content": content})


if __name__ == "__main__":
    # For dev. In prod: gunicorn or similar.
    app.run(host="0.0.0.0", port=os.getenv("BACKEND_PORT"), debug=False)
