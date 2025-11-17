import time
from typing import List, Dict, Any, Optional

from flask import Flask, request, jsonify, abort, send_from_directory
import os

from index import LegalIndexer  # your LegalIndexer class

DB_PATH = "/scratch/akshit.kumar/indices/legal_index.db"
FAISS_PATH = "/scratch/akshit.kumar/indices/faiss.index"

app = Flask(__name__)

import dotenv

dotenv.load_dotenv()

indexer = LegalIndexer.load_from_existing(DB_PATH, FAISS_PATH)
print("FAISS vectors:", indexer.faiss_index.ntotal)
print("Chunks:", len(indexer.chunk_metadata))
print("BM25 docs:", len(indexer.bm25_doc_ids))

PDF_ROOT = "/scratch/akshit.kumar/pdfs/sources"


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
