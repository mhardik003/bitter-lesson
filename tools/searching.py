import os
from typing import List, Dict, Any, Optional
import requests
import urllib.parse
import dotenv

dotenv.load_dotenv()
INDEXER_BASE_URL = f"http://localhost:{os.getenv('BACKEND_PORT') or 8000}"


def search_bm25(qtext: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Keyword search over doc level metadata.
    Returns doc_id and extracted metadata (includes case_title, decision_date,
    statutes_mentioned, court, parties, topics, summary, sections, pdf_url, pdf_path).
    """
    resp = requests.post(
        f"{INDEXER_BASE_URL}/search/bm25",
        json={"query": qtext, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("hits", [])


def search_faiss(qtext: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Semantic search over document chunks.
    Returns doc_id and pages and chunk text.
    """
    resp = requests.post(
        f"{INDEXER_BASE_URL}/search/faiss",
        json={"query": qtext, "top_k": top_k},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("hits", [])


def get_doc_metadata(doc_id: str) -> Dict[str, Any]:
    """Get document metadata (includes case_title, decision_date,
    statutes_mentioned, court, parties, topics, summary, sections, pdf_url and pdf_path) given a doc_id."""
    resp = requests.post(
        f"{INDEXER_BASE_URL}/doc/metadata",
        json={"doc_id": doc_id},
        timeout=30,
    )
    resp.raise_for_status()

    meta = resp.json()

    pdf_path = meta.get("pdf_path")
    if pdf_path:
        encoded = urllib.parse.quote(pdf_path, safe="/")
        meta["pdf_url"] = f"{INDEXER_BASE_URL}/pdf?path={encoded}"
    return meta


def get_doc_content(doc_id: str, pages: Optional[List[int]] = None) -> str:
    """
    Load markdown for doc_id via backend.
    If pages is given (recommended), backend will return only those pages.
    """
    resp = requests.post(
        f"{INDEXER_BASE_URL}/doc/content",
        json={"doc_id": doc_id, "pages": pages},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("content", "")
