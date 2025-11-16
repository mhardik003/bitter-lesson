import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from chunk import (
    chunk_md_with_granularities,
    deepseek_clean,
    _split_into_pages,
)


@dataclass
class ChunkRecord:
    vector_id: int  # index in FAISS
    chunk_uid: str  # doc_id::g{granularity}::c{local_idx}
    doc_id: str
    md_path: str
    pages: List[int]
    granularity: int
    text: str


class LegalIndexer:
    """
    Handles:
      - Chunking markdown and indexing into FAISS
      - Building BM25 index over extracted_meta
      - Persisting doc and chunk metadata in SQLite
    """

    def __init__(
        self,
        db_path: str = "legal_index.db",
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        # SQLite
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

        # Embedding model
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # FAISS over L2 normalized vectors
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)

        # In memory mirror of chunk metadata for fast lookup
        self.chunk_metadata: List[ChunkRecord] = []

        # BM25 state
        self.bm25_corpus_tokens: List[List[str]] = []
        self.bm25_doc_ids: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    # --------------- DB init ---------------

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS docs (
                doc_id TEXT PRIMARY KEY,
                md_path TEXT,
                pdf_path TEXT,
                extracted_meta_json TEXT,
                raw_output TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                vector_id INTEGER PRIMARY KEY,
                chunk_uid TEXT,
                doc_id TEXT,
                md_path TEXT,
                pages_json TEXT,
                granularity INTEGER,
                chunk_text TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        self.conn.commit()

    # --------------- BM25 helpers ---------------

    @staticmethod
    def _build_bm25_text(meta: Dict[str, Any]) -> str:
        """
        Turn extracted_meta into a single text string for BM25.
        Includes parties.
        """
        parts: List[str] = []

        case_title = meta.get("case_title")
        if case_title:
            parts.append(case_title)

        doc_type = meta.get("doc_type")
        if doc_type:
            parts.append(doc_type)

        decision_date_human = meta.get("decision_date_human") or meta.get(
            "decision_date"
        )
        if decision_date_human:
            parts.append(str(decision_date_human))

        court = meta.get("court")
        if court:
            parts.append(court)

        jurisdiction = meta.get("jurisdiction")
        if jurisdiction:
            parts.append(jurisdiction)

        statutes = meta.get("statutes_mentioned") or []
        parts.extend(statutes)

        topics = meta.get("topics") or []
        parts.extend(topics)

        for sec in meta.get("sections", []):
            heading = sec.get("heading")
            summary = sec.get("summary")
            if heading:
                parts.append(heading)
            if summary:
                parts.append(summary)

        # Parties
        for party in meta.get("parties", []):
            name = party.get("name")
            role = party.get("role")
            if name and role:
                parts.append(f"{name} {role}")
            elif name:
                parts.append(name)

        if not parts:
            raw = meta.get("raw_output")
            if raw:
                parts.append(raw)

        return " \n".join(parts)

    @staticmethod
    def _tokenize_for_bm25(text: str) -> List[str]:
        return text.lower().split()

    # --------------- indexing ---------------

    def index_jsonl(
        self,
        jsonl_path: str,
        token_window_sizes: Tuple[int, ...] = (384, 256, 128),
        min_tokens: int = 32,
        encoding: str = "utf-8",
    ) -> None:
        """
        Stream over a jsonl of documents and index everything.
        Each line is one record like your example.
        """

        jsonl_path = str(jsonl_path)
        total = sum(1 for _ in open(jsonl_path, "r", encoding=encoding))
        with open(jsonl_path, "r", encoding=encoding) as f:
            for line in tqdm(f, total=total, desc="Indexing documents"):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                self._index_one_record(
                    rec,
                    token_window_sizes=token_window_sizes,
                    min_tokens=min_tokens,
                )

        self._finalize_bm25()
        self.conn.commit()

    def _index_one_record(
        self,
        rec: Dict[str, Any],
        token_window_sizes: Tuple[int, ...],
        min_tokens: int,
    ) -> None:
        doc_id: str = rec["doc_id"]
        md_path: str = rec["md_path"]
        pdf_path: str = rec.get("pdf_path", "")

        # Persist doc level metadata
        extracted_meta = rec.get("extracted_meta")
        extracted_meta_json = json.dumps(extracted_meta) if extracted_meta else None
        raw_output = rec.get("raw_output")

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO docs (doc_id, md_path, pdf_path, extracted_meta_json, raw_output)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                md_path = excluded.md_path,
                pdf_path = excluded.pdf_path,
                extracted_meta_json = excluded.extracted_meta_json,
                raw_output = excluded.raw_output
            """,
            (doc_id, md_path, pdf_path, extracted_meta_json, raw_output),
        )

        # Drop old chunks for this doc_id, if any
        cur.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))

        # Read md and chunk
        md_text = Path(md_path).read_text(encoding="utf-8", errors="ignore")

        chunk_map = chunk_md_with_granularities(
            md_text,
            tokenizer=None,
            token_window_sizes=list(token_window_sizes),
            min_tokens=min_tokens,
        )

        all_texts: List[str] = []
        all_metadata: List[
            Tuple[str, List[int], int, str]
        ] = []  # (chunk_uid, pages, granularity, text)

        for granularity, chunks in chunk_map.items():
            for local_idx, ch in enumerate(chunks):
                chunk_uid = f"{doc_id}::g{granularity}::c{local_idx}"
                pages = ch.get("pages", [])
                text = ch["text"]

                all_texts.append(text)
                all_metadata.append((chunk_uid, pages, granularity, text))

        if all_texts:
            # Embed and add to FAISS
            embeddings = self.embedding_model.encode(
                all_texts,
                batch_size=256,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")
            faiss.normalize_L2(embeddings)

            start_vec_id = self.faiss_index.ntotal
            self.faiss_index.add(embeddings)
            num_vectors = embeddings.shape[0]

            for i in range(num_vectors):
                vec_id = start_vec_id + i
                chunk_uid, pages, granularity, text = all_metadata[i]

                # In memory
                rec_chunk = ChunkRecord(
                    vector_id=vec_id,
                    chunk_uid=chunk_uid,
                    doc_id=doc_id,
                    md_path=md_path,
                    pages=pages,
                    granularity=granularity,
                    text=text,
                )
                self.chunk_metadata.append(rec_chunk)

                # Persist
                cur.execute(
                    """
                    INSERT INTO chunks (vector_id, chunk_uid, doc_id, md_path, pages_json, granularity, chunk_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        vec_id,
                        chunk_uid,
                        doc_id,
                        md_path,
                        json.dumps(pages),
                        granularity,
                        text,
                    ),
                )

        # BM25 doc
        self._maybe_add_bm25(rec)

    def _maybe_add_bm25(self, rec: Dict[str, Any]) -> None:
        extracted_meta = rec.get("extracted_meta")
        raw_output = rec.get("raw_output")

        if extracted_meta:
            bm25_text = self._build_bm25_text(extracted_meta)
        else:
            bm25_text = raw_output or ""

        if not bm25_text.strip():
            return

        tokens = self._tokenize_for_bm25(bm25_text)
        if not tokens:
            return

        doc_id = rec["doc_id"]
        self.bm25_corpus_tokens.append(tokens)
        self.bm25_doc_ids.append(doc_id)

    def _finalize_bm25(self) -> None:
        if self.bm25_corpus_tokens:
            self.bm25 = BM25Okapi(self.bm25_corpus_tokens)
        else:
            self.bm25 = None

    # --------------- search ---------------

    def search_bm25(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Keyword search over doc level metadata.
        Returns doc_id and full extracted_meta (plus paths).
        Never silently hides all-zero results; you always see top_k with scores.
        """
        if self.bm25 is None:
            return []

        tokens = self._tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Debug sanity if things look weird
        # print("BM25 query tokens:", tokens)
        # print("BM25 score stats: min=", scores.min(), "max=", scores.max())

        # Get indices sorted by score desc
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: List[Dict[str, Any]] = []
        cur = self.conn.cursor()

        for idx in top_indices:
            score = float(scores[idx])
            doc_id = self.bm25_doc_ids[idx]

            row = cur.execute(
                "SELECT * FROM docs WHERE doc_id = ?", (doc_id,)
            ).fetchone()
            if not row:
                continue

            extracted_meta = (
                json.loads(row["extracted_meta_json"])
                if row["extracted_meta_json"]
                else None
            )

            results.append(
                {
                    "doc_id": doc_id,
                    "score": score,
                    "md_path": row["md_path"],
                    "pdf_path": row["pdf_path"],
                    "metadata": extracted_meta,
                }
            )

        return results

    def search_faiss(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Semantic search over chunks.
        Returns doc_id and pages (plus a bit of chunk metadata).
        """
        if self.faiss_index.ntotal == 0:
            return []

        q_vec = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, ids = self.faiss_index.search(q_vec, top_k)
        scores = scores[0]
        ids = ids[0]

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores, ids):
            if idx < 0:
                continue
            # mirror list is aligned with vector_id
            if idx >= len(self.chunk_metadata):
                continue
            meta = self.chunk_metadata[idx]
            item = asdict(meta)
            item["score"] = float(score)
            # tool facing fields: doc_id and pages are here
            results.append(item)
        return results

    # --------------- doc content helper ---------------

    def get_doc_content(self, doc_id: str, pages: Optional[List[int]] = None) -> str:
        """
        Load markdown for doc_id.
        If pages is given, return only those pages based on DeepSeek page markers.
        """
        cur = self.conn.cursor()
        row = cur.execute(
            "SELECT md_path FROM docs WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            raise ValueError(f"Unknown doc_id: {doc_id}")

        md_path = row["md_path"]
        md_text = Path(md_path).read_text(encoding="utf-8", errors="ignore")

        if not pages:
            return md_text

        # Reuse same cleaning and page splitting as chunker
        cleaned = deepseek_clean(md_text)
        page_list = _split_into_pages(cleaned)  # List[(page_no, text)]
        page_map = {p_no: txt for p_no, txt in page_list}

        selected: List[str] = []
        for p in pages:
            if p in page_map:
                selected.append(page_map[p])

        if not selected:
            # Fallback to full text if pages not found
            return md_text

        return "\n\n".join(selected)

    # --------------- FAISS save/load ---------------

    def save_faiss(self, path: str) -> None:
        faiss.write_index(self.faiss_index, path)

    @staticmethod
    def load_from_existing(
        db_path: str,
        faiss_path: str,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> "LegalIndexer":
        """
        Recreate indexer from existing SQLite and FAISS index.

        Rebuilds:
          - in memory chunk_metadata from `chunks` table
          - BM25 corpus and model from `docs` table
        """
        indexer = LegalIndexer(
            db_path=db_path,
            embedding_model_name=embedding_model_name,
        )

        # Load FAISS
        indexer.faiss_index = faiss.read_index(faiss_path)
        indexer.embedding_dim = indexer.faiss_index.d

        cur = indexer.conn.cursor()

        # Rebuild chunk_metadata from SQLite
        rows = cur.execute("SELECT * FROM chunks ORDER BY vector_id").fetchall()

        indexer.chunk_metadata = []
        for row in rows:
            pages = json.loads(row["pages_json"]) if row["pages_json"] else []
            indexer.chunk_metadata.append(
                ChunkRecord(
                    vector_id=row["vector_id"],
                    chunk_uid=row["chunk_uid"],
                    doc_id=row["doc_id"],
                    md_path=row["md_path"],
                    pages=pages,
                    granularity=row["granularity"],
                    text=row["chunk_text"],
                )
            )

        # Rebuild BM25 from docs table
        indexer.bm25_corpus_tokens = []
        indexer.bm25_doc_ids = []

        rows = cur.execute(
            "SELECT doc_id, extracted_meta_json, raw_output FROM docs"
        ).fetchall()

        for row in rows:
            doc_id = row["doc_id"]
            extracted_meta = (
                json.loads(row["extracted_meta_json"])
                if row["extracted_meta_json"]
                else None
            )
            raw_output = row["raw_output"]

            # Reuse same logic as indexing
            if extracted_meta:
                bm25_text = indexer._build_bm25_text(extracted_meta)
            else:
                bm25_text = raw_output or ""

            if not bm25_text.strip():
                continue

            tokens = indexer._tokenize_for_bm25(bm25_text)
            if not tokens:
                continue

            indexer.bm25_corpus_tokens.append(tokens)
            indexer.bm25_doc_ids.append(doc_id)

        indexer._finalize_bm25()

        return indexer


if __name__ == "__main__":
    # Example usage
    jsonl_path = "test_metadata.jsonl"

    indexer = LegalIndexer(db_path="legal_index.db")
    indexer.index_jsonl(jsonl_path)

    print("FAISS vectors:", indexer.faiss_index.ntotal)
    print("Chunks stored:", len(indexer.chunk_metadata))
    print("BM25 docs:", len(indexer.bm25_doc_ids))

    q = "mahindra notice plaintiff"
    bm25_hits = indexer.search_bm25(q.lower(), top_k=3)
    vec_hits = indexer.search_faiss(q, top_k=3)

    print("BM25 hits:")
    for hit in bm25_hits:
        print(hit["doc_id"], hit["score"], hit["metadata"]["case_title"])

    print("\nFAISS hits:")
    for hit in vec_hits:
        print(hit["doc_id"], hit["pages"], hit["score"])

    # Example doc_content call
    if vec_hits:
        doc_id = vec_hits[0]["doc_id"]
        pages = vec_hits[0]["pages"]
        content = indexer.get_doc_content(doc_id, pages=pages)
        print("\nSample content length:", len(content))
