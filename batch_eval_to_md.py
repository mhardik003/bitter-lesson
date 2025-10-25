
import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import faiss
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# ---------------- helpers ----------------

def load_chunks(chunks_path: Path):
    id2chunk = {}
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            id2chunk[c["chunk_id"]] = c
    return id2chunk

def bm25_search(bm25_pkl: Path, query: str, k: int = 10) -> List[Tuple[str, float]]:
    with open(bm25_pkl, "rb") as f:
        obj = pickle.load(f)
    bm25 = obj["bm25"]
    ids = obj["ids"]
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_idx = np.argsort(scores)[::-1][:k]
    return [(ids[i], float(scores[i])) for i in top_idx]

def encode_query(model_name: str, device: str, text: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
    with torch.no_grad():
        tok = tokenizer([text.lower()], padding=True, truncation=True, max_length=512, return_tensors="pt")
        tok = {k: v.to(device) for k, v in tok.items()}
        out = model(**tok)
        if hasattr(out, "last_hidden_state"):
            vec = out.last_hidden_state[:,0,:].detach().cpu().numpy()
        else:
            vec = out[0][:,0,:].detach().cpu().numpy()
    faiss.normalize_L2(vec)
    return vec.astype("float32")

def faiss_search(index_path: Path, ids_path: Path, query_vec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
    index = faiss.read_index(str(index_path))
    with ids_path.open("r", encoding="utf-8") as f:
        ids = [line.strip() for line in f if line.strip()]
    D, I = index.search(query_vec, k)
    hits = []
    for dist, idx in zip(D[0], I[0]):
        if idx >= 0 and idx < len(ids):
            hits.append((ids[idx], float(dist)))
    return hits

def md_escape(s: str) -> str:
    # Escape pipe for markdown tables
    return s.replace("|", r"\|").replace("\n", " ").strip()

def summarize_text(chunk_text: str, max_chars: int) -> str:
    t = chunk_text.strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars-1] + "…"

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default="queries.jsonl")
    ap.add_argument("--chunks", default="data_out/chunks.jsonl")
    ap.add_argument("--bm25", default="data_out/index/bm25/bm25_g512.pkl")
    ap.add_argument("--mpnet_index", default="data_out/index/vector/faiss_sentence-transformers-all-mpnet-base-v2_g512.ids")
    ap.add_argument("--mpnet_ids", default="data_out/index/vector/faiss_sentence-transformers-all-mpnet-base-v2_g512.ids")
    ap.add_argument("--legal_index", default="data_out/index/vector/faiss_nlpaueb-legal-bert-base-uncased_g512.index")
    ap.add_argument("--legal_ids", default="data_out/index/vector/faiss_nlpaueb-legal-bert-base-uncased_g512.ids")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out_md", default="eval/results.md")
    ap.add_argument("--snippet_chars", type=int, default=300)
    args = ap.parse_args()

    # Load
    id2chunk = load_chunks(Path(args.chunks))
    queries = []
    with Path(args.queries).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # Prepare model encoders on demand to avoid loading twice
    # We will encode once per model per query
    # Build table lines
    rows = []
    header = "Question | Model Name | Chunk 1 text | Chunk 2 text | Chunk 3 text"
    sep = "--- | --- | --- | --- | ---"
    rows.append(header)
    rows.append(sep)

    for q in queries:
        qid = q.get("qid")
        qtext = q.get("text", "").strip()
        q_disp = md_escape(qtext)

        # BM25
        bm25_hits = bm25_search(Path(args.bm25), qtext, k=args.topk)
        bm25_snips = []
        for cid, _ in bm25_hits:
            chunk = id2chunk.get(cid, {})
            bm25_snips.append(summarize_text(chunk.get("text_original", ""), args.snippet_chars))
        while len(bm25_snips) < args.topk:
            bm25_snips.append("")
        rows.append(f"{q_disp} | BM25 | {md_escape(bm25_snips[0])} | {md_escape(bm25_snips[1])} | {md_escape(bm25_snips[2])}")

        # MPNet
        mpnet_vec = encode_query("sentence-transformers/all-mpnet-base-v2", args.device, qtext)
        mpnet_hits = faiss_search(Path(args.mpnet_index), Path(args.mpnet_ids), mpnet_vec, k=args.topk)
        mpnet_snips = []
        for cid, _ in mpnet_hits:
            chunk = id2chunk.get(cid, {})
            mpnet_snips.append(summarize_text(chunk.get("text_original", ""), args.snippet_chars))
        while len(mpnet_snips) < args.topk:
            mpnet_snips.append("")
        rows.append(f"{q_disp} | MPNet | {md_escape(mpnet_snips[0])} | {md_escape(mpnet_snips[1])} | {md_escape(mpnet_snips[2])}")

        # Legal-BERT
        legal_vec = encode_query("nlpaueb/legal-bert-base-uncased", args.device, qtext)
        legal_hits = faiss_search(Path(args.legal_index), Path(args.legal_ids), legal_vec, k=args.topk)
        legal_snips = []
        for cid, _ in legal_hits:
            chunk = id2chunk.get(cid, {})
            legal_snips.append(summarize_text(chunk.get("text_original", ""), args.snippet_chars))
        while len(legal_snips) < args.topk:
            legal_snips.append("")
        rows.append(f"{q_disp} | Legal-BERT | {md_escape(legal_snips[0])} | {md_escape(legal_snips[1])} | {md_escape(legal_snips[2])}")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
