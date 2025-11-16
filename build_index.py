from index import LegalIndexer  # your module


JSONL_PATH = "/scratch/akshit.kumar/md/5k_meta.jsonl"
DB_PATH = "/scratch/akshit.kumar/indices/legal_index.db"
FAISS_PATH = "/scratch/akshit.kumar/indices/faiss.index"

indexer = LegalIndexer(db_path=DB_PATH)
indexer.index_jsonl(JSONL_PATH)

print("FAISS vectors:", indexer.faiss_index.ntotal)
print("Chunks stored:", len(indexer.chunk_metadata))
print("BM25 docs:", len(indexer.bm25_doc_ids))

indexer.save_faiss(FAISS_PATH)
indexer.conn.close()
