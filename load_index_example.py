from index import LegalIndexer

DB_PATH = "/scratch/akshit.kumar/indices/legal_index.db"
FAISS_PATH = "/scratch/akshit.kumar/indices/faiss.index"


indexer = LegalIndexer.load_from_existing(DB_PATH, FAISS_PATH)

print("FAISS vectors:", indexer.faiss_index.ntotal)
print("Chunks:", len(indexer.chunk_metadata))
print("BM25 docs:", len(indexer.bm25_doc_ids))
