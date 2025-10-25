[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/PHuQlbNP)
# **Bitter Lesson — Legal SME for Civil Law (India): Contracts and Agreements**

A RAG system for a Subject Matter Expert on Contracts and Agreements. Involves data acquisition, chunking, indexing and qualitative evaluation of three retrieval systems: BM25, LegalBERT and MPNet.

## Scope

The corpus currently includes:

* **Statutes** — Indian Contract Act, Specific Relief Act, Arbitration and Conciliation Act, etc.
* **Government documents** — GCCs (General Conditions of Contract), MCAs (Model Concession Agreements), tender and procurement manuals.
* **Judgments and case law** — From the Supreme Court, High Courts, and tribunals, focusing on contract enforcement, frustration (Section 56), damages and penalties (Sections 73–75), and related areas.
* **Templates and procedural documents** — Publicly released standard forms from CPSEs (BHEL, GAIL, IOCL, DFCCIL, etc.).

All data is **publicly available and open source**, scraped or downloaded from verified government and legal repositories.

## Repository Layout

```
.
├── ocr/                     # vLLM OCR runtime (separate venv)
├── data/                    # raw PDFs and source files
├── data_tmp/                # temporary rasterized PNGs
├── data_md/
│   ├── md/                  # merged Markdown per document
│   ├── work/<doc>/page_*    # per-page OCR outputs (.mmd)
│   └── corpus.jsonl         # metadata and provenance log
├── data_out/
│   ├── chunks.jsonl         # cleaned and tokenized chunks
│   └── index/
│       ├── bm25/
│       └── vector/
├── eval/
│   └── results.md           # qualitative retrieval table
├── batch_vllm_ocr.py
├── chunk_and_index.py
├── batch_eval_to_md.py
├── queries.jsonl            # evaluation questions
└── README.md
```


## Data Acquisition

For full details, refer to `Data_acquisition.md`.

| Phase | Source               | Method                   | Outcome              |
| ----- | -------------------- | ------------------------ | -------------------- |
| 1     | Government domains   | `wget` of GCC/MCA PDFs   | Seed corpus          |
| 2     | Supreme Court portal | Manual downloads         | Landmark cases       |
| 3     | Indian Kanoon        | Automated scraper        | Large-scale corpus   |
| 4     | Consolidation        | Deduplication + metadata | Clean corpus for OCR |

## OCR to Markdown

We use **DeepSeek-OCR** with **vLLM** for batched GPU inference, which outputs structured **Markdown** directly.
This retains headings and tables while removing the need for post-OCR heuristics.

### Features

* Batched OCR with progress tracking per PDF.
* Skips previously processed PDFs using hash checks.
* Saves both page-level `.mmd` and merged `.md` files.
* Efficient on L40s GPUs (≈32 pages per batch at 300 DPI).

```bash
python batch_vllm_ocr.py \
  --input data/ \
  --out data_md/ \
  --tmp data_tmp/ \
  --batch_pages 16 \
  --model_id deepseek-ai/DeepSeek-OCR \
  --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

## Preprocessing and Chunking

After OCR:

### Cleaning

* Strip DeepSeek tags `<|ref|>`, `<|det|>` and empty `<td></td>` cells.
* Collapse redundant punctuation (e.g., `. .` → `.`).
* Normalize whitespace, lowercase text, preserve section numbering.

### Chunking

* Multi-granularity chunks at **2048**, **512**, and **128** tokens.
* Deterministic IDs of form:

  ```
  <doc_id>::sec/<section_id>::g=<granularity>::o=<offset>
  ```
* Neighbor pointers allow multi-step retrieval.
* Chunks store both normalized and original text.


```bash
python chunk_and_index.py \
  --md_dir data_md/md \
  --out_dir data_out \
  --granularities 2048,512,128 \
  --bm25_granularity 512 \
  --faiss_granularity 512 \
  --vector_model sentence-transformers/all-mpnet-base-v2 \
  --device cpu
```


## Indexing

| Type   | Model                           | Library     | Purpose                    |
| ------ | ------------------------------- | ----------- | -------------------------- |
| Sparse | BM25                            | `rank_bm25` | Lexical baseline           |
| Dense  | all-mpnet-base-v2               | `faiss`     | General semantic retrieval |
| Dense  | nlpaueb/legal-bert-base-uncased | `faiss`     | Domain-specific baseline   |

Indexes are stored under `data_out/index/`.


## Evaluation

A **qualitative mid-submission evaluation** using 10 crafted queries representing realistic legal search intents.
Queries span offers and acceptance, penalties, frustration, e-contracts, arbitration clauses, and performance securities.


```bash
python batch_eval_to_md.py \
  --queries eval/queries.jsonl \
  --chunks data_out/chunks.jsonl \
  --bm25 data_out/index/bm25/bm25_g512.pkl \
  --mpnet_index data_out/index/vector/faiss_sentence-transformers-all-mpnet-base-v2_g512.index \
  --mpnet_ids   data_out/index/vector/faiss_sentence-transformers-all-mpnet-base-v2_g512.ids \
  --legal_index data_out/index/vector/faiss_nlpaueb-legal-bert-base-uncased_g512.index \
  --legal_ids   data_out/index/vector/faiss_nlpaueb-legal-bert-base-uncased_g512.ids \
  --device cpu \
  --topk 3 \
  --out_md eval/results.md \
  --snippet_chars 300
```

Outputs a Markdown table (`eval/results.md`) for visual inspection.


## Observations

* **MPNet** gives coherent statute-level matches.
* **BM25** excels in factual, keyword-specific retrieval.
* **Legal-BERT (CLS)** underperforms without mean pooling; improvement planned.
* OCR structural cues (e.g., “Section …”) significantly enhance retrieval.

