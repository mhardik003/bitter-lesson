[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/PHuQlbNP)

# Legal SME: Civil Law “Contracts and Agreements” (India)

**Team:** bitter lesson
**Audience:** junior associates in Indian law firms
**Jurisdiction:** India only
**Scope for mid:** Indian Contract Act, 1872 first, then grow
**Data policy:** open sources only
**Mid evaluation:** 25 Oct 2025
**Final submission:** 12 Nov 2025

This README is the single source of truth for humans and for LLM context. It defines scope, data specs, chunking policy, indexing plan, evaluation pipeline, and extension hooks.

---

## Project overview

Goal: build a retrieval-first Subject Matter Expert for Indian “Contracts and Agreements,” tuned to how junior associates actually work: finding the right section, the right paragraph, and the right case ratio fast, with precise citations to support drafting and research.

The course brief requires a RAG-capable SME with workflows that chain tools, with robust QA on the subject matter. Our mid submission covers data and retrieval up to Section C (collection, preprocessing, indexing). Later phases add agent workflows and task capabilities.  

---

## What counts for the mid submission (rubric alignment)

We must deliver Sections A through C:

* **A. Document Collection and Organization**: root-level corpus of heterogeneous formats, full metadata, automated type detection, justified corpus size. 
* **B. Preprocessing and Chunking**: multi-granularity segmentation (2048, 512, 128 tokens), content-aware overlap, and standard cleaning. We will justify choices. 
* **C. Embedding and Indexing**: baseline `all-mpnet-base-v2`, one legal-tuned model, vector DB with parent-child structure, BM25, optional rerank bonus. 

Everything beyond C (Core SME capabilities, agent design, tool-using features, and system components) is for final and demo.   

---

## 3. Problem framing and “bitter lesson” angle

Hand-crafted legal decision trees break on edge cases. Clause-aware retrieval over a clean corpus beats brittle rules. We will show empirical improvements from hybrid retrieval and minimal stitching over naive keyword-only search. The mid deliverables focus on doing the simple thing well: get the right text, fast, with traceable pointers.

---

## 4. Audience, jurisdiction, and scope

* **User:** junior associates
* **Jurisdiction:** India only
* **Initial law:** Indian Contract Act, 1872 (ICA)
* **Companion statutes for context:** Specific Relief Act, Limitation Act (only sections relevant to contracts)
* **Cases:** a small, landmark set around formation, vitiating factors, breach, and damages

---

## 5. Repository layout

```
.
├── data/
│   ├── sources/                    # raw docs by source
│   ├── corpus.jsonl                # one record per source document
│   ├── chunks.jsonl                # one record per chunk
│   └── licenses/                   # license and attribution snapshots
├── index/
│   ├── vector/                     # vector store files
│   └── bm25/                       # inverted index
├── eval/
│   ├── queries.jsonl               # {qid, text, tags}
│   ├── qrels.tsv                   # qid, chunk_id, rel
│   └── runs/                       # method_name.tsv (rankings)
└── traces/
    └── <method>/<qid>.json         # multi-step retrieval traces
```

---

## 6. Metadata schema (v1.0)

**Document record** in `data/corpus.jsonl`:

```json
{
  "doc_id": "ica1872@v1",
  "title": "The Indian Contract Act, 1872",
  "doc_type": "statute",
  "jurisdiction": "IN",
  "source_url": "https://example.gov.in/ica1872/official",
  "publisher": "Government of India",
  "date_published": "1872-09-01",
  "date_accessed": "2025-10-24",
  "version": "official",
  "topics": ["offer","acceptance","consideration","damages","penalty"],
  "citations": [
    {
      "kind": "statute",
      "doc_id": "ica1872@v1",
      "pointer": "ica1872@v1::sec/s74::g=512::o=0",
      "locator": {"section": "74"},
      "canonical_cite": "Indian Contract Act, 1872, §74"
    }
  ],
  "sections": [
    {
      "section_id": "s74",
      "label": "Section 74",
      "heading": "Compensation for breach of contract where penalty stipulated for",
      "paragraphs": [
        {"para_id": "s74-p1", "text": "<text...>"}
      ]
    }
  ],
  "crawl_timestamp": "2025-10-24T12:00:00Z",
  "license": "GODL-India or Public Domain (courts)"
}
```

Rationale for strong metadata and citations is baked into the brief’s expectations on organization and traceability. 

---

## 7. Chunking policy

Legal text needs context fidelity. We preserve natural boundaries and still satisfy multi-granularity and overlap requirements in the rubric. 

**Design rules**

* **Statutes**

  * Atomic anchor: section
  * Export three granularities: 2048, 512, 128 tokens
  * Do not split mid-paragraph; when a section is long, window by sentence boundaries
  * Stitch window at retrieval time: fetch neighbors `±1` section

* **Cases**

  * Parts: headnote, facts, issues, holding, ratio, obiter, disposition
  * Keep paragraph numbers inside parts; group into chunks that fit token budgets
  * Retrieval stitch: neighbors `±2` paragraphs within the same part

* **Contracts**

  * Atomic anchor: clause (retain clause id and clause_type such as termination, force majeure, confidentiality)
  * Optionally map each clause to likely governing statute sections or concepts for KG use later

* **Overlap**

  * Minimal by default since we keep natural units
  * Add small sentence-level windowing only if a paragraph exceeds the target size

**Chunk record** in `data/chunks.jsonl`:

```json
{
  "chunk_id": "ica1872@v1::sec/s74::g=512::o=0",
  "doc_id": "ica1872@v1",
  "scope": {"type": "statute_section", "section_id": "s74"},
  "parents": ["ica1872@v1::sec/s74"],
  "neighbors": [
    "ica1872@v1::sec/s73::g=512::o=0",
    "ica1872@v1::sec/s75::g=512::o=0"
  ],
  "text": "<text...>",
  "topics": ["damages","penalty","liquidated_damages"],
  "page_map": [{"page": 14, "char_start": 980, "char_end": 1560}],
  "sha256_chunk": "<hash>"
}
```

---

## 8. Canonical IDs and pointers

Deterministic, URL-like, and parseable. This is the backbone for citations, KG edges, eval, and multi-step traces.

```
<doc_id>::<scope>::g=<granularity>::o=<offset>
```

* `doc_id`: human-readable slug with version, for example `fatehchand_1963_supreme_court@v1`
* `scope`: `sec/s74` for statute sections, `case/ratio/p=12-16` for case parts and paragraph spans, `cl/12.3` for contract clauses
* `g`: token granularity (128 or 512 or 2048)
* `o`: byte or char offset for windowed subchunks, `0` if not windowed

---

## 9. Indexing and models

Rubric wants a baseline general model and one domain model, with vector DB and parent-child preservation, plus keyword search and optional rerank. 

* **Dense base:** `all-mpnet-base-v2`
* **Dense legal:** legal-tuned sentence transformer (to be selected from open models by shortlisting and a small head-to-head)
* **Sparse:** BM25
* **Storage:** vector store with parent-child links and metadata filters, and a BM25 inverted index
* **Rerank (bonus if time):** BGE Reranker on hybrid shortlist

---

## 10. Evaluation pipeline

We measure retrieval quality and trace multi-step retrieval behavior even before full RAG answers.

**Artifacts**

* `eval/queries.jsonl`

  ```json
  {"qid": "Q001", "text": "When can a penalty clause be reduced to reasonable compensation under Indian law?", "tags": ["damages","penalty","ICA-74"]}
  ```
* `eval/qrels.tsv` (gold relevance, multiple correct chunks per query)

  ```
  Q001  ica1872@v1::sec/s74::g=512::o=0  1
  Q001  fatehchand_1963_supreme_court@v1::case/ratio/p=12-16::g=512::o=0  1
  Q001  ongc_saw_pipes_2003_sc@v1::case/holding/p=5-9::g=512::o=0        1
  ```
* `eval/runs/<method>.tsv` (ranked lists with scores)

  ```
  Q001  1  ica1872@v1::sec/s74::g=512::o=0   0.83
  Q001  2  ongc_saw_pipes_2003_sc@v1::case/holding/p=5-9::g=512::o=0  0.79
  ```

**Metrics**

* **Hit@k** and **nDCG@k**
* **Manual citation faithfulness** on a small subset
* **Latency** and **index size** (engineering side)

**Ablations**

1. BM25 only
2. Dense only
3. Hybrid union with score normalization
4. Hybrid + rerank

**Multi-step traces** in `traces/<method>/<qid>.json`:

```json
{
  "qid": "Q001",
  "steps": [
    {"name": "initial_retrieval", "topk": ["ica1872@v1::sec/s74::g=512::o=0", "fatehchand_1963_supreme_court@v1::case/ratio/p=12-16::g=512::o=0"]},
    {"name": "stitch_expand", "added": ["ica1872@v1::sec/s73::g=512::o=0"]},
    {"name": "rerank", "reranked_topk": ["fatehchand_1963_supreme_court@v1::case/ratio/p=12-16::g=512::o=0", "ica1872@v1::sec/s74::g=512::o=0"]}
  ],
  "final_context": [
    "fatehchand_1963_supreme_court@v1::case/ratio/p=12-16::g=512::o=0",
    "ica1872@v1::sec/s74::g=512::o=0"
  ]
}
```

**Trace scoring**

* Path consistency: final_context intersects qrels
* Step utility: delta Hit@k from initial retrieval to rerank
* Redundancy penalty: downscore near-duplicate chunks that do not improve coverage

---

## 11. Data sources and licensing posture

We will use open government sources for statutes and official court texts where possible. Government and court texts are prioritized for clear licensing and citation hygiene. The brief permits web-scraped text files and heterogeneous formats. 

**Initial collection targets**

* **Acts:** Indian Contract Act, Specific Relief Act (select parts), Limitation Act (time bars for contract suits)
* **Judgments:** a small set of Supreme Court landmarks for offer and acceptance, vitiating factors, frustration, damages and penalties
* **Model contracts:** government-issued model agreements and concession templates for clause mining
* **Open educational notes:** only as background and for eval explanations, not as binding sources

**Licensing rules**

* Prefer official government and court publications
* Avoid publisher headnotes and editorial content
* Record license info and attribution in metadata and keep a snapshot in `data/licenses/`

---

## 12. Knowledge graph scaffolding (optional for mid, prepared now)

Emit edges while chunking to support concept-level retrieval later.

**Nodes**: `Case`, `StatuteSection`, `ContractClause`, `Concept`
**Edges**:

* `Case` cites `StatuteSection`
* `Case` overrules `Case` (if present)
* `ContractClause` governed_by `StatuteSection`
* `Concept` defined_in `StatuteSection`, applied_in `Case`

Each edge carries a grounding `chunk_id` and optional quote span.

```json
{
  "edge_id": "e-0001",
  "src": {"type": "case", "id": "fatehchand_1963_supreme_court@v1"},
  "dst": {"type": "statute_section", "id": "ica1872@v1::sec/s74"},
  "rel": "applies",
  "grounding_chunk": "fatehchand_1963_supreme_court@v1::case/ratio/p=12-16::g=512::o=0",
  "quote_span_char": [45, 120]
}
```

---

## 13. System components (for final phase, documented now)

The brief expects an API server, a chat or tool server that manages context, and a modular retrieve-and-rerank pipeline. We will align to this architecture for the final.  

* Main API server (Flask)
* Tool layer for retrieval, indexing, and document export
* RAG pipeline with hybrid retrieval and reranking
* Optional guardrails for prompt injection and output moderation (bonus) 

---

## 14. Deliverables checklist

**For mid (A to C):**

* Corpus under `data/sources/` with `corpus.jsonl` metadata
* Preprocessing outputs with `chunks.jsonl` at 2048, 512, 128 token granularities and the overlap policy documented
* Vector index and BM25 index under `index/`
* A small **evaluation** run with queries, qrels, at least 3 methods (BM25, dense, hybrid), with metrics and a short discussion
* A short methodology page or section in this README explaining choices and results

These items satisfy the rubric. 

**For final (D to I):** task capabilities, agent routing, tool use, system components, and demo video.  

---

## 15. Risks and mitigations

* **Statute variants and formatting drift:** pin versions, record `sha256_raw`, maintain `@vN` in `doc_id`
* **Case text quality:** prefer official PDFs; when using aggregator text, always store official citations and back up with parallel links
* **Over-chunking legal tests:** we use paragraph and section boundaries and retrieval-time stitching
* **Eval leakage and bias:** keep gold lists small but diverse; report per-topic breakdowns

---

## 16. Non-goals for the mid

* No legal advice
* No drafting automation beyond retrieval traces
* No agent routing or document generation required until final

---

## 17. Quick how-to (developer notes)

* Place raw sources under `data/sources/<family>/<slug>/`
* Run the cleaner which writes `corpus.jsonl`
* Run the chunker which writes `chunks.jsonl` with three granularities
* Build `index/vector` and `index/bm25`
* Run eval scripts to produce `eval/runs/*.tsv` and a `results.md` table
* Drop multi-step traces in `traces/`

---

## 18. Appendix: rubric excerpts we track

* Mid submission is up to Embedding and Indexing. 
* Multi-granularity chunking and overlap are expected. 
* Baseline model `all-mpnet-base-v2`, try a domain model, preserve parent-child, vector DB options, rerank bonus. 
* Final includes agent workflows, tool use, system components, and a demo video.  

---

## 19. Contact and disclaimer

* Team: bitter lesson
* Educational research tool only, not legal advice.

If you need to extend scope beyond the Indian Contract Act after mid, add new documents through the same metadata and chunking pipeline, keep the `doc_id` and `chunk_id` rules, and update qrels to include new gold pointers.
