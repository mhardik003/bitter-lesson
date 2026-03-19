# Bitter Lesson — Legal SME for Civil Law (India): Contracts and Agreements
This repository provides an end-to-end RAG SME for Indian Contracts Law.
A RAG system for a Subject Matter Expert on Contracts and
Agreements. Involves data acquisition, chunking, indexing and
qualitative evaluation of three retrieval systems: BM25, LegalBERT and
MPNet, metadata labelling using a smaller local model, creation of
tools and sub-agents to carry out tasks like document exports and
email and a meta agent for conversation with a UI which supports
streaming thinking tokens and tool calls live.

This README is a short report on the structure and philosophy behind
our design decisions.

# Quick Setup
Start with `uv sync` to download dependencies. Make sure to populate
the `.env` files with variables as specified in `.env.example`. All
files have constants declared upfront, make sure to tune them
according to folder structure.
## Preprocessing
### Curation
Curate documents from diverse sources, put them all into a single folder.

### OCR
Run OCR on the folder using `vllm/batch_vllm_ocr.py` (further instructions in `vllm/`) to convert PDFs to well formatted markdown documents.

### Metadata Labelling
Run a small local model (gemma-3-4b) to label metadata, fields can be specified as in `vllm/synth_metadata.py`.

## Retrieval system setup
### Chunking
Use `uv run chunker.py` to load pages from document markdown and create chunks.

### Indexing
Use `uv run index.py` to create a BM25 index based on document metadata and a FAISS index based on document chunks.

## Backend Server
Run `uv run indexer_service.py` to get the backend flask server running, it loads up the indexer from the saved `sqlite` db and FAISS index.

## Streamlit frontend
Run `uv run streamlit run webapp.py` to get the frontend `streamlit` server running.


## Repository Layout

```
.
├── chunker.py                              # chunking documents
├── vllm/                                   # for fast inference using vLLM
│   ├── synth_metadata.py                   # metadata labelling using Gemma-3 4b
│   └── batch_vllm_ocr.py                   # run batched Deepseek OCR on a folder of pdfs
├── index.py                                # create a BM25 and FAISS index
├── tools/
│   ├── emailer.py                          # sending mails with attachments
│   ├── searching.py                        # searching using the flask server endpoints
│   └── sectioner.py                        # getting sections and chapters from Indian Contract Act
├── agents/
│   ├── converters.py                       # LLM call + backend hit for converting between formats 
│   ├── core.py                             # core agents setup
│   └── reranker.py                         # agentic reranking of retrieved documents
├── eval/
│   └── results.md                          # qualitative retrieval table (for comparison between different embedding models)
└── webapp.py                               # run the streamlit web app
```

# More process details and design rationale

## The Bitter lesson

Legal expert systems in the 80s–90s failed because they tried to
encode statutes directly as rules, assuming the text itself contained
the operative logic. But the real “fundamental principles” of law live
in judicial reasoning, not in the bare wording of Acts:

> “Behind the Acts is a general body of common law and equity applying
> to all companies irrespective of their nature, and it is there that
> most of the fundamental principles will be found. The fact that most
> of the fundamental principles will not be found in the statutes
> themselves, but in the reported decisions of cases, has immense
> implications for those who would work outwards from the detailed
> wording of individual statutes.”  
> [Legal Knowledge Based Systems: JURIX 92 — Information Technology and Law](https://jurix.nl/pdf/j92-15.pdf)

This aligns with our design: instead of hard-coding legal rules, we
expose the agent to a large case corpus, metadata, and page-accurate
chunking so it can infer meaning and apply the Act as courts actually
do.

We deeply follow the bitter lesson: build robust, scalable pipelines
that give the agent rich access to the entire legal corpus. With
enough data, retrieval tools, metadata, and context, the agent can
decide when to use `bm25`, `faiss`, `document_pages`, or `metadata`
without handcrafted logic. The scaling principle behind this
philosophy is discussed in Sutton’s classic piece: [The Bitter
Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html).

To support this, we employ a comprehensive data annotation and
indexing process.

## Data Acquisition

For full details, refer to `Data_acquisition.md`.

| Phase | Source               | Method                   | Outcome                           |
|-------|----------------------|--------------------------|-----------------------------------|
| 0     | Indian Contracts Act | OCR+Section+Chapters     | Used directly by contracts agent. |
| 1     | Government domains   | `wget` of GCC/MCA PDFs   | Seed corpus                       |
| 2     | Supreme Court portal | Manual downloads         | Landmark cases                    |
| 3     | Indian Kanoon        | Automated scraper        | Large-scale corpus                |
| 4     | Consolidation        | Deduplication + metadata | Clean corpus for OCR              |


## OCR to Markdown

We use **DeepSeek-OCR** with **vLLM** for batched GPU inference, which outputs structured **Markdown** directly.
This retains headings and tables while removing the need for post-OCR heuristics.
More details in the `vllm/` folder.

## Preprocessing and Chunking

We clean the DeepSeek OCR output (remove ref/det tags, empty table cells, extra punctuation) but keep inline page markers like <<<PAGE X>>>. The text is then split into pages, concatenated into a single global string, and each page’s character span is recorded.

We tokenize the global text once using the MPNet tokenizer and build fixed-size, non-overlapping token windows at multiple granularities (384, 256, 128). Each chunk stores its text and the page numbers whose spans overlap it.

Short tail fragments are pruned, and larger granularities skip tails that the smallest granularity already covers. This avoids duplicate end-chunks and keeps the index compact and page-accurate.

```python
def chunk_md_with_granularities(
    md_text: str,
    tokenizer=None,
    token_window_sizes: List[int] = (384, 256, 128),
    min_tokens: int = 32,
) -> Dict[int, List[Dict[str, Any]]]
```

## Metadata labelling
To be able to use BM25 search for keywords, we label every document
using a small local LM (gemma-3-4b) with details like `case_name`,
`human_readable_date`, `parties`, `summary`, `sections (with pages)`,
`keywords`, `statutes_mentioned`.

A sample of the metadata labelling output can be found in `test_metadata.jsonl`.

## Retrieval system
We employ both keyword based and vector based retrieval systems,
letting the agent decide which one to use based on the
situation. There is a BM25 index created based on the document
metadata, to be used for initial keyword search for filtering down
relevant documents. Then there is a FAISS index based on chunks,
annotated with the exact page numbers and the `doc_id` they come
from. 

A chunk looks like this.
```
class ChunkRecord:
    vector_id: int  # index in FAISS
    chunk_uid: str  # doc_id::g{granularity}::c{local_idx}
    doc_id: str
    md_path: str
    pages: List[int]
    granularity: int
    text: str
```

We also create two more accompanying retrieval helper functions:
`get_doc_metadata(doc_id)` and `get_doc_content(doc_id,
pages:List[int])`, this lets the agent *zoom out*, when required, to
get more context on the chunks and docs it finds, this is a clear
alternative to other strategies like providing chunks before and after
the relevant chunks, except we believe this is much more scalable
since the agent can decide by itself how big the context around the
chunk it really needs.

## Tools and Agents
We create a variety of tools to help agent do its job correctly, all
tools have a comprehensive docstring and input/output parameter types
so that the agent is able to call them correctly.

We also create *subagents* - these are agents in of themselves, which
are able to call tools autonomously and have their own system prompts
specialised for their usecase like `query_indian_contracts_act`
sub-agent which works with the chapters and sections of the Indian
Contracts Act and provides grounded commentary on those, and the
`case_study` sub-agent which specialises in providing crisp case
studies with citation support, and the `conversion` and `repair`
subagents which check and format the markdown to be specialised for 
the format being converted to and repair the markdown in case of a 
conversion failure, respectively.

## Streamlit UI
We deliberately chose a thinking and tool-call side-by-side UI since
both had their own individual streams and would end up eating into
each other's vertical space if they were stacked. An interesting
decision was to hide the thinking trace of sub-agents yet keep the
tool-calls they make, this keeps the thinking trace clean while also
letting the user know that the sub-agents are working in the
background.

## Learnings from evaluation
BM25 consistently surfaced statute-heavy and keyword-dense documents
better than dense retrieval, while FAISS dominated on fact-pattern
similarity and nuanced legal phrasing. Agentic reranking helped unify
the two by letting the agent “pivot” mid-query. Overall,
mixed-retrieval outperformed any single method.

## Robustness comment
The pipeline survives noisy OCR, inconsistent formatting, duplicate
PDFs, tail fragments, conversion failures and malformed markdown
because every stage is designed to detect and repair flaws rather than
assume clean input. The system favors redundancy, sanity checks, and
tool-level fallback paths to keep the agent effective even when the
data is messy.



# TODO
- [x] Metadata pipeline
- [x] Chunking final
- [x] Vector Index
- [x] BM25 Index
- [x] Tools for above
- [x] RAG agent for case study
- [x] RAG agent which can call case study
- [x] Tools for email 
- [x] tools for pandoc/latex running
- [x] Restructure
- [x] Meta Agent
- [x] UI check
