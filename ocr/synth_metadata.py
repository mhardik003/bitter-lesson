import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from vllm import LLM, SamplingParams

# ------------- config -------------

MODEL_NAME = "google/gemma-3-4b-it"
CORPUS_PATH = "/scratch/akshit.kumar/md/5kcorpus.jsonl"
OUTPUT_META_PATH = "/scratch/akshit.kumar/md/5k_meta.jsonl"  # output jsonl

BATCH_SIZE = 16  # tune for your GPU and context length
MAX_CHARS_MD = 64000  # 64k chars: enough for metadata


# ------------- prompt builder -------------


def build_prompt(doc_meta: Dict[str, Any], markdown_text: str) -> str:
    """
    Build a single prompt for one document.
    We ask for a single JSON object and nothing else.
    """

    md_trunc = markdown_text[:MAX_CHARS_MD]

    prompt = f"""
You are an information extraction model.

Extract structured metadata from the following legal or technical document.

Return a single JSON object with exactly these keys:

- "doc_id": string
- "case_title": string, if there is no explicit title, create an appropriate title
- "doc_type": one of ["judgment", "contract", "statute", "paper", "other"]
- "decision_date": string, DD/MM/YY or null
- "decision_date_human": string, DD Month YYYY or null
- "statutes_mentioned": List[string], can be empty
- "jurisdiction": string or null
- "court": string
- "parties": list of objects with keys:
    - "name": string
    - "role": string, such as "plaintiff", "defendant", "petitioner", "respondent", "author", "other"
- "topics": list of 3 to 5 short strings (1 to 5 words each)
- "summary": at most 2 sentences summarising the whole document
- "sections": list of at most 5 objects with keys:
    - "heading": short string
    - "summary": at most 2 sentences summary of that section
    - "pages": A two-element JSON list of integers in the form [start_page, end_page]. Never use parentheses or tuples. Extract start_page and end_page from the <<<Page x>>> markers in the markdown.



Strict rules:
- Output valid JSON only.
- Do not include any text before or after the JSON.
- Do not put the double quote character (") inside any string values. If you need to mention a term, write it without quotes.
- Use null if you cannot find a field.
- If you are unsure about doc_type or jurisdiction, guess from the content or use "other" or null.

Document metadata:
- doc_id: {doc_meta.get("doc_id")}
- title: {doc_meta.get("title")}
- doc_type: {doc_meta.get("doc_type")}
- jurisdiction: {doc_meta.get("jurisdiction")}

Document content (markdown):

\"\"\"markdown
{md_trunc}
\"\"\"


Now output the JSON object for this document:
"""
    return prompt.strip()


# ------------- load corpus -------------


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    docs = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            docs.append(json.loads(line))
    return docs


# ------------- JSON extraction helper -------------


def extract_json_from_text(
    raw_text: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[Tuple[Exception, str]]]:
    """
    Try to recover a JSON object from the model output.

    Returns:
        (parsed_json, None) on success
        (None, (error, candidate_text)) on failure
    """
    text = raw_text.strip()

    # 1. Remove ``` fences if present
    if "```" in text:
        fence_start = text.find("```")
        fence_end = text.rfind("```")
        if fence_end > fence_start:
            inner = text[fence_start + 3 : fence_end].strip()
        else:
            inner = text
        # remove leading "json" if present
        if inner.lower().startswith("json"):
            inner = inner[4:].strip()
        text = inner

    # 2. Truncate to last closing brace
    last_brace = text.rfind("}")
    if last_brace != -1:
        candidate = text[: last_brace + 1]
    else:
        candidate = text

    # 3. Try parsing
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, (e, candidate)


# ------------- batching helper -------------


def yield_batches(seq, batch_size):
    for i in range(0, len(seq), batch_size):
        yield seq[i : i + batch_size]


# ------------- main pipeline -------------


def main():
    # init model once
    llm = LLM(model=MODEL_NAME)

    sampling_params = SamplingParams(
        temperature=0.0,  # deterministic for JSON
        top_p=1.0,
        max_tokens=4096,  # tune if summaries get cut off
    )

    docs = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(docs)} docs from {CORPUS_PATH}")

    out_f = open(OUTPUT_META_PATH, "w", encoding="utf-8")

    for batch_docs in yield_batches(docs, BATCH_SIZE):
        prompts = []
        meta_for_batch = []

        for doc in batch_docs:
            md_path = Path(doc["output_markdown"])
            try:
                md_text = md_path.read_text(encoding="utf-8", errors="ignore")
            except FileNotFoundError:
                print(f"Warning: markdown not found for {doc['doc_id']} at {md_path}")
                continue

            prompt = build_prompt(doc, md_text)
            prompts.append(prompt)
            meta_for_batch.append(doc)

        if not prompts:
            continue

        outputs = llm.generate(prompts, sampling_params)

        for doc, out in zip(meta_for_batch, outputs):
            raw_text = out.outputs[0].text.strip()

            parsed, err = extract_json_from_text(raw_text)
            if parsed is None:
                e, candidate = err
                print(f"[PARSE ERROR] doc_id={doc['doc_id']}: {e}")
                meta_json = {
                    "doc_id": doc["doc_id"],
                    "parse_error": str(e),
                    "raw_output": candidate,
                }
            else:
                meta_json = parsed

            combined = {
                "doc_id": doc["doc_id"],
                "pdf_path": os.path.normpath(doc["source_path"]),
                "md_path": doc["output_markdown"],
                "extracted_meta": meta_json,
            }
            out_f.write(json.dumps(combined, ensure_ascii=False) + "\n")

    out_f.close()
    print(f"Saved metadata to {OUTPUT_META_PATH}")


if __name__ == "__main__":
    main()
