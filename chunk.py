import re
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer

# ---------------------------
# Cleaning regexes
# ---------------------------

REF_TAG_RE = re.compile(r"<\|ref\|\>.*?<\|/ref\|\>", re.S)
DET_TAG_RE = re.compile(r"<\|det\|\>.*?<\|/det\|\>", re.S)
EMPTY_TD_RE = re.compile(r"\s*<\s*td\s*>\s*</\s*td\s*>\s*", re.I)
TD_BLOCK_RE = re.compile(r"(?:\s*<\s*td\s*>\s*</\s*td\s*>\s*){2,}", re.I)

# <<<PAGE 12>>> or <<<Page 12>>> etc (case insensitive)
PAGE_RE = re.compile(r"<<\<PAGE\s+(\d+)>>>", re.I)


def remove_deepseek_tags(md: str) -> str:
    # Strip grounding and detection tags entirely
    md = REF_TAG_RE.sub("", md)
    md = DET_TAG_RE.sub("", md)
    return md


def remove_empty_tds(md: str) -> str:
    # Remove runs of empty table cells first, then any isolated empty cell
    md = TD_BLOCK_RE.sub("", md)
    md = EMPTY_TD_RE.sub("", md)
    return md


def collapse_punctuation(md: str) -> str:
    # Replace sequences like ". ." ".  ." "… .." with a single period and space
    md = re.sub(r"(?:\s*\.\s*){2,}", ". ", md)
    # Collapse repeated punctuation (commas, semicolons, colons) conservatively
    md = re.sub(r"([,;:])\s*(\1\s*){1,}", r"\1 ", md)
    # Normalize whitespace-only lines
    md = re.sub(r"[ \t]+\n", "\n", md)
    # Collapse >2 blank lines
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


def deepseek_clean(md: str) -> str:
    # Normalize newlines then apply all cleaning passes
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = remove_deepseek_tags(md)
    md = remove_empty_tds(md)
    md = collapse_punctuation(md)
    return md


# ---------------------------
# Page splitting and spans
# ---------------------------


def _split_into_pages(md_text: str) -> List[Tuple[int, str]]:
    """
    Split cleaned DeepSeek OCR markdown into (page_no, raw_text_for_page),
    handling inline <<<PAGE N>>> markers.

    Text before the first marker is treated as page 1.
    Text between <<<PAGE i>>> and <<<PAGE j>>> is page i, etc.
    """
    pages_raw: Dict[int, List[str]] = {}
    current_page = 1
    last_pos = 0

    for match in PAGE_RE.finditer(md_text):
        page_no = int(match.group(1))
        start, end = match.span()

        # Text before this marker belongs to current_page
        if last_pos < start:
            pages_raw.setdefault(current_page, []).append(md_text[last_pos:start])

        # Switch to new page
        current_page = page_no
        last_pos = end

    # Tail after last marker
    if last_pos < len(md_text):
        pages_raw.setdefault(current_page, []).append(md_text[last_pos:])

    # Turn dict into sorted list of (page_no, text)
    pages: List[Tuple[int, str]] = []
    if not pages_raw:
        # No page markers at all: treat whole doc as page 1
        pages.append((1, md_text))
    else:
        for p in sorted(pages_raw.keys()):
            joined = "".join(pages_raw[p]).strip()
            if joined:
                pages.append((p, joined))

    return pages


def _build_global_text_and_spans(
    pages: List[Tuple[int, str]],
) -> Tuple[str, List[Tuple[int, int, int]]]:
    """
    Concatenate pages into one big string and record page spans as
    (char_start, char_end, page_no).
    """
    chunks: List[str] = []
    spans: List[Tuple[int, int, int]] = []
    cursor = 0

    for page_no, text in pages:
        start = cursor
        chunks.append(text)
        cursor += len(text)
        end = cursor
        spans.append((start, end, page_no))

    return "".join(chunks), spans


# ---------------------------
# Chunking
# ---------------------------


def chunk_md_with_granularities(
    md_text: str,
    tokenizer=None,
    token_window_sizes: List[int] = (384, 256, 128),
    min_tokens: int = 32,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Clean DeepSeek OCR markdown, split by pages, then chunk in token space.

    Returns: {window_size: [chunks...]}

    Each chunk:
    {
        "chunk_id": int,
        "granularity": int,   # token window size
        "text": str,
        "pages": List[int],   # pages this chunk overlaps
    }
    """
    # Cleaning step: keep PAGE markers, drop DeepSeek tags and junk
    md_text = deepseek_clean(md_text)

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-mpnet-base-v2"
        )
        # For chunking we only need offsets, not model forward.
        # Bump max length to avoid warnings on long docs.
        tokenizer.model_max_length = 10000000

    # 1. Split cleaned text into pages
    pages = _split_into_pages(md_text)
    if not pages:
        return {ws: [] for ws in token_window_sizes}

    # 2. Build global text and page spans
    global_text, page_spans = _build_global_text_and_spans(pages)
    if not global_text:
        return {ws: [] for ws in token_window_sizes}

    # 3. Tokenize once with offsets
    enc = tokenizer(
        global_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    offsets = enc["offset_mapping"]
    num_tokens = len(offsets)

    results: Dict[int, List[Dict[str, Any]]] = {}

    if not token_window_sizes:
        return results

    smallest_window = min(token_window_sizes)

    # 4. Chunk at different granularities
    for window_size in token_window_sizes:
        chunks: List[Dict[str, Any]] = []
        chunk_id = 0
        tok_start = 0

        while tok_start < num_tokens:
            tok_end = min(tok_start + window_size, num_tokens)
            remaining_tokens = num_tokens - tok_start
            token_count = tok_end - tok_start

            # Drop tiny tail chunk globally
            if token_count < min_tokens:
                break

            # Avoid duplicated tails across granularities:
            # if the remaining sequence is shorter than the smallest window,
            # only the smallest granularity should create a tail chunk.
            if window_size > smallest_window and remaining_tokens < smallest_window:
                break

            char_start = offsets[tok_start][0]
            char_end = offsets[tok_end - 1][1]
            text_span = global_text[char_start:char_end]

            # Which pages intersect this char span
            touched_pages: List[int] = []
            for p_start, p_end, p_no in page_spans:
                if not (char_end <= p_start or char_start >= p_end):
                    touched_pages.append(p_no)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "granularity": window_size,
                    "text": text_span,
                    "pages": sorted(set(touched_pages)),
                }
            )

            chunk_id += 1
            tok_start = tok_end

        results[window_size] = chunks

    return results


if __name__ == "__main__":
    # test with one document
    from pathlib import Path

    FILE_CONTENT = (
        Path("data_md/md/sc_fateh_chand_1963.md").read_text().replace("\n", " ")
    )
    result = chunk_md_with_granularities(FILE_CONTENT)
    print("Example of a chunk")
    print(result[384][0])
