# core_agents.py

from dataclasses import dataclass
from typing import List, Callable, Optional
from agents.core import (
    search_rag,
    looped_search_rag,
    # import your other agent functions as they get added
)

@dataclass
class MetaResult:
    final_answer: Optional[str] = None
    pdf_bytes: Optional[bytes] = None
    reasoning: Optional[str] = None
    rag_citations: Optional[List[str]] = None
    web_links: Optional[List[str]] = None


def run_meta_agent(
    query: str,
    mode: str,
    status_callback: Optional[Callable[[str], None]] = None
) -> MetaResult:
    
    def update(msg):
        if status_callback:
            status_callback(msg)

    update(f"Mode selected: **{mode}**")

    # ----------------------
    # MODE: GENERATE CASE STUDY
    # ----------------------
    if mode == "Generate Case Study":
        update("Running RAG retrieval…")
        reasoning, case_study, chunks, citations = looped_search_rag(query)

        update("Case study generated.")
        return MetaResult(
            final_answer=case_study,
            reasoning=reasoning,
            rag_citations=citations
        )

    # ----------------------
    # MODE: SUMMARISE STATUTES
    # ----------------------
    elif mode == "Summarise Statutes":
        update("Retrieving statute context…")

        # TODO: call your statute summarizer agent
        # For now:
        summary = f"(mock) Summary of statutes for: {query}"

        return MetaResult(
            final_answer=summary,
            reasoning="(mock reasoning for statute summarization)"
        )

    # ----------------------
    # MODE: COURT STYLE DOCUMENT
    # ----------------------
    elif mode == "Generate Court Style Document":
        update("Generating court-style document…")

        # TODO: call your document generation agent + PDF generator
        # For now:
        mock_doc = "(mock court document text)"
        pdf_bytes = None  # Or generate a real PDF

        return MetaResult(
            final_answer=mock_doc,
            pdf_bytes=pdf_bytes,
            reasoning="(mock reasoning for court document)"
        )

    # ----------------------
    # MODE: GENERIC
    # ----------------------
    else:
        update("Running generic LLM pipeline…")

        # TODO: call your generic LLM agent
        generic_answer = f"(mock generic answer for: {query})"

        return MetaResult(
            final_answer=generic_answer,
            reasoning="(mock generic reasoning)"
        )
