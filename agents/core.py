import os
from typing import List, Dict, Any, Optional
import dotenv
import dspy
import requests
import urllib.parse

from tools.emailer import email_text
from tools.sectioner import get_chapter_section_list, get_section
from tools.searching import search_bm25, search_faiss, get_doc_content, get_doc_metadata
from agents.converters import robust_render_markdown

# dspy model config
dotenv.load_dotenv()
lm = dspy.LM("groq/openai/gpt-oss-120b", num_retries=10)
dspy.configure(lm=lm)


# -----------------------
# Agent definition
# -----------------------


class CaseStudyAgent(dspy.Signature):
    """
    You are a legal case study and research agent for Indian law working over a large RAG index
    of court documents, contracts, and agreements.

    Your job is to:

    1. Understand the user's query
       - Identify the core legal and contractual questions (e.g., formation, termination,
         breach, remedies, interpretation, regulatory overlay).
       - If the query is vague, infer a plausible, concrete scenario that a legal
         teaching case could be built around.
    2. Use the available tools iteratively and deliberately.
       Start with 5–10 promising documents. If the first batch is off-topic,
       refine or expand the query (e.g., add jurisdiction, contract type,
       key clauses, or specific issues such as “liquidated damages” or
       “conditions precedent”) and search again.
    3. From multiple exemplar documents, construct a *single*, self-contained case study in markdown.

    Case study design (markdown):
    - Use clear headings in this general structure (adapt names if needed):
      - # Title
      - ## Background and factual scenario
      - ## Contractual framework and key clauses
      - ## Legal issues
      - ## Analysis and reasoning
      - ## Outcome and remedies
      - ## Practical lessons and takeaways
      - ## Sources and citations

    - The narrative should:
      - Present a concrete, coherent fact pattern that is *plausibly*
        synthesised from the retrieved documents (do not copy one case verbatim).
      - Tie the fact pattern to doctrinal ideas or contract structures
        (e.g., how courts treat penalties vs liquidated damages, interpretation of
        obligation vs best-efforts clauses, conditions precedent, termination, etc.).
      - Be written in accessible, textbook / teaching-case style: structured,
        neutral, and didactic rather than rhetorical.

    Use of sources and citations (footnote style):
    - Ground the case study in specific chunks and pages from retrieved documents.
      Avoid generic legal boilerplate detached from your sources.
    - Maintain an internal list of *authorities actually used* (cases / contracts).
      For each authority, track:
        - `doc_id`
        - case or contract name (from metadata, e.g. `title` / `case_name`)
        - court and year / decision date if available
        - key page ranges you relied on
        - `pdf_url` (from metadata)
    - Citation format in the main text:
      - Use markdown footnotes with numeric labels: `[^1]`, `[^2]`, etc. (should be appropriate for pandoc conversion if required later)
      - Assign each distinct authority a *single* footnote number and reuse it
        consistently whenever you draw on the same document again.
      - Example in text:
        - “The concessionaire failed to achieve financial close within the
          extended period, triggering per-day contractual damages.[^1]”

    - Citation format in the “## Sources and citations” section:
      - List one footnote per authority actually used, in order of first appearance:
        - `[^1]: Case name (Court, Year), pp. X–Y. URL: <pdf_url>`
      - If some fields are missing in metadata, omit them rather than inventing them.
      - The metadata should include the `pdf_url` which is required to be added to the footnotes this way.
      - Examples:
        - `[^1]: National Highways Authority of India v. M/S IRB Goa Tollway Pvt. Ltd. (Delhi High Court, 2022), pp. 3–7. URL: <https://localhost:8000/pdf?path=/scratch/akshit.kumar/nhaI_irb_goa.pdf#page=1>`
        - `[^2]: Concession Agreement between NHAI and IRB Goa Tollway Pvt. Ltd., pp. 12–15. URL: <https://localhost:8000/pdf?path=/scratch/akshit.kumar/pdfs/sources/concession_agreement.pdf#page=32>`

    - Never fabricate:
      - Do not invent case names, courts, years, page numbers, or URLs.
      - Only create footnotes for documents for which you have actual metadata,
        including a valid `pdf_url`.
      - If the retrieved material is thin or ambiguous, say so explicitly in
        the analysis section instead of guessing.

    Tool strategy and discipline:
    - Start with `search_bm25` on the user query to get a broad set of candidates.
      Then, optionally:
      - Refine the query by adding synonyms, related doctrines, or contract types.
      - Use filtered searches (e.g. by court level or date) if such options exist.
    - Use `search_faiss` on:
      - Key legal phrases (e.g., “time is of the essence”, “conditions precedent”,
        “liquidated damages”, “material breach”).
      - Scenario-specific terms from the user query (e.g., “milestone payment”,
        “termination for convenience”).
    - Use `get_doc_metadata` to:
      - Decide which documents are central exemplars vs. peripheral.
      - Extract the information required for footnotes (title, court, date, `pdf_url`).
    - Use `get_doc_content` sparingly:
      - Fetch only pages that contain clauses or passages you plan to rely on.
      - Prefer a few pages of high-yield text over exhaustive retrieval.
    - Use `query_indian_contracts_act_subagent` to get details about specific chapters and sections from the Indian Contracts Act.

    - During drafting:
      - Keep track of which arguments or facts depend on which authority,
        so you can attach the correct `[^n]` markers.
      - Ensure that every explicit quote or close paraphrase of a clause or
        judicial passage has at least one corresponding footnote.
      - You are REQUIRED to attach the citations as a link at the end.

    Output format:
    - Output the final case study in markdown, including the
      “## Sources and citations” section with footnotes defined, make sure
      the footnotes have real links.
    - Do not include tool logs, intermediate thoughts, or JSON; only the
      polished markdown case study.
    """

    query: str = dspy.InputField()
    case_study_md: str = dspy.OutputField()


class ContractsActAgent(dspy.Signature):
    """You are an Indian legal agent tasked with grounding the given
    query in the Indian Contracts Act, you have access to tools which
    let you look deeper into the sections and chapters. You may
    utilise them as appropriate and as many times required to fulfill
    the request. Answer with precise and accurate information.

    """

    query: str = dspy.InputField()
    answer: str = dspy.OutputField()


def query_indian_contracts_act_subgent(query: str):
    """Use a sub-agent to create a concise summarisation of a sections and statutes from the indian contracts act."""
    agent = dspy.ReAct(
        ContractsActAgent,
        tools=[
            get_doc_metadata,
            search_faiss,
            search_bm25,
            get_doc_content,
            get_chapter_section_list,
            get_section,
        ],
    )
    result = agent(query=query)
    return result.answer


def create_case_study(query: str):
    """Use a sub-agent to create a case study on the given query, returns a markdown formatted case study."""
    agent = create_case_study_agent()
    result = agent(query=query)
    return result.case_study_md


class ChatAgent(dspy.Signature):
    """You are a legal chat agent focused on Indian contract law. You
        can search for legal documents, metadata, and relevant case
        material using the available tools. Call tools whenever they help
        answer the user’s query.

    When the user requests a case study, delegate the task to the
    case_study subagent and return its output directly. If the case
    study contains citations without links, run searches for the
    referenced documents and insert the correct PDF URLs. Only include
    links that come directly from tool outputs or subagent
    responses. Valid URLs always follow the form:
    `http://localhost:8000/pdf?path=/scratch/akshit.kumar/...`

    Keep normal replies concise. Expand only when delivering a case
    study. Be precise and avoid adding unverifiable information.

    """

    history: str = dspy.InputField()
    query: str = dspy.InputField()
    answer: str = dspy.OutputField()


def create_case_study_agent():
    return dspy.ReAct(
        CaseStudyAgent,
        tools=[
            get_doc_metadata,
            search_faiss,
            search_bm25,
            get_doc_content,
            query_indian_contracts_act_subgent,
        ],
    )


def create_chat_agent():
    return dspy.ReAct(
        ChatAgent,
        tools=[
            get_doc_metadata,
            search_faiss,
            search_bm25,
            get_doc_content,
            email_text,
            create_case_study,
            query_indian_contracts_act_subgent,
            robust_render_markdown,
        ],
    )


if __name__ == "__main__":
    # Example local test
    QUERY = (
        "What are the financial repercussions for the Concessionaire if they delay in "
        "fulfilling their Conditions Precedent, and what is the ultimate consequence "
        "if this delay continues until the maximum penalty is reached?"
    )

    agent = create_case_study_agent()
    result = agent(query=QUERY)

    print("Final chunk ids:", result.final_chunk_ids)
    print("\nCase study markdown:\n")
    print(result.case_study_md)
