import dspy
import dotenv
import os
from batch_eval_to_md import encode_query, faiss_search, summarize_text, load_chunks
from pathlib import Path
from typing import List, Dict, Any, Optional
from index import LegalIndexer

DB_PATH = "/scratch/akshit.kumar/indices/legal_index.db"
FAISS_PATH = "/scratch/akshit.kumar/indices/faiss.index"

indexer = LegalIndexer.load_from_existing(DB_PATH, FAISS_PATH)

dotenv.load_dotenv()
lm = dspy.LM("groq/openai/gpt-oss-120b")
dspy.configure(lm=lm)


def search_bm25(qtext: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Keyword search over doc level metadata.  Returns doc_id and
    extracted metadata (includes case_title, decision_date,
    statutes_mentioned, court, parties, topics, summary, sections).
    """
    return indexer.search_bm25(qtext, top_k)


def search_faiss(qtext: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Semantic search over document chunks.
    Returns doc_id and pages and chunk text"""
    return indexer.search_faiss(qtext, top_k)


def get_doc_metadata(doc_id: str) -> Dict[str, Any]:
    """Get document metadata"""
    return indexer.get_doc_metadata(doc_id)


def get_doc_content(doc_id: str, pages: Optional[List[int]] = None) -> str:
    """
    Load markdown for doc_id.
    If pages is given, return only those pages based on DeepSeek page markers.
    Pages is a list of exact pages to be retrieved
    """
    return indexer.get_doc_content(doc_id, pages)


class CaseStudyAgent(dspy.Signature):
    """You are a RAG search agent that helps search for the
    appropriate chunks useful to create a case study on the given
    query. You may try different methods of refining the query to
    finally get the best documents from the RAG index which has
    thousands of chunks from different court documents related to
    contracts and agreements. You are tasked with creating a
    comprehensive case study in markdown format using multiple exemplar documents
    and cite your sources whenever appropriate.

    """

    query: str = dspy.InputField()
    final_chunk_ids: List[str] = dspy.OutputField()
    case_study_md: str = dspy.OutputField()


QUERY = "What are the financial repercussions for the Concessionaire if they delay in fulfilling their Conditions Precedent, and what is the ultimate consequence if this delay continues until the maximum penalty is reached?"
agent = dspy.ReAct(
    CaseStudyAgent, tools=[get_doc_metadata, search_faiss, search_bm25, get_doc_content]
)
result = agent(query=QUERY)
print(result)
