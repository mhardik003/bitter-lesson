import dspy
import dotenv
import os
from batch_eval_to_md import encode_query, faiss_search, summarize_text, load_chunks
from pathlib import Path


MPNET_INDEX = Path(
    "data_out/index/vector/faiss_sentence-transformers-all-mpnet-base-v2_g512.index"
)
MPNET_IDS = Path(
    "data_out/index/vector/faiss_sentence-transformers-all-mpnet-base-v2_g512.ids"
)
CHUNKS_PATH = Path("data_out/chunks.jsonl")

dotenv.load_dotenv()
lm = dspy.LM("gemini/gemini-2.5-flash", api_key=os.getenv("GEMINI_KEY"))
dspy.configure(lm=lm)
id2chunk = load_chunks(CHUNKS_PATH)


def get_docs(qtext: str):
    mpnet_vec = encode_query("sentence-transformers/all-mpnet-base-v2", "cuda", qtext)
    mpnet_hits = faiss_search(MPNET_INDEX, MPNET_IDS, mpnet_vec, k=5)

    mpnet_snips = []
    for cid, _ in mpnet_hits:
        chunk = id2chunk.get(cid, {})
        mpnet_snips.append(summarize_text(chunk.get("text_original", ""), 1000))
    return "\n\n".join(mpnet_snips)


class GenerateCaseStudy(dspy.Signature):
    """Generate a case study for the given query using the documents fetched from the RAG system"""

    query: str = dspy.InputField()
    documents: str = dspy.InputField()

    is_relevant: bool = dspy.OutputField(
        desc="If the documents fetched are relevant to the query"
    )
    refined_query: str = dspy.OutputField(
        desc="Refined query if the older query didnt retrieve relevant documents"
    )
    case_study: str = dspy.OutputField(
        desc="Comprehensive case study on the given query using the provided context"
    )


case_study_module = dspy.ChainOfThought(GenerateCaseStudy)


def search_rag(query: str):
    documents = get_docs(query)
    result = case_study_module(query=query, documents=documents)
    return result.is_relevant, result.refined_query, documents, result.case_study


def looped_search_rag(query: str, max_tries=5):
    is_relevant, refined_query, documents, case_study = search_rag(query)
    tries = 0
    queries = []
    while not is_relevant or tries == 5:
        print(refined_query)
        tries += 1
        is_relevant, refined_query, documents, case_study = search_rag(refined_query)
        queries.append(refined_query)
    return case_study


# for testing
if __name__ == "__main__":
    print(
        looped_search_rag(
            "When can a penalty or liquidated damages clause be reduced by courts under Section 74 of the Indian Contract Act?"
        )
    )
