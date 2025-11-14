import dspy
import dotenv
import os
from batch_eval_to_md import encode_query, faiss_search, summarize_text, load_chunks
from pathlib import Path
from typing import List


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
        mpnet_snips.append("[Chunk ID : " + str(cid) + "] : " + summarize_text(chunk.get("text_original", ""), 1000))
    # return "\n\n".join(mpnet_snips)
    return mpnet_snips


class GenerateCaseStudy(dspy.Signature):
    """Generate a case study for the given query using the chunks fetched from the RAG system and cite every chunk used wherever applicable with its ID in the format [Chunk ID : <ID>]."""

    query:  str = dspy.InputField()
    chunks: List[str] = dspy.InputField()


    case_study: str = dspy.OutputField(
        desc="Comprehensive case study on the given query using the provided context"
    )
    citations: List[str] = dspy.OutputField(
        desc="Chunk IDs cited in the case study"
    )

case_study_module = dspy.ChainOfThought(GenerateCaseStudy)

class RAGReranker(dspy.Signature):
    """Rerank the chunks fetched from the RAG system for their relevance to the given query."""

    query:  str = dspy.InputField()
    chunks: List[str] = dspy.InputField()
    task: str = dspy.InputField()

    is_relevant_list: List[bool] = dspy.OutputField(
        desc="If the chunks fetched are relevant to the query"
    )
    refined_query: str = dspy.OutputField(
        desc="Refined query if the older query didnt retrieve relevant chunks"
    )

rerank_module = dspy.ChainOfThought(RAGReranker)


def search_rag(query: str):
    retrieved_chunks = get_docs(query)

    print("Chunks fetched:\n", retrieved_chunks)
    print('\n\n')
    # return result.is_relevant, result.refined_query, chunks, result.case_study, result.citations

    return retrieved_chunks


def looped_search_rag(query: str, max_tries=5):
    # find the first set of documents retrieved by RAG
    chunks = search_rag(query)

    # check how many are relevant using the reranker
    
    reranking_result = rerank_module(query=query, chunks=chunks, task="Determine if the given chunks are relevant to the query for generating a case study related to the query. If not, refine the query to fetch better chunks.")
    is_relevant_list, refined_query = reranking_result.is_relevant_list, reranking_result.refined_query
    print("Relevant List:", is_relevant_list)
    
    # function to get only relevant chunks
    refined_chunks = lambda chunks, is_relevant_list: [chunks[i] for i in range(len(chunks)) if is_relevant_list[i]]
    tries = 0
    queries = []

    # until we have 5 or more relevant chunks /  we have tried 5 times
    while sum(is_relevant_list) < 5 or tries == 5:
        print(refined_query)
        tries += 1

        # add the new chunks fetched to the existing relevant ones
        chunks.extend(search_rag(refined_query))
        chunks = list(set(chunks))  # deduplicate
        
        # rerank again (previous relevant + new RAG retrieved chunks)
        reranking_result = rerank_module(query=refined_query, chunks=chunks, task="Determine if the given chunks are relevant to the query for generating a case study related to the query. If not, refine the query to fetch better chunks.")
        is_relevant_list, refined_query = reranking_result.is_relevant_list, reranking_result.refined_query
        print("Relevant List:", is_relevant_list)

        # only keep the relevant chunks
        chunks = refined_chunks(chunks, is_relevant_list)

        queries.append(refined_query)
    
    final_revelant_chunks = refined_chunks(chunks, is_relevant_list)
    case_study_outputs = case_study_module(query=query, chunks=final_revelant_chunks)
    case_study_reasoning, case_study, citations = case_study_outputs.reasoning, case_study_outputs.case_study, case_study_outputs.citations

    return case_study_reasoning, case_study, final_revelant_chunks, citations


# for testing
if __name__ == "__main__":
    # query = "When can a penalty or liquidated damages clause be reduced by courts under Section 74 of the Indian Contract Act?"
    query = "When a Non-Disclosure Agreement (NDA) genuinely protects trade secrets versus when it functions as legal cosplay."

    print(
        looped_search_rag(query)
    )
