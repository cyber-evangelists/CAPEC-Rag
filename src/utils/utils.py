from typing import List, Dict, Any

# from sentence_transformers import CrossEncoder

def prepare_prompt(query: str, context: str) -> str:
    """
    Prepare a prompt by combining query and context with specific formatting.

    Args:
        query (str): The user's query string.
        context (str): The context information to include in the prompt.

    Returns:
        str: The formatted prompt string.
    """
    initial_string = """You are provided with a Context: some CAPEC dataset rows, which include attack patterns, their description, CWE IDs and related data.
Use this context to answer the following Query. Please analyze the provided Context in depth, and deliver a clear and concise answer based on the query. You HATE repetition. Just provide PRECISE answer.
    
    Query: """
    
    # Escaping special characters in query and context using repr()
    prompt = r"" + initial_string + repr(query)[1:-1] + r"\n\nContext:\n" + repr(context)[1:-1] + r"\n\nYour response:"
    
    return prompt


def rerank_docs(
    query: str,
    top_5_results: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Rerank documents based on their relevance to the query using a cross-encoder.

    Args:
        query (str): The search query.
        top_5_results (List[Dict[str, str]]): Initial top 5 search results to rerank.

    Returns:
        List[Dict[str, str]]: Reranked list of documents.
    """
    # Re-ranking using cross-encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    # Prepare pairs for reranking
    pairs = [[query, doc["content"]] for doc in top_5_results]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort by new scores
    reranked_results = [
        doc for _, doc in sorted(
            zip(scores, top_5_results),
            reverse=True
        )
    ]

    return reranked_results