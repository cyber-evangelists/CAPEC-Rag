
from typing import List, Dict, Any
from loguru import logger

from src.config.config import Config
from sentence_transformers import CrossEncoder

class RerankDocuments:

    def __init__(self, reranking_model_path: str = Config.RERANKING_MODEL_PATH) -> None:
        self.reranker = CrossEncoder(reranking_model_path)


    def rerank_docs(self,
        query: str,
        top_5_results: List[str]
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
        # Prepare pairs for reranking
        pairs = [[query, doc] for doc in top_5_results]

        # Get relevance scores
        scores = self.reranker.predict(pairs) 

        # Sort by new scores
        reranked_results = [
            doc for _, doc in sorted(
                zip(scores, top_5_results),
                reverse=True
            )
        ]

        return reranked_results

        