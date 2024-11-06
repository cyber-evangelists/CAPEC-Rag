import numpy as np
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.embeddings.base import BaseEmbedding
from src.config.config import Config
from pydantic import Field
from loguru import logger


class EmbeddingWrapper(BaseEmbedding):

    embed_model: HuggingFaceEmbedding = Field(default=None)

    def __init__(self, model_path=Config.EMBEDDING_MODEL_PATH):
        super().__init__()
        self.embed_model = HuggingFaceEmbedding(model_path)
        logger.info("Embedding model loaded.")
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list): A list of strings to generate embeddings for.
        
        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row corresponds to a text input.
        """
        embeddings = self._get_text_embedding(texts)
        return np.array(embeddings)
    

    def _get_query_embedding(self, query: str) -> list:
        """Required override: Get embeddings for a query text."""
        return self.embed_model.get_text_embedding(query)
    
    def _get_text_embedding(self, text: str) -> list:
        """Required override: Get embeddings for a text."""
        return self.embed_model.get_text_embedding(text)
    
    def _get_text_embeddings(self, texts: list[str]) -> list[list]:
        """Optional override: Get embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            embedding = self.embed_model.get_text_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    async def _aget_query_embedding(self, query: str) -> list:
        """Async version of query embedding."""
        # Since HuggingFaceEmbedding doesn't support async, we just call the sync version
        return self._get_query_embedding(query)
    
    async def _aget_text_embedding(self, text: str) -> list:
        """Async version of text embedding."""
        # Since HuggingFaceEmbedding doesn't support async, we just call the sync version
        return self._get_text_embedding(text)

