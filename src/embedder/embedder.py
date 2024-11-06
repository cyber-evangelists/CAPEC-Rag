import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

EMBEDDING_MODEL_PATH = "src/embedder/model/"

class EmbeddingWrapper:
    def __init__(self, model_path=EMBEDDING_MODEL_PATH):
        self.model = SentenceTransformer(model_path)
        logger.info("Embedding model loaded.")
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list): A list of strings to generate embeddings for.
        
        Returns:
            numpy.ndarray: A 2D array of embeddings, where each row corresponds to a text input.
        """
        embeddings = self.model.encode(texts)
        return np.array(embeddings)

