import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.config import Config


class EmbeddingWrapper:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL_PATH)
    
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
