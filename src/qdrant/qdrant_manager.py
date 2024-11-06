from typing import List, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import VectorParams
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage
from src.config.config import Config
import os
from loguru import logger

class QdrantManager:
    """ A class to manage the interaction with the Qdrant vector database. """

    def __init__(self, host: str, port: int, collection_name: str) -> None:
        self.host = host
        self.port = port
        self.collection_name = collection_name
        logger.info(f"Initializing QdrantManager with host: {self.host}, port: {self.port}, collection: {self.collection_name}")
        self.client = self._connect_to_qdrant()

    def _connect_to_qdrant(self) -> QdrantClient:
        """ Connects to the Qdrant server and returns the client instance. """
        try:
            client = QdrantClient(path=Config.PERSIST_DIR) # Connecting to the Dockerized Qdrant instance
            logger.info("Connected to Qdrant successfully.")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {str(e)}")
            raise

    def ensure_collection_exists(self) -> None:
        """ Ensures that a collection with the specified name exists in Qdrant. """
        try:
            collections = self.client.get_collections().collections
            logger.info(f"Available collections: {[collection.name for collection in collections]}")
            if not any(collection.name == self.collection_name for collection in collections):
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=384, distance=rest.Distance.COSINE)
                )
                logger.info(f"Created new collection: {self.collection_name}")
            else:
                logger.info(f"Collection {self.collection_name} already exists.")
        except Exception as e:
            logger.error(f"Failed to manage collection {self.collection_name}: {str(e)}")
            raise

    def initialize_vector_store(self) -> QdrantVectorStore:
        """ Initializes a Qdrant vector store for the specified collection. """
        try:
            vector_store = QdrantVectorStore(client=self.client, collection_name=self.collection_name)
            logger.info("Qdrant Vector Store initialized successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {str(e)}")
            raise

    def create_and_persist_index(self, docs: List[Any], storage_context: StorageContext, embed_model: Any, persist_dir: str) -> VectorStoreIndex:
        """ Creates an index from the given documents and persists it. """
        try:
            if not docs:
                logger.warning("No documents provided for indexing.")
                return None
            
            logger.info("Indexing documents into vector store...")
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
            index.storage_context.persist(persist_dir=persist_dir)
            logger.info("Data ingested into VectorDb and index persisted successfully.")
            return index
        except Exception as e:
            logger.error(f"Failed to create and persist index: {str(e)}")
            raise

    def load_index(self, persist_dir: str, embed_model: Optional[Any] = None) -> VectorStoreIndex:
        """ Load the index from disk. """
        try:
            self.ensure_collection_exists()
            vector_store = self.initialize_vector_store()
            
            if os.path.isdir(persist_dir):
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir, vector_store=vector_store)
                loaded_index = load_index_from_storage(storage_context=storage_context, embed_model=embed_model)
                logger.info(f"Index successfully loaded from {persist_dir}")
                return loaded_index
            else:
                logger.error(f"Persist directory not found: {persist_dir}.")
                raise FileNotFoundError(f"Persist directory not found: {persist_dir}.")
                
        except Exception as e:
            logger.error(f"An error occurred while loading index: {str(e)}")
            raise