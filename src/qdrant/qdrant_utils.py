import time
from typing import List, Dict, Any, Optional

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    FilterSelector,
    CollectionInfo
)


class QdrantWrapper:
    """A wrapper class for Qdrant vector database operations."""

    def __init__(self, collection_name: str = "threadmon-reports-ioc") -> None:
        """
        Initialize the QdrantWrapper with connection settings.

        Args:
            collection_name (str): Name of the collection to use.
        """
        self.host = "qdrant"
        self.port = 6333
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        self.client: Optional[QdrantClient] = None
        self.collection_name = collection_name
        self._connect_with_retry()

    def _connect_with_retry(self) -> None:
        """
        Establish connection to Qdrant with retry logic.

        Raises:
            Exception: If connection fails after maximum retries.
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(
                    f"Attempting to connect to Qdrant at {self.host}:{self.port} "
                    f"(Attempt {attempt + 1}/{self.max_retries})"
                )
                self.client = QdrantClient(
                    host=self.host,
                    port=self.port,
                    timeout=60  # Increased timeout for stability
                )
                # Test connection by calling an API
                self.client.get_collections()
                logger.info("Successfully connected to Qdrant")
                self._create_collection_if_not_exists()
                break
            except Exception as e:
                logger.error(f"Connection attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    raise Exception(
                        f"Failed to connect to Qdrant after {self.max_retries} attempts"
                    )

    def _create_collection_if_not_exists(self) -> None:
        """
        Create the collection if it doesn't exist.

        Raises:
            Exception: If collection creation fails.
        """
        collections = self.client.get_collections().collections
        if not any(collection.name == self.collection_name for collection in collections):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info("Collection is Created")
        else:
            logger.info("Collection already exists")

    def clear_collection(self) -> None:
        """
        Delete all vectors/points from the collection while keeping the structure.

        Raises:
            Exception: If clearing collection fails.
        """
        try:
            if self.client:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=FilterSelector(filter=None)
                )
                logger.info(
                    f"Successfully cleared all vectors from collection: "
                    f"{self.collection_name}"
                )
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            raise

    def ingest_embeddings(self, docs: List[Dict[str, Any]]) -> None:
        """
        Ingest document embeddings into the collection.

        Args:
            docs (List[Dict[str, Any]]): List of documents with embeddings to ingest.

        Raises:
            Exception: If ingestion fails.
        """
        points = [
            PointStruct(
                id=i,
                vector=doc["embeddings"],
                payload={"text": doc["text"], "document": doc["document"]}
            )
            for i, doc in enumerate(docs)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(
        self,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict[str, str]]:
        """
        Search vectors in Qdrant collection with empty collection check.

        Args:
            query_vector (List[float]): Vector to search for.
            limit (int): Number of results to return.

        Returns:
            List[Dict[str, str]]: List of search results containing documents and content.

        Raises:
            ValueError: If collection is empty or doesn't exist.
            Exception: For other search-related errors.
        """
        try:
            # Get collection info to check if it's empty
            collection_info = self.client.get_collection(self.collection_name)

            # Check if collection exists and has points
            if collection_info.points_count == 0:
                logger.warning(f"Collection '{self.collection_name}' is empty")
                raise ValueError(
                    f"The collection '{self.collection_name}' is empty. "
                    "Please ingest data first."
                )


            # Perform search if collection has data
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )

            return [
                {
                    "document": hit.payload["document"],
                    "content": hit.payload["text"]
                }
                for hit in search_result
            ]

        except Exception as e:
            if "Collection not found" in str(e):
                raise ValueError(
                    f"Collection '{self.collection_name}' does not exist. "
                    "Please create it first."
                )
            raise