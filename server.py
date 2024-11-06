from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from loguru import logger


import asyncio
from typing import Dict, Any, List, Optional

from src.config.config import Config
from src.embedder.embedder_llama_index import EmbeddingWrapper
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings
from src.parser.csv_parser import CsvParser
Settings.llm = None

from src.llm.groqwrapper import GroqWrapper
from loguru import logger

from src.llm.groqwrapper import GroqWrapper
from src.qdrant.qdrant_manager import QdrantManager
from src.parser.threatmon_parser import FileProcessor
from src.utils.utils import prepare_prompt, rerank_docs
from src.utils.connections_manager import ConnectionManager
from src.config.config import Config

import os

app = FastAPI()

# Initialize all clients/wrappers
groq_client = GroqWrapper()
file_processor = FileProcessor()

collection_name = Config.COLLECTION_NAME
qdrantManager = QdrantManager(Config.QDRANT_HOST, Config.QDRANT_PORT, collection_name)

embedding_client = EmbeddingWrapper()

index = qdrantManager.load_index(persist_dir=Config.PERSIST_DIR, embed_model=embedding_client)

retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5
        )


# Create the connection manager instance
connection_manager = ConnectionManager(max_connections=Config.MAX_CONNECTIONS)
connections: Dict[WebSocket, Dict[str, Any]] = {}


async def handle_search(websocket: WebSocket, query: str) -> None:
    """
    Handle search action with proper error handling.

    Args:
        websocket (WebSocket): The WebSocket connection to send responses.
        query (str): The search query string.

    Returns:
        None: Responses are sent through the WebSocket connection.

    Raises:
        Exception: Any unexpected errors during the search process.
    """
    try:
        logger.info(f"Processing search query: {query}")

        # Generate embeddings
        logger.info("Retrieving Relevant nodes")
        relevant_nodes = retriever.retrieve(query)

        context = [node.text for node in relevant_nodes]

        # Only attaching top 2 results
        prompt = prepare_prompt(query, context[:2])

        # Generate response using Groq
        logger.info("Generating response from Groq")
        response = groq_client.get_response(prompt)

        # response = "Answer"

        await websocket.send_json({
            "result": response
        })

    except Exception as e:
        logger.error(f"Error in search handling: {str(e)}")
        await websocket.send_json({
            "error": f"Search failed: {str(e)}"
        })


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Handle WebSocket connections and route messages to appropriate handlers.

    Args:
        websocket (WebSocket): The WebSocket connection.

    Returns:
        None
    """
    if not await connection_manager.connect(websocket):
        return

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            payload = data.get("payload")

            if action == "pong":
                continue  # Handle heartbeat response

            if not action:
                await websocket.send_json({"error": "No action specified"})
                continue

            if action == "search":
                await handle_search(websocket, payload["query"])
            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        pass
        
