from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from src.utils.utils import find_file_names
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

from typing import Dict, Any, List, Optional

from src.config.config import Config
from src.embedder.embedder_llama_index import EmbeddingWrapper
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings
Settings.llm = None

from src.qdrant.qdrant_manager import QdrantManager
from src.utils.connections_manager import ConnectionManager
from src.chatbot.rag_chat_bot import RAGChatBot
from src.reranker.re_ranking import RerankDocuments

import os

app = FastAPI()

chatbot = RAGChatBot()

collection_name = Config.COLLECTION_NAME
qdrantManager = QdrantManager(Config.QDRANT_HOST, Config.QDRANT_PORT, collection_name)

embedding_client = EmbeddingWrapper()


data_dir = Config.CAPEC_DATA_DIR

reranker = RerankDocuments()

index = qdrantManager.load_index(persist_dir=Config.PERSIST_DIR, embed_model=embedding_client)

retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=5
        )

# Manually added file names of the CAPEC daatset. In production, These files will be fetched from database
database_files = ["333.csv", "658.csv", "659.csv", "1000.csv", "3000.csv"]

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
        logger.info(f"Processing search query")

        filename = find_file_names(query, database_files)

        if filename:
            logger.info("Searching for file names...")

            filters = MetadataFilters(filters=[ExactMatchFilter(key="source_file", value=filename)])
            relevant_nodes =  index.as_retriever(filters=filters).retrieve(query)
            if not relevant_nodes:
                logger.info("Searching without file name filter....")
                relevant_nodes = retriever.retrieve(query)
        else:
            logger.info("Searching without file names....")
            relevant_nodes = retriever.retrieve(query)


        context = [node.text for node in relevant_nodes]
    
        reranked_docs =  reranker.rerank_docs(query, context)
        
        # only top 2 documents are passing as a context
        response, conversation_id  = chatbot.chat(query, reranked_docs[:2])



        logger.info("Generating response from Groq")

        await websocket.send_json({
            "result": response
        })

    except Exception as e:
        logger.error(f"Error in search handling: {str(e)}")
        await websocket.send_json({
            "error": f"Search failed: {str(e)}"
        })

async def add_feedback(websocket: WebSocket, action:str,  comment: str) -> None:

    try:
        logger.info(f"in the add feedback function...")

        logger.info(action)
        logger.info(comment)

        chatbot.add_feedback(action, comment)

        await websocket.send_json({
            "result": "Feedback added successfully"
        })

    except Exception as e:
        logger.error(f"Error in search handling: {str(e)}")
        await websocket.send_json({
            "error": f"Feedback Addition failed: {str(e)}"
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
            elif  action == "search":
                await handle_search(websocket, payload["query"])
            elif action == "positive":
                 await add_feedback(websocket, action , payload["comment"])
            elif action == "negative":
                 await add_feedback(websocket, action , payload["comment"])
            else:
                await websocket.send_json({"error": f"Unknown action: {action}"})

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
    finally:
        connection_manager.disconnect(websocket)
        

