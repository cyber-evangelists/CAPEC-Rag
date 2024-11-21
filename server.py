from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from loguru import logger
from src.utils.utils import find_file_names
from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter

from typing import Dict, Any, List, Optional

from src.config.config import Config
from src.qdrant.qdrant_utils import QdrantWrapper
from src.embedder.embedder_llama_index import EmbeddingWrapper
from src.parser.csv_parser import CsvParser
from llama_index.core import Settings
Settings.llm = None

from src.utils.connections_manager import ConnectionManager
from src.chatbot.rag_chat_bot import RAGChatBot
from src.reranker.re_ranking import RerankDocuments

app = FastAPI()

chatbot = RAGChatBot()
file_processor = CsvParser(data_dir = Config.DATA_DIRECTORY)

collection_name = Config.COLLECTION_NAME
qdrant_client = QdrantWrapper()
embedding_client = EmbeddingWrapper()


try:

    processed_chunks = file_processor.process_directory()
    qdrant_client.ingest_embeddings(processed_chunks)

    logger.info("Successfully ingested Data")

except Exception as e:
    logger.error(f"Error in data ingestion: {str(e)}")

reranker = RerankDocuments()

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

        query_embeddings = embedding_client.generate_embeddings(query)

        top_5_results = qdrant_client.search(query_embeddings, 5)
        logger.info("Retrieved top 5 results")

        if not top_5_results:
            logger.warning("No results found in database")
            await websocket.send_json({
                "result": "The database is empty. Please ingest some data first before searching."
            })
            return
        

        reranked_docs = reranker.rerank_docs(query, top_5_results)
        reranked_top_5_list = [item['content'] for item in reranked_docs]

        context = reranked_top_5_list[:2]

        # only top 2 documents are passing as a context
        response, conversation_id  = chatbot.chat(query, context)

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
        

