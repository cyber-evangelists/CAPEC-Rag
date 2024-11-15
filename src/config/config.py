
from dotenv import load_dotenv
import os

load_dotenv()

class Config:

    MODEL_NAME = "llama3-8b-8192"
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL_NAME = "llama-3.1-8b-instant"
    MAX_CHAT_HISTORY = 20
    GRADIO_SERVER_NAME = "0.0.0.0" 
    GRADIO_SERVER_PORT = int(7860)
    WEBSOCKET_URI = "ws://rag-server:8000/ws"
    DATA_DIRECTORY = "data/"
    WEBSOCKET_TIMEOUT = 300  # 5 minutes
    HEARTBEAT_INTERVAL = 30  # 30 seconds
    MAX_CONNECTIONS = 100

    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
    
    COLLECTION_NAME = "capec-collection-v1"

    EMBEDDING_VERSION_NUMBER = "v1.0"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    CAPEC_DATA_DIR = "./capec-dataset/"
    PERSIST_DIR = "/app/src/index/index/"

    QDRANT_HOST = "qdrant"
    QDRANT_PORT = 6333

    EMBEDDING_MODEL_PATH = "./src/embedder/embedding_model/"
    RERANKING_MODEL_PATH = "./src/reranker/re_ranker_model/"


