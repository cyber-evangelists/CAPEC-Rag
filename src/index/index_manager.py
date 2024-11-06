from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.storage.vector_store import SimpleVectorStore
import os
import pickle
from datetime import datetime
import json

class IndexManager:
    """Manager class for index persistence operations"""
    
    def __init__(self, index, base_path: str = "indexes"):
        self.index = index
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def save_index_simple(self, index_name: str = "default"):
        """
        Simple method to save index to disk
        """
        persist_dir = os.path.join(self.base_path, index_name)
        self.index.storage_context.persist(persist_dir=persist_dir)
        
        # Save metadata
        metadata = {
            "saved_at": datetime.now().isoformat(),
            "index_name": index_name,
            "num_nodes": len(self.index.docstore.docs)
        }
        
        with open(os.path.join(persist_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        return persist_dir
    
    def load_index_simple(self, index_name: str = "default"):
        """
        Simple method to load index from disk
        """
        persist_dir = os.path.join(self.base_path, index_name)
        
        # Load storage context
        storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
        
        # Rebuild index from storage
        loaded_index = load_index_from_storage(storage_context)
        return loaded_index