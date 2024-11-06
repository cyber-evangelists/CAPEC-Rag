
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from llama_index.core.schema import Document
from llama_index.core.node_parser import SentenceSplitter

from datetime import datetime
from dataclasses import dataclass


from loguru import logger

from src.config.config import Config


@dataclass
class DocumentMetadata:
    """Class to store document metadata"""
    source_file: str
    ingestion_timestamp: str
    last_updated_timestamp: str
    embedding_version: str
    embedding_model: str
    processing_status: str


class CsvParser:

    def __init__(self, data_dir: str, embedding_version: str =  Config.EMBEDDING_VERSION_NUMBER, embedding_model: str = Config.EMBEDDING_MODEL) -> None:
        self.data_dir = Path(data_dir)
        self.embedding_version = embedding_version
        self.embedding_model = embedding_model
        self.node_parser = SentenceSplitter(chunk_size=1200, chunk_overlap=200)
        

    def create_document_metadata(self, row: pd.Series, file_name: str,) -> DocumentMetadata:
        """Create comprehensive document metadata"""
        current_time = datetime.now().isoformat()
        
        return DocumentMetadata(
            source_file=str(file_name),
            ingestion_timestamp=current_time,
            last_updated_timestamp=current_time,
            embedding_version=self.embedding_version,
            embedding_model=self.embedding_model,  # In practice, this might be different
            processing_status="processed",
        )


    def process_file(self, file_path: Path) -> List[Document]:
        """Process a single CSV file with enhanced metadata and version control"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
                        
            documents = []
            for _, row in df.iterrows():
                # Combine text fields
                text_content = self.get_text(row)
                
                # Create comprehensive metadata
                metadata = self.create_document_metadata(row, file_path.name)
                
                # Create Document object with enhanced metadata
                doc = Document(
                    text=text_content,
                    metadata=metadata.__dict__
                )

                nodes = self.node_parser.get_nodes_from_documents([doc])
                documents.extend(
                    [Document(text=node.text, metadata=node.metadata) for node in nodes]
                )
                            
            logger.info(f"Successfully processed {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            raise



    def get_text(self, row: pd.Series) -> str:
        """
        Combine all columns into text field, excluding null values and cleaning the text.
        
        Args:
            row: pandas Series containing row data
            
        Returns:
            str: Combined text from all columns
        """
        text_parts = []
        
        # Process each column in the row
        for col in row.index:
            cleaned_text = str(row[col]).strip() if pd.notna(row[col]) else ""
            if cleaned_text:  # Only include non-empty values
                text_parts.append(f"{col}: {cleaned_text}")
        
        # Join all parts with a separator
        return " | ".join(text_parts)


    def process_directory(self) -> List[Document]:
        """Process all CSV files in directory"""
        all_documents = []
        
        logger.info("Attempting to read all .csv files and indexing....")
        for file_path in self.data_dir.glob('*.csv'):
            try:
                documents = self.process_file(file_path)
                all_documents.extend(documents)
            except Exception as e:
                logger.error(f"Skipping file {file_path} due to error: {str(e)}")
                continue
        
        logger.info("All .csv files indexed....")
        return all_documents