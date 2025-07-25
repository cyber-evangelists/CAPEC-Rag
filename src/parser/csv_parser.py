
import pandas as pd
from typing import List, Dict, Any, Optional, TypedDict
from pathlib import Path
import numpy as np
from src.embedder.embedder import EmbeddingWrapper

from datetime import datetime
from dataclasses import dataclass
from loguru import logger
from src.config.config import Config


@dataclass
class DocumentMetadata(TypedDict):
    """Class to store document metadata"""
    source_file: str
    ingestion_timestamp: str
    last_updated_timestamp: str
    embedding_version: str
    embedding_model_name: str
    processing_status: str



class ProcessedChunk(TypedDict):
    """Type definition for processed file chunks."""
    embeddings: List[float]
    text: str
    metadata: str



class CsvParser:

    def __init__(self, data_dir: str, embedding_version: str =  Config.EMBEDDING_VERSION_NUMBER, embedding_model_name: str = Config.EMBEDDING_MODEL) -> None:
        self.data_dir = Path(data_dir)
        self.embedding_version = embedding_version
        self.embedding_model_name = embedding_model_name
        self.embedder = EmbeddingWrapper()
        self.chunks: List[ProcessedChunk] = []

    def create_document_metadata(self, row: pd.Series, file_name: str,) -> DocumentMetadata:
        """Create comprehensive document metadata"""
        current_time = datetime.now().isoformat()
        
        return DocumentMetadata(
            source_file=str(file_name),
            ingestion_timestamp=current_time,
            last_updated_timestamp=current_time,
            embedding_version=self.embedding_version,
            embedding_model_name=self.embedding_model_name,  # In practice, this might be different
            processing_status="processed",
        )



    def read_file(self, file_path: Path) -> pd.DataFrame:
        df = pd.read_csv(file_path, 
                sep=',',
                encoding='utf-8',
                skipinitialspace=True, index_col=None)
        
        df.columns = df.columns.map(lambda x: x.strip("'\"")) 
        df_reset = df.reset_index(drop=False)

        col_names = df.columns

        df.columns = col_names

        df = df_reset.iloc[:, :-1]

        df.columns = col_names
        
        return df


    def process_file(self, file_path: Path) -> None:
        """Process a single CSV file with enhanced metadata and version control"""
        try:
            logger.info(f"Processing file: {file_path}")
            
            # Read CSV file
            df = self.read_file(file_path)
                        
            for _, row in df.iterrows():
                # Combine text fields
                text_content = self.get_text(row)
                
                # Create comprehensive metadata
                metadata = self.create_document_metadata(row, file_path.name)
                embeddings = self.embedder.generate_embeddings(text_content)


                # Create Document object with enhanced metadata
                doc : ProcessedChunk = {
                    "embeddings": embeddings,
                    "text":text_content,
                    "metadata":"metadata"
                }


                self.chunks.append(doc)
                            
            logger.info(f"Successfully processed all documents from {file_path}")

            
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
        for col, value in row.items():  # Change here to access both col and value
            cleaned_text = str(value).strip() if pd.notna(value) else ""
            if cleaned_text:  # Only include non-empty values
                text_parts.append(f"{col}: {cleaned_text}")

        # Join all parts with a separator
        return " | ".join(text_parts)


    def process_directory(self) -> List[Dict[str, Any]]:
        """Process all CSV files in directory"""
        all_documents = []
        
        logger.info("Attempting to read all .csv files and indexing....")
        for file_path in self.data_dir.glob('*.csv'):
            try:
                self.process_file(file_path)
            except Exception as e:
                logger.error(f"Skipping file {file_path} due to error: {str(e)}")
                continue
            
        
        logger.info("All .csv files processed. Returning chunks...")
        return self.chunks