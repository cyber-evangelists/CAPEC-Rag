from typing import List, Dict, Any
import re
from loguru import logger


def match_file_names(filename, database_files):
    if filename in database_files:
        return filename
    else:
        return ""


def find_file_names(query: str, database_files: List) -> str:

    pattern = r"\b(\w+\.\w+)\b(?=\s*(?:file|$|\s))"

    match = re.search(pattern, query)
    if match:
        filename = match.group(1)
        logger.info(f"File name {filename}")
        matched_file = match_file_names(filename, database_files)
        logger.info(f"matched file {matched_file}")
        if matched_file:
            logger.info(f"Extracted filename: {filename}")
            return matched_file
        else:
            return ""
    else:
        logger.info("No filename found.")
        return ""


