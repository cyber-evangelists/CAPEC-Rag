from typing import List, Dict, Any
import re
from loguru import logger

# from sentence_transformers import CrossEncoder

def prepare_prompt(query: str, context: str) -> str:
    """
    Prepare a prompt by combining query and context with specific formatting.

    Args:
        query (str): The user's query string.
        context (str): The context information to include in the prompt.

    Returns:
        str: The formatted prompt string.
    """
    initial_string = """You are a Cyber Securiry Expert. You are provided with a Context which is CAPEC dataset entries and a Query. Please analyze the provided Context in depth, and deliver a clear and concise Response based on the Query. 
The Context contains information about a dataset in the following format: column_name: value | column_name: value etc. 

Below is the explanation of the CAPEC dataset schema:

ID: Unique identifier for the capec attack pattern (CAPEC IDs).
Name: Name of the attack pattern.
Abstraction: Generalization level of the attack pattern.
Status: Current status of the attack pattern (e.g., Draft, Stable).
Description: Detailed description of the attack pattern.
Alternate Terms: Other terms used to describe the attack.
Likelihood Of Attack: Probability of the attack occurring.
Typical Severity: Expected severity of the attack impact.
Related Attack Patterns: List of attack patterns related to this one (CAPEC IDs)
Execution Flow: Steps or stages involved in carrying out the attack.
Prerequisites: Conditions required for the attack to succeed.
Skills Required: Skill level required to perform the attack.
Resources Required: Resources needed to execute the attack.
Indicators: Signs or symptoms indicating the attack may be occurring.
Consequences: Potential effects or impacts of the attack.
Mitigations: Strategies to prevent or reduce the impact of the attack.
Example Instances: Examples of real-world cases of the attack.
Related Weaknesses: Weaknesses related to this attack pattern (Common Weakness Enumeration(CWE) IDs)
Taxonomy Mappings: Mappings to external taxonomies or classifications.
Notes: Additional information or notes about the attack pattern.


Here are the steps you need to follow:

1. Understand the schema of the dataset well. 
2. You are very sensitive to IDs, so if an ID from a query does not matches the context. Go to step 5.
3. Analyze and understand the Query well. Understand the relationship between Query and Context. 
4. Just provide PRECISE Response. You HATE repetition. 
5. if the you think the Context does not contain any related responses, simple Respond with "I do not have any related knowledge for this specific topic"

Provide a clear and concise Response. You are ONLY required to use Context as your knowledge base for Response generation. 
DO NOT Generate any extra text, JUST provide clearn and concise answer.
    
    Query: """
    
    # Escaping special characters in query and context using repr()
    prompt = r"" + initial_string + repr(query)[1:-1] + r"\n\nContext:\n" + repr(context)[1:-1] + r"\n\nYour Response:"
    
    return prompt


def rerank_docs(
    query: str,
    top_5_results: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Rerank documents based on their relevance to the query using a cross-encoder.

    Args:
        query (str): The search query.
        top_5_results (List[Dict[str, str]]): Initial top 5 search results to rerank.

    Returns:
        List[Dict[str, str]]: Reranked list of documents.
    """
    # Re-ranking using cross-encoder
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    # Prepare pairs for reranking
    pairs = [[query, doc["content"]] for doc in top_5_results]

    # Get relevance scores
    scores = reranker.predict(pairs)

    # Sort by new scores
    reranked_results = [
        doc for _, doc in sorted(
            zip(scores, top_5_results),
            reverse=True
        )
    ]

    return reranked_results



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