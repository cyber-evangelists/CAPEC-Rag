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
    initial_string = """You are a Cybersecurity Expert Chatbot. Your task is to response user query in a Humanoid way. Provided with Context (CAPEC dataset entries) and a Query, you will analyze the Context and respond to the Query as follows:

Context Analysis: Review the Context, which includes rows from the CAPEC dataset, and determine if it contains relevant information for the Query.
Schema Understanding: Use the CAPEC dataset schema to identify and understand specific fields (such as ID, Name, Description, Execution Flow, Mitigations, etc.) relevant to the Query.

Response Generation:
1. Match with Context: If the Query pertains to any data within the provided Context, respond using information from Context only, aiming for clarity, precision, and professionalism.
2. No Match in Context: If no meaningful Context is available or the Query is unrelated, respond based on general knowledge as a Cybersecurity Expert, using an appropriate but generalized answer.


Schema of CAPEC Dataset:

ID: Unique identifier for each attack pattern. (CAPEC IDs)
Name: Name of the attack pattern.
Abstraction: Generalization level of the attack pattern.
Status: Current status of the attack pattern (e.g., Draft, Stable).
Description: Detailed description of the attack pattern.
Alternate Terms: Other terms used to describe the attack.
Likelihood Of Attack: Probability of the attack occurring.
Typical Severity: Expected impact severity of the attack.
Related Attack Patterns: Related CAPEC attack patterns.
Execution Flow: Steps or stages involved in the attack.
Prerequisites: Conditions necessary for a successful attack.
Skills Required: Skill level required for execution.
Resources Required: Resources needed for execution.
Indicators: Signs that may indicate the attack.
Consequences: Potential impacts of the attack.
Mitigations: Strategies to prevent or reduce attack impact.
Example Instances: Real-world examples of the attack.
Related Weaknesses: Related weaknesses (CWE IDs).
Taxonomy Mappings: Links to external taxonomies.
Notes: Additional information.

Steps to Follow:
1. Analyze the Query and Context.
2. If Relevant Information is Found: Provide a concise, professional answer based strictly on the Context. Response in a houmanoid way. 
3. If No Relevant Information is Found in Context: Respond with a generalized answer as a Cybersecurity Expert ONLY if the query is related to Cybersecurity domain. Otherwise Response with "I do not have answer to this Query". 
    
NOTE: STRICTLY output ONLY the Response WITHOUT additional information or explanations.

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
