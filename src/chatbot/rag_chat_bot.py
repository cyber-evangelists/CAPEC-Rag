
from typing import Dict, List
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from langchain_core.output_parsers import StrOutputParser

# from src.config.config import Config

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class RAGChatBot:
    def __init__(self):
        # Set your Groq API key

        # Initialize the chat model
        self.llm = ChatGroq(
            model_name="llama-3.1-8b-instant",
            temperature=0,
            max_tokens=4096,
        )

        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history"
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Cybersecurity Expert Chatbot Providing Expert Guidance. Respond in a natural, human-like manner. You will be given Context and a Query."""),
             
            ("system", """The Context contains CAPEC dataset entries. Key Fields:
             
ID: Unique identifier for each attack pattern. (CAPEC IDs)
Name: Name of the attack pattern.
Abstraction: Generalization level of the attack pattern.
Status: Current status of the attack pattern (e.g., Draft, Stable).
Description: Detailed description of the attack pattern.
Alternate Terms: Other terms used to describe the attack.
Likelihood Of Attack: Probability of the attack occurring.
Typical Severity: Expected impact severity of the attack.
Related Attack Patterns: Related CAPEC attack patterns Explaining Relationships among CAPEC attack patterns like Child, Parent, CanPrecede, CanFollow etc.
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
Notes: Additional information."""),

            ("system", """For each Query follow these guidelines:
            
            Response Guidelines:
            1. If Query matches Context: Provide focused answer using only provided Context.If asked for Explanation, Explain the desired thing in detial.
            2. If Query does not matches with Context but cybersecurity-related: Provide general expert guidance.
            3. Otherwise: Respond with "I am programmed to answer queries related to Cyber Security Only.\""""),

        ("system", """Keep responses professional yet conversational, focusing on practical security implications.
         Context {context}: """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])


    def _create_chain(self, query: str, context: str) -> RunnableSequence:
        """Create a chain for a single query-context pair"""

        def get_context_and_history(_: dict) -> dict:
            chat_history = self.memory.load_memory_variables({})["chat_history"]
            return {"context": context, "chat_history": chat_history, "input": query}

        return (
            RunnablePassthrough()
            | get_context_and_history
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _update_memory(self, input_text: str, output_text: str) -> None:
        """Update conversation memory with the latest interaction"""
        self.memory.save_context({"input": input_text}, {"output": output_text})

    def chat(self, query: str, context: List[str]) -> str:
        """
        Process a single message with provided context and return the response

        Args:
            query (str): The user's question
            docs (List[str]): List of relevant document contents/contexts

        Returns:
            str: The model's response
        """
        # Format the context

        # Create and run the chain
        chain = self._create_chain(query, context)
        response = chain.invoke({})

        # Update memory
        self._update_memory(query, response)

        return response

    def get_chat_history(self) -> List[BaseMessage]:
        """Return the current chat history"""
        return self.memory.load_memory_variables({})["chat_history"]

