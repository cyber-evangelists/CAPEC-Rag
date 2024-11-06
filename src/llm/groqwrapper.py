from typing import Generator, Any

from groq import Groq
from groq.types.chat import ChatCompletion
from loguru import logger

from src.config.config import Config


class GroqWrapper:
    """A wrapper class for the Groq API client to handle chat completions."""

    def __init__(
        self,
        model_name: str = Config.GROQ_MODEL_NAME,
        api_key: str = Config.GROQ_API_KEY
        
    ) -> None:
        """
        Initialize the GroqWrapper with model configuration.

        Args:
            model_name (str): The name of the Groq model to use.
            api_key (str): The API key for Groq authentication.
        """
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def _response(self, message: str) -> Generator[ChatCompletion, None, None]:
        """
        Generate a streaming response from the Groq API.

        Args:
            message (str): The input message to send to the model.

        Returns:
            Generator[ChatCompletion, None, None]: A generator of chat completion chunks.
        """
        messages = [
            {
                "role": "user",
                "content": message
            }
        ]

        return self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=4096,
            stream=True,
            stop=None,
        )

    def get_response(self, query: str) -> str:
        """
        Get a complete response from the model for a given query.

        Args:
            query (str): The input query to process.

        Returns:
            str: The complete response from the model.
        """
        response = self._response(query)
        final_answer = ""
        for chunk in response:
            final_answer += chunk.choices[0].delta.content or ""

        return final_answer