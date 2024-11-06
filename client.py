import asyncio
import json
import os
from typing import Tuple, List, Optional, Dict, Any

import gradio as gr
import websockets
from loguru import logger

from src.config.config import Config


class ChatbotUI:
    """A class to handle the chatbot user interface and WebSocket communication."""

    def __init__(self):
        """Initialize the ChatbotUI with configuration settings."""
        self.max_history = Config.MAX_CHAT_HISTORY  # Maximum number of messages to keep
        self.websocket = None

    def clear_chat(self) -> Optional[List[Tuple[str, str]]]:
        """
        Clear the chat history.

        Returns:
            Optional[List[Tuple[str, str]]]: None to clear the chat history.
        """
        return None


    async def connect(self):
        try:
            uri = Config.WEBSOCKET_URI
            if not uri.startswith(('ws://', 'wss://')):
                logger.error("Invalid WebSocket URI format")
                return
            self.websocket = await websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=60,
                max_size=10_485_760
            )
            
        except Exception as e:
            logger.error(f"Connection error: {e}")



    async def handle_request(
        self, action: str, payload: dict
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Handle WebSocket requests to the server.

        Args:
            action (str): The action to perform (e.g., 'search', 'ingest_data').
            payload (dict): The payload containing request data.

        Returns:
            Tuple[str, List[Tuple[str, str]]]: A tuple containing the response message
            and updated chat history.
        """

        query = payload["query"]
        if not query.strip():
            logger.error(f"No input provided")
            return "", [(payload.get("query", ""), "No query Entered")]


        uri = Config.WEBSOCKET_URI

        if not uri.startswith(('ws://', 'wss://')):
            logger.error(f"Invalid WebSocket URI format: {uri}")
            return "", [(payload.get("query", ""), "Invalid WebSocket URI configuration")]

        if self.websocket is None or self.websocket.closed:
            await self.connect()

        try:
        
            # Set up connection timeout
            response_future = asyncio.create_task(
                self._handle_websocket_communication(action, payload)
            )

            try:
                result = await asyncio.wait_for(
                    response_future,
                    timeout=300  # 5-minute timeout
                )
                return result
            except asyncio.TimeoutError:
                logger.error("Request timed out")
                return "", [(payload.get("query", ""), "Request timed out")]

        except Exception as e:
            logger.error(f"Connection error: {e}")
            return "", [(payload.get("query", ""), f"Connection error: {str(e)}")]
        finally:
            self.websocket = None

    async def _handle_websocket_communication(
        self, action: str, payload: dict
    ) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Handle the WebSocket communication with the server.

        Args:
            action (str): The action to perform.
            payload (dict): The payload containing request data.

        Returns:
            Tuple[str, List[Tuple[str, str]]]: A tuple containing the response message
            and updated chat history.
        """
        try:
            await self.websocket.send(json.dumps({"action": action, "payload": payload}))

            while True:
                response = await self.websocket.recv()
                response_data = json.loads(response)

                # Handle heartbeat
                if response_data.get("type") == "ping":
                    await self.websocket.send(json.dumps({"action": "pong"}))
                    continue

                result = response_data.get("result", "No response from server")
                if result:
                    if action == "search":
                        history = payload.get("history", [])
                        new_message = (payload.get("query", ""), result)
                        updated_history = history + [new_message]
                        return "", updated_history
                    elif action == "ingest_data":
                        return result, []

                error = response_data.get("error")
                if error:
                    return "", [(payload.get("query", ""), f"Error: {error}")]

        except Exception as e:
            logger.error(f"Communication error: {e}")
            return "", [(payload.get("query", ""), f"Communication error: {str(e)}")]

    def create_interface(self) -> gr.Blocks:
        """
        Create the Gradio interface for the chatbot.

        Returns:
            gr.Blocks: The configured Gradio interface.
        """
        with gr.Blocks(
            title="ASM Chatbot",
            theme=gr.themes.Soft(),
            css=".gradio-container {max-width: 800px; margin: auto}"
        ) as demo:
            gr.Markdown("""
            # ASM Chatbot
            Ask questions about ASM and get detailed responses.
            """)

            with gr.Row():
                msg = gr.Textbox(
                    label="Type your message here...",
                    placeholder="Enter your query",
                    show_label=True,
                    container=True,
                    scale=8
                )

            with gr.Row():
                search_btn = gr.Button("Search", variant="primary", scale=2)
                # ingest_data_btn = gr.Button("Ingest Data", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                status_box = gr.Textbox(visible=False)

            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
                container=True,
                elem_id="chatbot"
            )


            submit_click = search_btn.click(
                fn=lambda msg, history: asyncio.run(
                    self.handle_request(
                        "search",
                        {"query": msg, "history": history if history else []}
                    )
                ),
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )



            clear_btn.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot]
            )


        return demo


def launch_gradio_interface() -> None:
    """Launch the Gradio interface with configured settings."""
    chatbot_ui = ChatbotUI()
    demo = chatbot_ui.create_interface()

    server_name = Config.GRADIO_SERVER_NAME
    server_port = int(Config.GRADIO_SERVER_PORT)

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=False,
        debug=True,
        show_error=True,
    )


if __name__ == "__main__":
    launch_gradio_interface()