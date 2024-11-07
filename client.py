# client.py
import gradio as gr
import websockets
import json
import asyncio
import logging
from typing import Tuple, List, Optional, Dict, Any
from loguru import logger

from src.config.config import Config



class WebSocketClient:
    def __init__(self, uri: str = "ws://rag-server:8000/ws"):
        self.uri = uri
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._connection_lock = asyncio.Lock()
        
    async def connect(self):
        if not self.websocket:
            self.websocket = await websockets.connect(self.uri)
            logger.info("Connected to WebSocket server")
        return self.websocket
        
    async def disconnect(self):
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
            logger.info("Disconnected from WebSocket server")
            
    async def send_search_query(self, query: str, payload:str) -> str:
        try:
            # Ensure connection is established
            if not self.websocket:
                await self.connect()
                
            # Send search query
            await self.websocket.send(json.dumps({"query": query}))
            
            # Wait for response
            response = await self.websocket.recv()
            data = json.loads(response)
            
            if data["status"] == "success":
                # Format results for display
                results = data["results"]
                formatted_results = "\n\n".join([
                    f"Title: {result['title']}\nURL: {result['url']}"
                    for result in results
                ])
                return formatted_results, payload
            else:
                return f"Error: {data['message'], payload}"
                
        except websockets.exceptions.ConnectionClosedError:
            logger.error("Connection closed unexpectedly. Attempting to reconnect...")
            self.websocket = None
            return "Connection lost. Please try again."
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return f"Error: {str(e)}"
        
    async def ensure_connection(self): 
        if self.websocket is None or self.websocket.closed:
            await self.connect()


    async def connect(self):
        """Establish WebSocket connection and start heartbeat monitoring"""
        async with self._connection_lock:
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
                logger.info("Connected to server.....")

                # self._heartbeat_task = asyncio.create_task(self._heartbeat())

                return True

            except Exception as e:
                logger.error(f"Connection error: {e}")
                return False
            
    async def handle_request(
        self, action: str, payload: dict = {}
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

        logger.info("Into handle search function..")

        if action ==  "search":
            query = payload["query"]
            if not query.strip():
                logger.error(f"No input provided")
                return "", [(payload.get("query", ""), "No query Entered")]
            
        try:
            
            logger.info("Ensuring Connection....")
            await ws_client.ensure_connection()

            result = await self._handle_websocket_communication(action, payload)
            return  result
        
        except Exception as e:
            logger.error(f"Connection error: {e}")
            await self.disconnect() 
            return "", [(payload.get("query", ""), f"Connection error: {str(e)}")]
        
        
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

                logger.info("Response received...")

                # Handle heartbeat
                if response_data.get("type") == "ping":
                    await self.websocket.send(json.dumps({
                        "action": "pong",
                        "timestamp": response_data.get("timestamp")
                    }))
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
        


# Create WebSocket client instance
ws_client = WebSocketClient(Config.WEBSOCKET_URI)


async def search_click(msg, history):
    return await ws_client.handle_request(
        "search",
        {"query": msg, "history": history if history else []}
    )


async def handle_ingest() -> gr.Info:
    """
    Handle the data ingestion process.

    Args:
        ws_client (WebSocketClient): The WebSocket client instance.

    Returns:
        gr.Info: A Gradio info or warning message.
    """
    message, _ = await ws_client.handle_request("ingest_data", {})
    return gr.Info(message) if "success" in message.lower() else gr.Warning(message)


def clear_chat() -> Optional[List[Tuple[str, str]]]:
        """
        Clear the chat history.

        Returns:
            Optional[List[Tuple[str, str]]]: None to clear the chat history.
        """
        return None


# Create Gradio interface
with gr.Blocks(
            title="CAPEC Chatbot",
            theme=gr.themes.Soft(),
            css=".gradio-container {max-width: 800px; margin: auto}"
        ) as demo:
            gr.Markdown("""
            # ASM Chatbot
            Ask questions about CAPEC Dataset and get detailed responses.
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
                clear_btn = gr.Button("Clear", variant="secondary", scale=1)
                status_box = gr.Textbox(visible=False)


            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
                container=True,
                elem_id="chatbot"
            )

            search_btn.click(
                fn=search_click,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )

            clear_btn.click(
                fn=clear_chat,
                inputs=[],
                outputs=[chatbot]
            )

            

if __name__ == "__main__":
    server_name = Config.GRADIO_SERVER_NAME
    server_port = int(Config.GRADIO_SERVER_PORT)
    logger.info("Launching Gradio..")
    demo.queue().launch(server_name=server_name,
        server_port=server_port,
        share=False,
        debug=True,
        show_error=True,)