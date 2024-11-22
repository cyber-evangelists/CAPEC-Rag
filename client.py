import gradio as gr
import websockets
from typing import Tuple, List, Optional, Dict, Any
from loguru import logger

from src.config.config import Config
from src.websocket.web_socket_client import WebSocketClient
from src.guardrails.guardrails import GuardRails


ws_client = WebSocketClient(Config.WEBSOCKET_URI)
guardrails_model = GuardRails()


async def search_click(msg: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]], gr.Info]:

    if not msg.strip():
        logger.error(f"No input provided")
        return "", history,  gr.Warning("Please enter a query.")

    response = int(guardrails_model.classify_prompt(msg))

    if response == 0:
        result =  await ws_client.handle_request(
            "search",
            {"query": msg, "history": history if history else []}
        )
        if result[2] == "right":

            styled_response = (f"<div style='direction: rtl; text-align: right; direction: right;'>{result[1]}</div>")
        else:
            styled_response = f"<div style='direction: ltr; text-align: left; direction: left;'>{result[1]}</div>"
        
        # Append the styled response to the chat history
        updated_history = history + [(msg, styled_response)]


        return result[0], updated_history, gr.Info("Query Processed")

    else:
        return await return_protection_message(msg, history)


async def return_protection_message(msg, history):

    new_message = (msg, "Your query appears inappropriate. Do you have any other question?I am here to help.. ")
    updated_history = history + [new_message]
    return "", updated_history, gr.Warning("Query is Inapproprite..")

                    

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



async def record_feedback(feedback, msg ) -> gr.Info:
    """
    Handle the data ingestion process.

    Args:
        ws_client (WebSocketClient): The WebSocket client instance.

    Returns:
        gr.Info: A Gradio info or warning message.
    """

    logger.info(feedback)
    logger.info(msg)


    if not msg.strip():
        logger.error(f"No Comments provided")
        return gr.Info("Please Enter Some Feed back First"), ""

    message, _ = await ws_client.handle_request(feedback, {"comment": msg})
    return gr.Info(message) if "success" in message.lower() else gr.Warning(message), ""


with gr.Blocks(
    title="EASM RAG Chatbot",
    theme=gr.themes.Soft(),
    css="""
        .gradio-container {
            max-width: 700px; 
            margin: auto; 
            font-family: Arial, sans-serif;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        #header {
            text-align: center; 
            font-size: 1.5rem; 
            font-weight: bold; 
            color: #008080; 
            padding: 0.125rem;
            flex: 0 0 auto;
        }
        #input-container {
            display: flex; 
            align-items: center;
            background-color: #f7f7f8;
            padding: 0.25rem; 
            border-radius: 8px;
            margin-top: 0.25rem;
            flex: 0 0 auto;
        }
        #chatbot-left {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            background-color: #FFFFFF;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow-y: auto; /* To allow scrolling if content overflows */
            min-height: 62vh; 
            text-direction: left;
            direction: left;
            text-align: left;
        }
        #chatbot-right {
            border: 1px solid #E5E7EB;
            border-radius: 8px;
            background-color: #FFFFFF;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            flex: 1 1 auto;
            min-height: 0;
            display: flex;
            flex-direction: column;
            overflow-y: auto; /* To allow scrolling if content overflows */
            min-height: 62vh; 
            text-direction: right;
            direction: right;
            text-align: right;
        }
        #feedback-button {
            max-width: 0.25vh;
        }
        .gr-button-primary {
            background-color: #008080;
            border-color: #008080;
        }
        .gr-button-primary:hover {
            background-color: #006666;
        }
    """
) as demo:

    # Header
    gr.Markdown(
        "<div id='header'>EASM RAG Application</div>"
    )

    # Chatbot Component
    chatbot = gr.Chatbot(
        show_label=False,
        container=True,
        elem_id="chatbot-left"
    )

    with gr.Row(elem_id="feedback-container"):
        thumbs_up = gr.Button("👍", elem_id="feedback-button")
        thumbs_down = gr.Button("👎", elem_id="feedback-button")
        feedback_msg = gr.Textbox(
            placeholder="Type a comment...",
            show_label=False,
            container=False,
            lines=1,
            scale=10,
        )
        status_box = gr.Textbox(visible=False)

    # Chat Input Row
    with gr.Row(elem_id="input-container"):
        msg = gr.Textbox(
            placeholder="Type a message...",
            show_label=False,
            container=False,
            lines=1,
            scale=10,
        )
        send_button = gr.Button("Send", variant="primary", scale=1)
        clear_button = gr.Button("Clear Chat", variant="secondary")
        

    # Button Functionality
    send_button.click(
        fn=search_click,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, status_box]
    )
    clear_button.click(
        fn=clear_chat,
        inputs=[],
        outputs=[chatbot]
    )

    thumbs_up.click(
                fn=record_feedback,
                inputs=[gr.Textbox(value="positive", visible=False), feedback_msg],
                outputs=[status_box, feedback_msg]
            )
    
    thumbs_down.click(
                fn=record_feedback,
                inputs=[gr.Textbox(value="negative", visible=False), feedback_msg],
                outputs=[status_box, feedback_msg]
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
