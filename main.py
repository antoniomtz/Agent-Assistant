import os
import base64
import requests
import json
import gradio as gr
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# NVIDIA API configuration
API_KEY = os.getenv("NVIDIA_API_KEY")
INVOKE_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"

def encode_image(image):
    """Convert image to base64 string"""
    if image is None:
        return None
    with open(image, "rb") as f:
        return base64.b64encode(f.read()).decode()

def chat_with_llm(message, history, image=None):
    """Process chat with or without image"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "text/event-stream"
    }
    
    # Prepare message content
    if image:
        image_b64 = encode_image(image)
        if image_b64:
            content = f'{message} <img src="data:image/jpg;base64,{image_b64}" />'
        else:
            content = message
    else:
        content = message

    # Prepare messages with history
    messages = []
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": content})

    payload = {
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": True
    }

    try:
        response = requests.post(INVOKE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
                
            try:
                # Decode the line and remove any whitespace
                decoded_line = line.decode("utf-8").strip()
                if not decoded_line:
                    continue
                    
                # Print raw response for debugging
                print(f"Raw response: {decoded_line}")
                
                # Handle SSE format by stripping "data: " prefix
                if decoded_line.startswith("data: "):
                    decoded_line = decoded_line[6:]  # Remove "data: " prefix
                
                # Skip [DONE] message
                if decoded_line == "[DONE]":
                    continue
                
                # Try to parse the JSON
                json_response = json.loads(decoded_line)
                
                # Extract content from the response
                if "choices" in json_response and len(json_response["choices"]) > 0:
                    delta = json_response["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        content = delta["content"]
                        full_response += content
                        yield [(message, full_response)]
                        
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line: {decoded_line}")
                continue
            except Exception as e:
                print(f"Error processing response: {e}")
                continue
                
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(error_msg)
        yield [(message, error_msg)]

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Neva Chat Interface")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600)
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False
                )
                image_input = gr.Image(type="filepath", label="Upload Image (Optional)")
                submit_btn = gr.Button("Send")
    
    submit_btn.click(
        chat_with_llm,
        inputs=[msg, chatbot, image_input],
        outputs=[chatbot],
        api_name="chat"
    ).then(
        lambda: ("", None),
        None,
        [msg, image_input],
        queue=False
    )
    
    msg.submit(
        chat_with_llm,
        inputs=[msg, chatbot, image_input],
        outputs=[chatbot],
        api_name="chat"
    ).then(
        lambda: ("", None),
        None,
        [msg, image_input],
        queue=False
    )

if __name__ == "__main__":
    demo.launch()
