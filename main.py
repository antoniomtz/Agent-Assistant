import os
import base64
import requests
import json
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI as DirectOpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai_like import OpenAILike
from typing import Optional, List, Dict, Any
from gradio import ChatMessage

# Load environment variables
load_dotenv()

# API configurations
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable not set. Please set it before running.")

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
NEVA_INVOKE_URL = "https://ai.api.nvidia.com/v1/vlm/nvidia/neva-22b"

# Initialize LlamaIndex OpenAILike wrapper for Nemotron
llm = OpenAILike(
    model=MODEL_NAME,
    api_base=NVIDIA_API_BASE,
    api_key=NVIDIA_API_KEY,
    is_chat_model=True,
    temperature=0.6,
    max_tokens=4096,
    additional_kwargs={
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
)

def encode_image(image):
    """Convert image to base64 string"""
    if image is None:
        return None
    with open(image, "rb") as f:
        return base64.b64encode(f.read()).decode()

def process_image_with_neva(image_path: str, query: str) -> str:
    """Process an image using Neva API and return the description"""
    # Add validation to ensure image path is provided
    if not image_path or not isinstance(image_path, str):
        return "Error: Invalid image path provided"
        
    print(f"Processing image: {image_path} with query: {query}")
    
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "text/event-stream"
    }
    
    image_b64 = encode_image(image_path)
    if not image_b64:
        return f"Error: Could not encode image at {image_path}"
        
    content = f'{query} <img src="data:image/jpg;base64,{image_b64}" />'
    
    messages = [{"role": "user", "content": content}]
    
    payload = {
        "messages": messages,
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.70,
        "seed": 0,
        "stream": True
    }

    try:
        response = requests.post(NEVA_INVOKE_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        full_response = ""
        for line in response.iter_lines():
            if not line:
                continue
                
            try:
                decoded_line = line.decode("utf-8").strip()
                if not decoded_line:
                    continue
                    
                if decoded_line.startswith("data: "):
                    decoded_line = decoded_line[6:]
                
                if decoded_line == "[DONE]":
                    continue
                
                json_response = json.loads(decoded_line)
                
                if "choices" in json_response and len(json_response["choices"]) > 0:
                    delta = json_response["choices"][0].get("delta", {})
                    if "content" in delta and delta["content"]:
                        full_response += delta["content"]
                        
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                continue
            except Exception as e:
                print(f"Error processing response: {e}")
                continue
                
        print(f"Image processing completed with response length: {len(full_response)}")
        return full_response
                
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        print(error_msg)
        return error_msg

# Create the image processing tool
image_tool = FunctionTool.from_defaults(
    fn=process_image_with_neva,
    name="process_image",
    description="Use this tool when you need to analyze or describe an image. Input should be the image path and a query about what to analyze in the image."
)

# Initialize the ReAct agent with tools
agent = ReActAgent.from_tools(
    [image_tool],
    llm=llm,
    verbose=True,
    system_prompt="""You are a helpful AI assistant that can analyze both text and images. 
    When a user uploads an image and asks about it, use the process_image tool to analyze it.
    Be conversational, helpful, and focus on addressing the user's needs.
    
    Do NOT include your thinking process or tool calls in your final answer to the user.
    Only return the final analysis or response that would be helpful to the user.
    """
)

def gradio_chat(message, history, image=None):
    """Handle chat interactions through Gradio"""
    # Initialize empty history if None
    if history is None:
        history = []
    
    # Add user message to history
    if image:
        # Format for display to show image was uploaded
        display_msg = f"{message} [Image uploaded]" 
        history.append({"role": "user", "content": display_msg})
    else:
        history.append({"role": "user", "content": message})
        
    # Show typing indicator
    history.append({"role": "assistant", "content": "..."})
    yield history
    
    try:
        # Prepare the message for the agent
        user_message = message
        if image:
            user_message = f"Analyze this image: {message}\nImage path: {image}"
            
            # Show processing indicator
            history[-1] = {
                "role": "assistant", 
                "content": "Analyzing your image...",
                "metadata": {"title": "⏳ Processing"}
            }
            yield history
        
        # Set up a print capture for debugging
        import io
        import sys
        original_stdout = sys.stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Get response from the agent
        response = agent.chat(
            message=user_message,
            chat_history=[]  # Start fresh for each query
        )
        
        # Restore stdout
        sys.stdout = original_stdout
        
        # Get the captured output for debugging
        debug_output = captured_output.getvalue()
        print(f"DEBUG: Agent process: {debug_output}")
        
        # Extract the final answer (not the thinking process)
        final_response = response.response
        
        # Check if response contains Action/Thought patterns and extract only final answer if needed
        if "Action:" in final_response or "Thought:" in final_response:
            # Extract just the final answer if it exists
            if "Answer:" in final_response:
                answer_parts = final_response.split("Answer:")
                if len(answer_parts) > 1:
                    final_response = answer_parts[1].strip()
            
        # Extract agent reasoning if available (for metadata/accordion)
        thinking = None
        if hasattr(response, 'sources') and response.sources:
            # Format agent reasoning and tool usage for display
            thinking = []
            for source in response.sources:
                if isinstance(source, list) and source:
                    for step in source:
                        if hasattr(step, 'tool_name'):
                            thinking.append(f"Used tool: {step.tool_name}")
                            thinking.append(f"Tool input: {step.tool_input}")
                            thinking.append(f"Tool output: {step.observation}")
        
        # Update with final response, including thinking if available
        if thinking:
            history[-1] = {
                "role": "assistant", 
                "content": final_response,
                "metadata": {"title": "Agent Reasoning", "content": "\n".join(thinking)}
            }
        else:
            history[-1] = {"role": "assistant", "content": final_response}
            
        yield history
            
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"Error in gradio_chat: {error_msg}")
        print(f"Error details: {repr(e)}")
        
        # Replace typing indicator with error message
        history[-1] = {"role": "assistant", "content": f"I encountered an error: {error_msg}"}
        yield history

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# AI Assistant with Image Analysis")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=600, type="messages")
            with gr.Row():
                msg = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    show_label=False,
                    container=False
                )
                image_input = gr.Image(type="filepath", label="Upload Image (Optional)")
                submit_btn = gr.Button("Send")
    
    clear_btn = gr.Button("Clear Conversation")
    
    # Clear conversation handler
    def clear_conversation():
        return None, None, []
    
    # Set up event handlers
    submit_btn.click(
        gradio_chat,
        inputs=[msg, chatbot, image_input],
        outputs=[chatbot]
    ).then(
        lambda: ("", None),
        None,
        [msg, image_input]
    )
    
    msg.submit(
        gradio_chat,
        inputs=[msg, chatbot, image_input],
        outputs=[chatbot]
    ).then(
        lambda: ("", None),
        None,
        [msg, image_input]
    )
    
    clear_btn.click(
        clear_conversation,
        None,
        [msg, image_input, chatbot]
    )

if __name__ == "__main__":
    demo.launch()
