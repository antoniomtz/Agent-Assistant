"""
RAG Assistant with FAISS Vector Storage

This script implements a Retrieval-Augmented Generation (RAG) system with:
- FAISS vector storage for efficient embedding persistence
- ReAct agent pattern for step-by-step reasoning
- Dual-display Gradio interface showing both answers and reasoning process
- NVIDIA API integration for LLM and embedding models
"""

import os
import re
import io
import sys
import hashlib
import faiss
import time
import numpy as np
from pathlib import Path
from contextlib import redirect_stdout
from dotenv import load_dotenv
import gradio as gr

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    Settings, 
    StorageContext,
    load_index_from_storage
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.litellm import LiteLLMEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

# --- Configuration ---

# Load environment variables
load_dotenv()

# NVIDIA API configuration
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY environment variable not set. Please set it before running.")

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
LLM_MODEL_NAME = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
EMBEDDING_MODEL_NAME = "baai/bge-m3"
EMBEDDING_DIMENSION = 1024  # Dimension for BGE-M3 embeddings

# Storage configuration
PERSIST_DIR = Path("./stored_indexes")
PERSIST_DIR.mkdir(exist_ok=True, parents=True)

# Chunk configuration
CHUNK_SIZE = 256
CHUNK_OVERLAP = 20

# FAISS configuration
FAISS_USE_GPU = False  # Set to True if you have GPU support with faiss-gpu
FAISS_NPROBE = 10      # Number of cells to probe (trade-off between accuracy and speed)

# Global state
global_agent = None
current_index = None

# --- Core Functions ---

def setup_models():
    """Initialize the LLM and embedding models from NVIDIA's API"""
    # Initialize the LLM (Chat Model)
    llm = OpenAILike(
        model=LLM_MODEL_NAME,
        api_base=NVIDIA_API_BASE,
        api_key=NVIDIA_API_KEY,
        is_chat_model=True,
        temperature=0.6,
        max_tokens=2048,
    )
    
    # Initialize the Embedding Model
    embed_model = LiteLLMEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        api_key=NVIDIA_API_KEY,
        api_base=NVIDIA_API_BASE,
        litellm_kwargs={
            "api_key": NVIDIA_API_KEY,
            "api_base": NVIDIA_API_BASE,
            "encoding_format": "float",
        }
    )
    
    # Configure global settings
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    return llm, embed_model

def get_index_id(file_path):
    """Generate a unique ID for the index based on the file path"""
    return hashlib.md5(str(file_path).encode()).hexdigest()

def create_index_from_file(file_path, force_reload=False):
    """
    Create or load a vector index from a file using FAISS persistence
    
    Args:
        file_path: Path to the file to index
        force_reload: Whether to force reloading the document even if a persisted index exists
        
    Returns:
        VectorStoreIndex: The loaded or created index
    """
    # Create a unique index ID based on file path
    index_id = get_index_id(file_path)
    index_dir = PERSIST_DIR / index_id
    
    # Check if index already exists
    if not force_reload and index_dir.exists():
        print(f"Loading existing index from {index_dir}")
        try:
            start_time = time.time()
            # Load the vector store from disk
            vector_store = FaissVectorStore.from_persist_dir(persist_dir=str(index_dir))
            
            # Create a storage context and load the index
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir=str(index_dir)
            )
            index = load_index_from_storage(storage_context=storage_context)
            print(f"Successfully loaded index from disk in {time.time() - start_time:.2f} seconds")
            
            # Verify FAISS index is loaded properly
            faiss_index = vector_store.client
            if faiss_index is not None:
                print(f"FAISS index loaded with {faiss_index.ntotal} vectors")
                # Optimize search parameters if index is loaded correctly
                if isinstance(faiss_index, faiss.IndexFlatL2):
                    print("Using FAISS IndexFlatL2 for exact search")
                elif hasattr(faiss_index, 'nprobe'):
                    # For IndexIVF-based indexes, set nprobe
                    faiss_index.nprobe = FAISS_NPROBE
                    print(f"Set FAISS nprobe to {FAISS_NPROBE}")
            else:
                print("Warning: FAISS index not properly loaded")
                
            return index
        except Exception as e:
            print(f"Error loading index: {e}. Creating new index instead.")
            # Continue to create new index if loading fails
    
    # Create new index
    print(f"Creating new index for {file_path}")
    index_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Load document
        documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
        print(f"Document loaded with {len(documents)} pages")
        
        # Parse into nodes
        parser = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        nodes = parser.get_nodes_from_documents(documents)
        print(f"Document parsed into {len(nodes)} nodes")
        
        # Create FAISS index
        if FAISS_USE_GPU and faiss.get_num_gpus() > 0:
            # Use IVF index for faster retrieval with many vectors
            faiss_index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(EMBEDDING_DIMENSION), 
                EMBEDDING_DIMENSION, 
                min(100, max(4, 4 * int(len(nodes) / 1000)))  # Adjust number of centroids based on document size
            )
            faiss_index.nprobe = FAISS_NPROBE
            print(f"Using GPU-enabled FAISS IVFFlat index with {faiss_index.nlist} clusters")
            
            # Move to GPU if available
            res = faiss.StandardGpuResources()
            faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        else:
            # For smaller collections, use exact search for better accuracy
            if len(nodes) > 1000:
                # Use IVF index for larger collections
                base_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
                faiss_index = faiss.IndexIVFFlat(
                    base_index, 
                    EMBEDDING_DIMENSION, 
                    min(100, max(4, 4 * int(len(nodes) / 1000)))
                )
                faiss_index.nprobe = FAISS_NPROBE
                # Need to train the index
                print("Training FAISS index...")
                # We would need embeddings for training, use dummy ones temporarily
                # In production, consider pre-computing embeddings and training properly
                dummy_data = np.random.random((max(256, len(nodes)), EMBEDDING_DIMENSION)).astype('float32')
                faiss_index.train(dummy_data)
                print(f"Using CPU FAISS IVFFlat index with {faiss_index.nlist} clusters")
            else:
                # Use exact search for small collections
                faiss_index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
                print("Using CPU FAISS FlatL2 index for exact search (smaller document)")
        
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create vector index
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        
        # Persist the index
        storage_context.persist(persist_dir=str(index_dir))
        print(f"Index created and saved to {index_dir} in {time.time() - start_time:.2f} seconds")
        
        return index
    except Exception as e:
        print(f"Error creating index: {e}")
        raise

def setup_rag_agent(index, llm):
    """Create a ReAct agent with a RAG tool"""
    # Create a query engine from the index with optimized parameters
    query_engine = index.as_query_engine(
        similarity_top_k=3,            # Retrieve more contexts for better accuracy
        vector_store_query_mode="default",  # Use default mode for FAISS
        alpha=0.5,                     # Weight between keyword and semantic search if using hybrid
        response_mode="compact"        # Get concise answers
    )
    
    # Create a QueryEngineTool
    rag_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="retail_operations_manual",
            description= "Use this tool to answer questions about retail operations, "
            "store procedures, product catalog, employee guidelines, pricing policies, "
            "and customer FAQs. The source contains inventory data, return policies, "
            "opening/closing checklists, promotions, and product descriptions.",
        ),
    )
    
    # Create a ReAct agent with the tool
    agent = ReActAgent.from_tools(
        tools=[rag_tool],
        llm=llm,
        verbose=True
    )
    
    return agent

def initialize_system():
    """Initialize the RAG system and return the agent"""
    global global_agent, current_index
    
    try:
        # Set up models
        llm, embed_model = setup_models()
        
        # Create path to the sample file
        script_dir = Path(__file__).parent
        sample_file = script_dir / "test_data" / "retail-manual.pdf"
        
        # Ensure the test_data directory exists
        if not sample_file.exists():
            return f"Error: Sample file not found at {sample_file}. Make sure the test_data directory contains retail-manual.pdf"
        
        # Create index
        current_index = create_index_from_file(sample_file, force_reload=False)
        
        # Set up agent
        global_agent = setup_rag_agent(current_index, llm)
        
        return "System initialized! You can now ask questions about paint products."
    except Exception as e:
        return f"Error initializing system: {str(e)}"

# --- Gradio UI Functions ---

def format_chain_of_thought(cot_text):
    """Format the captured chain of thought with emojis and markdown"""
    # Clean ANSI color codes and other artifacts
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    cot_text = ansi_escape.sub('', cot_text)
    
    # Format the Chain of Thought with emojis
    formatted_cot = ""
    lines = cot_text.split('\n')
    
    for line in lines:
        # Skip debugging/step information lines
        if line.strip().startswith(">") or "Running step" in line or not line.strip():
            continue
            
        # Apply consistent emoji formatting
        if line.strip().startswith("Thought:"):
            formatted_cot += f"🤔 **{line.strip()}**\n\n"
        elif line.strip().startswith("Action:"):
            formatted_cot += f"🛠️ **{line.strip()}**\n\n"
        elif line.strip().startswith("Action Input:"):
            formatted_cot += f"📥 **{line.strip()}**\n\n"
        elif line.strip().startswith("Observation:"):
            formatted_cot += f"👁️ **{line.strip()}**\n\n"
        elif line.strip().startswith("Answer:"):
            formatted_cot += f"✅ **{line.strip()}**\n\n"
        elif line.strip():
            formatted_cot += f"{line.strip()}\n\n"
    
    return formatted_cot

def user_message(message, history):
    """Add user message to chatbot"""
    return "", history + [{"role": "user", "content": message}]

def bot_message(history):
    """Generate bot response for the latest message and capture the chain of thought"""
    if global_agent is None:
        return history + [{"role": "assistant", "content": "System not initialized properly. Please restart the application."}], "System not initialized"
    
    user_message = history[-1]["content"]
    
    try:
        # Capture the agent's verbose output
        f = io.StringIO()
        with redirect_stdout(f):
            start_time = time.time()
            response = global_agent.query(user_message)
            end_time = time.time()
        
        # Format the captured output
        cot_text = f.getvalue()
        formatted_cot = format_chain_of_thought(cot_text)
        
        # Add performance information
        query_time = end_time - start_time
        
        # Add agent's response to chat history
        history.append({"role": "assistant", "content": str(response)})
        
        return history, formatted_cot
    except Exception as e:
        error_message = f"Error: {str(e)}"
        history.append({"role": "assistant", "content": error_message})
        return history, f"🔴 **Error occurred:**\n\n{error_message}"

def use_sample_question(sample_question, history):
    """Use a sample question"""
    return "", history + [{"role": "user", "content": sample_question}]

def clear_chat(initialize_message):
    """Clear the chat history"""
    return [{"role": "assistant", "content": initialize_message}], "", ""

def create_ui():
    """Create a Gradio UI for the RAG system"""
    # Initialize the system on startup
    initialize_message = initialize_system()
    
    with gr.Blocks(title="Retail Operations Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🏪 Retail Operations Assistant")
        gr.Markdown("Ask questions about products, store policies, customer service procedures, returns, promotions, and more.")
        
        # Disclaimer notice
        with gr.Accordion("⚠️ Disclaimer", open=False):
            gr.Markdown("""
            **Important Notice**: This system is a demonstration using entirely fictional data. 
            Information provided should not be used for real-world decisions or business operations.
            All product information, prices, and recommendations are fictional and created for
            educational purposes only. See DISCLAIMER.md for full legal details.
            """)
        
        with gr.Row():
            # Left column for chat
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    show_label=False,
                    height=500,
                    type='messages',
                    value=[{"role": "assistant", "content": initialize_message}]
                )
                
                with gr.Row():
                    question_input = gr.Textbox(
                        placeholder="Ask a question about paint products...",
                        scale=10,
                        show_label=False,
                        container=False
                    )
                    submit_button = gr.Button("Submit", variant="primary", scale=1)
                    clear_button = gr.Button("Clear Chat", variant="secondary", scale=1)
                
                gr.Markdown("### Sample Questions")
                with gr.Row():
                    sample_btn1 = gr.Button("How do I process a return for an online order in the store?")
                    sample_btn2 = gr.Button("What is the price of Compact Treadmill?")
                
                with gr.Row():
                    sample_btn3 = gr.Button("What are the store opening and closing procedures?")
                    sample_btn4 = gr.Button("Tell me more about the back-to-school tech promotion")
            
            # Right column for Chain of Thought
            with gr.Column(scale=4):
                gr.Markdown("### 🔄 Chain of Thought")
                gr.Markdown("See the agent's reasoning process")
                cot_output = gr.Markdown("")
        
        # Define clear_chat with initialize_message closure
        def clear_chat_with_message():
            return clear_chat(initialize_message)
        
        # Event handlers
        submit_button.click(
            user_message, 
            [question_input, chatbot], 
            [question_input, chatbot]
        ).then(
            bot_message, 
            [chatbot], 
            [chatbot, cot_output]
        )
        
        question_input.submit(
            user_message, 
            [question_input, chatbot], 
            [question_input, chatbot]
        ).then(
            bot_message, 
            [chatbot], 
            [chatbot, cot_output]
        )
        
        clear_button.click(
            clear_chat_with_message,
            [],
            [chatbot, question_input, cot_output]
        )
        
        # Sample question event handlers
        for btn in [sample_btn1, sample_btn2, sample_btn3, sample_btn4]:
            btn.click(
                use_sample_question, 
                [btn, chatbot], 
                [question_input, chatbot]
            ).then(
                bot_message, 
                [chatbot], 
                [chatbot, cot_output]
            )
        
    return demo

def main():
    """Main entry point for the application"""
    demo = create_ui()
    # Launch server with queue enabled for better handling of multiple requests
    demo.queue().launch(share=False)

if __name__ == "__main__":
    main() 