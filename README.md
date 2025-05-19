# RAG Assistant with FAISS vector storage

This project implements an intelligent retail operations assistant using Retrieval-Augmented Generation (RAG) with FAISS vector storage for persistence and a dual-display chat interface showing both answers and reasoning.

## Disclaimer

**IMPORTANT**: This application uses fictional data for demonstration purposes only. The information provided by this system should not be used for actual business decisions. Please refer to the [DISCLAIMER.md](DISCLAIMER.md) file for comprehensive legal information regarding the use of this software.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Architecture

The system follows a RAG (Retrieval-Augmented Generation) architecture that combines vector search with large language models:

![RAG System Architecture](https://github.com/user-attachments/assets/cc9fd64a-0a04-478a-9149-5d11d8eff9d1 "RAG System Architecture")

## Demo (youtube)

[![Demo](https://img.youtube.com/vi/IxIhe4VsCm8/0.jpg)](https://www.youtube.com/watch?v=IxIhe4VsCm8)

## Setup

### 1. Set up a Python virtual environment (recommended)

It's recommended to use a virtual environment to avoid package conflicts:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

Once your virtual environment is activated, install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file with your NVIDIA API key:
```
NVIDIA_API_KEY=your_api_key_here
```

### 4. Prepare documents

Add your PDF documents to the `test_data` directory.

## Running the System

With your virtual environment activated, launch the chat application:

```bash
python main.py
```

The web interface will be available at http://127.0.0.1:7860 by default.

## Models Used

This application leverages powerful AI models through NVIDIA's API:

- **Large Language Model**: NVIDIA Llama 3.1 Nemotron Ultra (253B parameters)
  - A state-of-the-art LLM for high-quality reasoning and responses
  - Used for the ReAct agent's reasoning and answer generation

- **Embedding Model**: BAAI BGE-M3
  - Used to convert text chunks into vector embeddings
  - Provides 1024-dimensional vectors for semantic search

Both models are accessed through the NVIDIA API for optimal performance.

## NVIDIA API Cold Start

**Note about first-time usage**: The NVIDIA API may experience "cold start" issues when first initialized. The first 2-3 queries to the system might fail or timeout while the API resources are being allocated. This is normal behavior for cloud-based AI services.

If you receive errors on your first few queries:
- Wait a few seconds and try again
- Keep trying up to 3 times
- After 2-3 attempts, the API should stabilize and respond normally

This cold start behavior only affects the system after fresh initialization or long periods of inactivity.

## Features

- **Dual-Display Interface**: 
  - Main chat window for conversations
  - Chain of Thought window showing the agent's reasoning process with emoji indicators
- **ReAct Agent Integration**: Uses the ReAct (Reasoning + Acting) pattern to dynamically use tools to answer questions
- **FAISS Vector Persistence**: Stores embeddings on disk to avoid reprocessing documents
- **NVIDIA API Integration**: Leverages NVIDIA's powerful LLM and embedding models
- **Emoji-Enhanced Reasoning**: Visualizes agent reasoning with:
  - 🤔 Thought steps - showing the agent's reasoning process
  - 🛠️ Action steps - showing which tool is being used
  - 📥 Action Input - showing the parameters for the tool
  - 👁️ Observation - showing results from the tool
  - ✅ Answer - final conclusion after reasoning
- **Sample Questions**: Quick-access buttons for common retail queries
- **Automatic Initialization**: System initializes on startup

## How It Works

1. **Document Processing**:
   - Retail operations manual is loaded from the `test_data` directory
   - Content is chunked into smaller pieces and embedded using NVIDIA's embedding model
   - Embeddings are stored in a FAISS vector store for efficient similarity search

2. **Vector Persistence**:
   - The system assigns a unique ID to each source document
   - Embeddings are stored on disk in the `stored_indexes` directory
   - When run again, the system loads existing embeddings rather than reprocessing

3. **Query Handling**:
   - The ReAct agent receives employee questions through the chat interface
   - It determines when to use the RAG tool to retrieve relevant context from the operations manual
   - All reasoning steps are captured and displayed in the Chain of Thought panel
   - Final answers are displayed in the chat interface

## Using the Interface

1. Type your question in the text field (e.g., "How do I process a return?")
2. Press Enter or click "Submit" to get a response
3. Watch the Chain of Thought panel to see the agent's reasoning process
4. Use the sample question buttons for quick access to common queries
5. Click "Clear Chat" to start a new conversation

## Directories

- `stored_indexes/` - Contains the persisted FAISS indexes (it will be created during runtime)
- `test_data/` - Place your PDF documents here

## Technical Details

This system demonstrates modern RAG techniques with the following components:

- LlamaIndex for the core RAG functionality
- FAISS for fast vector search and persistence
- Gradio for the web interface
- ReAct agent pattern for step-by-step reasoning
- NVIDIA API for LLM inference and embeddings 