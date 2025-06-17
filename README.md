# Personal Bio-AI Assistant (Local RAG Agent with Ollama & LangChain)

## Overview

This project focuses on building a **local AI assistant** tailored for bioinformatics tasks. By combining a Large Language Model (LLM) with Retrieval-Augmented Generation (RAG), this assistant aims to help users query their personal bioinformatics notes, code snippets, research papers (PDFs), and other domain-specific documents. A key benefit of this project is its **entirely local operation**, ensuring your data remains private and core functionalities work without an internet connection.

The assistant integrates a local LLM (like Mistral), a local Embedding Model (Nomic Embed), and a local Vector Database (ChromaDB) to provide relevant and accurate answers based on your private knowledge base.

## Key Features

* **Local LLM Integration:** Uses Ollama to run LLMs (e.g., Mistral) directly on your machine.
* **Local Embedding Generation:** Utilizes Ollama's Nomic Embed model for creating text embeddings without external APIs.
* **Private Knowledge Base:** Stores and queries your personal bioinformatics documents and code in a local ChromaDB vector store.
* **Retrieval-Augmented Generation (RAG):** Enhances LLM responses by retrieving relevant information from your documents, providing more informed answers.
* **Multi-Document Support:** Capable of processing various document types (PDFs, `.txt`, `.md`, `.py`, `.R`, `.sh` etc.).
* **Modular Architecture:** Built with LangChain for a flexible and extensible design.
* **Data Privacy:** All your data remains securely on your local machine.

## Project Structure

The repository is structured to organize the various components of the AI assistant:

```text
personal-bio-ai-assistant/
├── .venv/                   # Python Virtual Environment
├── .vscode/                 # VS Code specific settings
│   └── settings.json        # Editor and Python interpreter configuration
├── data/                    # Directory for your personal documents (PDFs, .txt, .md, code, etc.)
├── src/                     # Source code
│   ├── core/                # Core functionalities: LLM and Embedding Handlers
│   ├── utils/               # Utility functions: Document loading & processing
│   ├── knowledge_base/      # Knowledge base creation
│   └── agents/              # Main agent logic and orchestration
├── .chroma_db/              # Local ChromaDB persistent storage (auto-generated)
├── config.py                # Centralized project configuration (model names, chunk sizes, paths)
├── app.py                   # Streamlit UI interface
├── main.py                  # Command line interface
├── .env                     # Environment variables
├── .gitignore               # Specifies files/folders to be ignored by Git
├── README.md                # This file
└── requirements.txt         # List of Python dependencies
```

## Setup and Installation

Follow these steps to set up the project on your local machine.

### Prerequisites

* **Git:** For cloning the repository.
* **Python 3.9+:** Recommended for LangChain and Ollama compatibility.
* **Ollama:** The local LLM server.
    * Download and install from [ollama.com](https://ollama.com/).
    * Ensure the Ollama daemon is running in the background (check your system's menu bar).

## 1. Clone the Repository

```bash
git clone https://github.com/sebastianpirmann/personal-ai-agent-for-bioinformatics.git

cd personal-ai-agent-for-bioinformatics
```

## 2. Install Ollama Models
Download the required LLM and Embedding models using Ollama:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

Ensure these models are listed when you run ollama list.

## 3. Set up Python Virtual Environment
It's highly recommended to use a virtual environment to manage project dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.\.venv\Scripts\activate   # On Windows (PowerShell)
```

## 4. Install Python Dependencies
Install all necessary Python libraries using pip:

```bash
pip install -r requirements.txt
```

## 5. Configure VS Code (Recommended)
To ensure VS Code correctly uses your virtual environment and understands project imports:

Open the project folder (personal-ai-agent-for-bioinformatics) in VS Code via File > Open Folder....

Open the Command Palette (Ctrl/Cmd + Shift + P) and select Python: Select Interpreter. Choose the interpreter located within your project's .venv folder (e.g., Python 3.x.x (.venv)).

Ensure a .vscode/settings.json file exists in your project root with the following content. If it doesn't exist, create the .vscode/ folder and then settings.json inside it.

```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.analysis.extraPaths": [
        "./src"
    ]
}
```

Restart VS Code after making these changes.

# Usage
### 1. Prepare Your Knowledge Base
Place your documents (PDFs, .txt, .md, .py, .R, .sh files, etc.) into the data/ directory.

### 2. Build the Knowledge Base (Vector Database)
This step processes your documents, converts them into numerical embeddings using the specified embedding model, and stores them in a local ChromaDB instance (by default in the `.chroma_db/` directory).
This process needs to be run whenever you add, remove, or modify documents in the `data/` directory.

```bash
python3 -m src.utils.build_knowledge_base
```

### 3. Configure the AI Assistant
Open config.py in the root directory of the project.
Here you can adjust key parameters for your assistant:

- **OLLAMA_LLM_MODEL**: The specific Large Language Model (LLM) to use (e.g., "mistral", "gemma:2b", or a quantized version like "mistral:7b-instruct-v0.2-q3_K_M"). Ensure this model is pulled via Ollama.
- **OLLAMA_EMBEDDING_MODEL**: The embedding model (e.g., "nomic-embed-text"). Ensure this model is pulled via Ollama.
- **CHROMA_DB_PATH**: The local directory where your ChromaDB knowledge base is persisted.
- **DATA_PATH**: The directory where your source documents are located.
- **CHUNK_SIZE / CHUNK_OVERLAP**: Parameters for how documents are split into smaller pieces.
- **CONTEXT_MODE**: Defines how the agent uses context from the knowledge base:
  - **"STRICT"**: The agent will answer ONLY based on the provided context. If the answer cannot be found in the context, it will truthfully state that it doesn't know.
  - **"REGULAR"**: The agent will prioritize the provided context from the knowledge base, but if it's insufficient to answer the question, it will use its general knowledge.

### 4. Interact with the AI Assistant
Once your knowledge base is built and your configurations are set, you can interact with the assistant:

a) Command-Line Interface (CLI)
Run the main script to start a text-based conversational interface in your terminal:

```bash
python3 main.py
```
Type exit or quit to end the session.

b) Web User Interface (UI)
For a more user-friendly and interactive experience, use the Streamlit-powered web interface:

```bash
streamlit run app.py
```
This will open the application in your default web browser (usually at http://localhost:8501).


License
This project is licensed under the MIT License - see the LICENSE file for details.