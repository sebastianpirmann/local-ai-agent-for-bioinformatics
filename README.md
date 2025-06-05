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
personal-bio-ai-assistant/  (Your project root)
├── .venv/                   # Python Virtual Environment
├── .vscode/                 # VS Code specific settings
│   └── settings.json        # Editor and Python interpreter configuration
├── data/                    # Directory for your personal bioinformatics documents (PDFs, .txt, .md, code, etc.)
├── src/                     # Source code for the AI assistant
│   ├── core/                # Core functionalities: LLM and Embedding Handlers
│   │   ├── llm_handler.py
│   │   └── embedding_handler.py
│   ├── utils/               # Utility functions: Document loading & processing
│   │   └── document_processor.py
│   └── agents/              # Main agent logic and orchestration
├── .chroma_db/              # Local ChromaDB persistent storage (auto-generated)
├── config.py                # Centralized project configuration (model names, chunk sizes, paths)
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

git clone [https://github.com/YOUR_GITHUB_USERNAME/personal-ai-agent-for-bioinformatics.git](https://github.com/YOUR_GITHUB_USERNAME/personal-ai-agent-for-bioinformatics.git)
cd personal-ai-agent-for-bioinformatics
(Replace YOUR_GITHUB_USERNAME with your actual GitHub username and adjust the repository name if you named it differently.)

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

pip install -r requirements.txt

## 5. Configure VS Code (Recommended)
To ensure VS Code correctly uses your virtual environment and understands project imports:

Open the project folder (personal-ai-agent-for-bioinformatics) in VS Code via File > Open Folder....

Open the Command Palette (Ctrl/Cmd + Shift + P) and select Python: Select Interpreter. Choose the interpreter located within your project's .venv folder (e.g., Python 3.x.x (.venv)).

Ensure a .vscode/settings.json file exists in your project root with the following content. If it doesn't exist, create the .vscode/ folder and then settings.json inside it.

JSON

{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.analysis.extraPaths": [
        "./src"
    ]
}
Restart VS Code after making these changes.

# Usage
### 1. Prepare Your Knowledge Base
Place your bioinformatics documents (PDFs, .txt, .md, .py, .R, .sh files, etc.) into the data/ directory.

### 2. Build the Knowledge Base (Vector Database)
(This step is to be implemented. It will involve running a script to process the data/ files and populate chromadb.)

python3 -m src.utils.build_knowledge_base

### 3. Interact with the AI Assistant
(This will be the final step, involving a command-line interface or a simple web UI for asking questions.)


License
This project is licensed under the MIT License - see the LICENSE file for details.