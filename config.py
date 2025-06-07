import os

# Path to the directory containing knowledge base data
DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

# Name of the LLM model, used by Ollama (e.g. "mistral", "llama3")
OLLAMA_LLM_MODEL = "mistral"

# Name of the Embedding model used by Llama ("nomic-embed-text")
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"

# Path to ChromaDB database
# Usually in subfolder '.chroma_db'
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), ".chroma_db")

# Optional: ChromaDB collection name (name for knowledge base)
# If several knowledge bases are available, specify the selected one by name
CHROMA_COLLECTION_NAME = "bio_knowledge_base"

# Maximum number of chunks to be send as context to the LLM
# More chunks can be more relevant but increase processing time and token usage
MAX_CONTEXT_CHUNKS = 5

# Size of the text chunks (in characters)
# A big chunk keeps more context, a small one is more specific
CHUNK_SIZE = 1000

# Overlap between text chunks (in characters)
# Makes sure important context is not lost at chunk boundaries
CHUNK_OVERLAP = 200
