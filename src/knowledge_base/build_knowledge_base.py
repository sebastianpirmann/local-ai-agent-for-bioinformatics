from src.utils.document_processor import DocumentProcessor
from src.core.embedding_handler import EmbeddingHandler
from src.knowledge_base.vector_database import VectorDatabase
from config import (
    DATA_PATH,
    OLLAMA_EMBEDDING_MODEL,
    CHROMA_DB_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
import os
import shutil  # For cleaning up previous DB during testing/rebuilds


def build_knowledge_base():
    """
    Builds the knowledge base by loading documents, processing them,
    generating embeddings, and storing them in the ChromaDB.
    """
    print("\n--- Starting Knowledge Base Build ---")

    # Initialize Document Processor
    processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Load Documents
    print(f"Loading documents from: {DATA_PATH}")
    documents = processor.load_documents_from_directory(DATA_PATH)
    if not documents:
        print(
            "No documents found in the data directory. Please add your documents to the 'data/' folder."
        )
        return

    # Split Documents into Chunks
    chunks = processor.split_documents(documents)
    if not chunks:
        print("No chunks generated from documents.")
        return

    # Initialize Embedding Handler
    print(f"Initializing embedding model: {OLLAMA_EMBEDDING_MODEL}")
    embedding_handler = EmbeddingHandler(OLLAMA_EMBEDDING_MODEL)
    embeddings_model = embedding_handler.get_embedding_model()

    # Initialize and Populate Vector Database (ChromaDB)
    print("Initializing ChromaDB...")

    # Clean up any previous ChromaDB instance to ensure a fresh build
    if os.path.exists(CHROMA_DB_PATH):
        print(f"Deleting existing ChromaDB at {CHROMA_DB_PATH} for a fresh build.")
        shutil.rmtree(CHROMA_DB_PATH)
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)

    db_handler = VectorDatabase(
        embeddings_model=embeddings_model, persist_directory=CHROMA_DB_PATH
    )

    # Create a new database with all chunks
    db = db_handler.initialize_db(chunks)

    if db:
        print("Knowledge Base (ChromaDB) built successfully!")
        print(f"Database stored at: {CHROMA_DB_PATH}")
    else:
        print("Failed to build Knowledge Base.")

    print("--- Knowledge Base Build Finished ---")


if __name__ == "__main__":

    # Ensure that 'data/' exists and create a dummy file if empty for test
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Created data directory: {DATA_PATH}")

    # Check if DATA_PATH is empty and create a dummy file if so
    if not os.listdir(DATA_PATH):
        dummy_file_path = os.path.join(DATA_PATH, "dummy_bio_info.txt")
        with open(dummy_file_path, "w") as f:
            f.write(
                "This is a dummy bioinformatics text for testing the knowledge base build process. "
                "It talks about DNA sequencing and protein synthesis. "
                "Sequencing helps understand genetic code, while protein synthesis involves translation from RNA."
            )
        print(f"Created dummy file: {dummy_file_path} for testing.")

    try:
        build_knowledge_base()
    finally:
        # Clean up the dummy file after test if it was created by this script
        if "dummy_file_path" in locals() and os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)
            print(f"Removed dummy file: {dummy_file_path}")
