from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from config import CHROMA_DB_PATH
from typing import List, Optional
import os


class VectorDatabase:
    """
    Handles interactions with the ChromaDB vector store.
    """

    def __init__(
        self, embeddings_model: Embeddings, persist_directory: str = CHROMA_DB_PATH
    ):
        """
        Initializes the VectorDatabase with an embedding model and a persistence directory.
        Args:
            embeddings_model (Embeddings): The embedding model to use for generating vectors.
            persist_directory (str): The directory where ChromaDB will store its data.
        """
        self.embeddings_model = embeddings_model
        self.persist_directory = persist_directory
        # Initialize Chroma client directly, will be used to create/load collection
        self.db = None
        print(
            f"VectorDatabase initialized. Data will be stored in: {self.persist_directory}"
        )

    def initialize_db(self, documents: Optional[List[Document]] = None) -> Chroma:
        """
        Initializes or loads the ChromaDB collection. If documents are provided,
        it creates a new collection from them. Otherwise, it attempts to load
        an existing collection.
        Args:
            documents (Optional[List[Document]]): List of documents to use for
                                                 creating a new collection.
        Returns:
            Chroma: The initialized or loaded ChromaDB instance.
        """
        if documents:
            print(f"Creating new ChromaDB from {len(documents)} documents...")
            # Create a new collection from documents
            self.db = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings_model,
                persist_directory=self.persist_directory,
            )
            self.db.persist()  # Ensure the changes are written to disk
            print("ChromaDB created and persisted.")
        else:
            # Attempt to load existing collection
            if os.path.exists(self.persist_directory) and os.listdir(
                self.persist_directory
            ):
                print(f"Loading existing ChromaDB from {self.persist_directory}...")
                self.db = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings_model,
                )
                print("ChromaDB loaded.")
            else:
                print(
                    f"No existing ChromaDB found at {self.persist_directory}. Please provide documents to create one."
                )
                self.db = None  # Explicitly set to None if no DB is found/created
        return self.db

    def add_documents(self, documents: List[Document]):
        """
        Adds new documents to an existing ChromaDB collection.
        If the database hasn't been initialized, it will initialize it with these documents.
        Args:
            documents (List[Document]): List of new documents to add.
        """
        if not documents:
            print("No documents provided to add.")
            return

        if self.db is None:
            print("Database not initialized. Initializing with provided documents.")
            self.initialize_db(documents)
        else:
            print(f"Adding {len(documents)} new documents to ChromaDB...")
            self.db.add_documents(documents)
            self.db.persist()  # Ensure the changes are written to disk
            print("Documents added and ChromaDB persisted.")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """
        Performs a similarity search in the ChromaDB.
        Args:
            query (str): The query string to search for.
            k (int): The number of most similar documents to retrieve.
        Returns:
            List[Document]: A list of retrieved LangChain Document objects.
        """
        if self.db is None:
            print("Database not initialized. Cannot perform similarity search.")
            return []

        print(f"Performing similarity search for query: '{query}'")
        results = self.db.similarity_search(query, k=k)
        print(f"Found {len(results)} relevant documents.")
        return results


# Example usage (for testing purposes, will be removed later)
if __name__ == "__main__":
    from src.core.embedding_handler import EmbeddingHandler
    from src.utils.document_processor import DocumentProcessor
    from config import OLLAMA_EMBEDDING_MODEL, CHROMA_DB_PATH
    import shutil  # For cleaning up test directory

    # --- Setup for testing ---
    # 1. Clean up any previous test database
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Cleaned up previous ChromaDB at {CHROMA_DB_PATH}")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)  # Ensure directory exists for Chroma

    # 2. Initialize Embedding Handler
    embedding_handler = EmbeddingHandler(OLLAMA_EMBEDDING_MODEL)
    embeddings_model = embedding_handler.get_embedding_model()

    # 3. Create dummy documents for testing
    processor = DocumentProcessor(
        chunk_size=100, chunk_overlap=20
    )  # Smaller chunks for demo
    test_data_for_db_dir = "temp_data_for_db_test"
    os.makedirs(test_data_for_db_dir, exist_ok=True)
    with open(os.path.join(test_data_for_db_dir, "test_article_part1.txt"), "w") as f:
        f.write(
            "CRISPR-Cas9 is a revolutionary gene-editing tool derived from a bacterial immune system. "
            "It allows scientists to make precise changes to DNA. This technology has wide applications "
            "in genetic engineering, disease research, and gene therapy. Its simplicity and accuracy "
            "have accelerated biological research significantly. "
            "Another topic is DNA replication, which is the biological process of producing two identical replicas of DNA from one original DNA molecule. This process occurs in all living organisms and is the basis for biological inheritance."
        )
    with open(os.path.join(test_data_for_db_dir, "test_article_part2.txt"), "w") as f:
        f.write(
            "PCR, or Polymerase Chain Reaction, is a widely used molecular biology technique to make "
            "many copies of a specific DNA segment. It's essential for various applications like "
            "forensics, genetic testing, and research. The process involves denaturation, annealing, and extension cycles."
        )

    loaded_docs_for_db = processor.load_documents_from_directory(test_data_for_db_dir)
    chunks_for_db = processor.split_documents(loaded_docs_for_db)
    print(f"Prepared {len(chunks_for_db)} chunks for the database.")

    # --- Testing VectorDatabase functionalities ---
    print("\n--- Initializing and Adding Documents to ChromaDB ---")
    db_handler = VectorDatabase(
        embeddings_model=embeddings_model, persist_directory=CHROMA_DB_PATH
    )

    # Test 1: Create DB with documents
    initial_db = db_handler.initialize_db(chunks_for_db)
    if initial_db:
        print("Initial DB creation successful.")

    # Test 2: Add more documents to existing DB
    new_doc_content = "RNA sequencing (RNA-Seq) is a revolutionary technology that uses next-generation sequencing to reveal the presence and quantity of RNA in a biological sample at a given moment in time."
    new_document = Document(
        page_content=new_doc_content, metadata={"source": "test_new_doc.txt"}
    )
    db_handler.add_documents([new_document])

    # Test 3: Load existing DB (should not create new one)
    print("\n--- Loading Existing ChromaDB ---")
    loaded_db_handler = VectorDatabase(
        embeddings_model=embeddings_model, persist_directory=CHROMA_DB_PATH
    )
    existing_db = (
        loaded_db_handler.initialize_db()
    )  # Should load existing, not create new
    if existing_db:
        print("Existing DB loaded successfully.")

    # Test 4: Perform similarity search
    print("\n--- Performing Similarity Search ---")
    query_text_1 = "What is gene editing and its uses?"
    results_1 = loaded_db_handler.similarity_search(query_text_1, k=2)
    print(f"\nResults for '{query_text_1}':")
    for i, doc in enumerate(results_1):
        print(
            f"  Result {i+1} (Source: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:150]}..."
        )

    query_text_2 = "Explain PCR method in detail."
    results_2 = loaded_db_handler.similarity_search(query_text_2, k=1)
    print(f"\nResults for '{query_text_2}':")
    for i, doc in enumerate(results_2):
        print(
            f"  Result {i+1} (Source: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:150]}..."
        )

    query_text_3 = "What is RNA-Seq?"
    results_3 = loaded_db_handler.similarity_search(query_text_3, k=1)
    print(f"\nResults for '{query_text_3}':")
    for i, doc in enumerate(results_3):
        print(
            f"  Result {i+1} (Source: {doc.metadata.get('source', 'N/A')}): {doc.page_content[:150]}..."
        )

    # --- Cleanup ---
    os.remove(os.path.join(test_data_for_db_dir, "test_article_part1.txt"))
    os.remove(os.path.join(test_data_for_db_dir, "test_article_part2.txt"))
    os.rmdir(test_data_for_db_dir)
    print(f"\nCleaned up dummy data in '{test_data_for_db_dir}' directory.")

    # Clean up the created ChromaDB
    if os.path.exists(CHROMA_DB_PATH):
        shutil.rmtree(CHROMA_DB_PATH)
        print(f"Cleaned up ChromaDB at {CHROMA_DB_PATH}")
