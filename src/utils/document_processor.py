from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP
import os
from typing import List


class DocumentProcessor:
    """
    Handles loading and splitting of documents for the knowledge base.
    """

    def __init__(
        self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
    ):
        """
        Initializes the DocumentProcessor with specified chunking parameters.
        Args:
            chunk_size (int): The maximum size of each text chunk.
            chunk_overlap (int): The number of characters to overlap between chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        print(
            f"DocumentProcessor initialized with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def load_documents_from_directory(self, directory_path: str) -> List[Document]:
        """
        Loads all supported documents (PDF, TXT) from a given directory.
        Args:
            directory_path (str): The path to the directory containing documents.
        Returns:
            List[Document]: A list of loaded LangChain Document objects.
        """
        loaded_documents = []
        if not os.path.isdir(directory_path):
            print(f"Error: Directory not found at {directory_path}")
            return []

        for root, _, files in os.walk(directory_path):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_name.endswith(".pdf"):
                    try:
                        loader = PyPDFLoader(file_path)
                        loaded_documents.extend(loader.load())
                        print(f"Loaded PDF: {file_name}")
                    except Exception as e:
                        print(f"Error loading PDF {file_name}: {e}")
                elif file_name.endswith(
                    (".txt", ".md", ".py", ".R", ".sh")
                ):  # Add more as needed
                    try:
                        loader = TextLoader(file_path, encoding="utf-8")
                        loaded_documents.extend(loader.load())
                        print(f"Loaded Text/Code: {file_name}")
                    except Exception as e:
                        print(f"Error loading text/code file {file_name}: {e}")
                else:
                    print(f"Skipping unsupported file type: {file_name}")
                    pass

        print(f"Total documents loaded: {len(loaded_documents)}")
        return loaded_documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Splits a list of LangChain Documents into smaller chunks.
        Args:
            documents (List[Document]): A list of LangChain Document objects.
        Returns:
            List[Document]: A list of smaller, chunked LangChain Document objects.
        """
        if not documents:
            print("No documents to split.")
            return []

        chunks = self.text_splitter.split_documents(documents)
        print(f"Original documents split into {len(chunks)} chunks.")
        return chunks


# Example usage for testing purposes
if __name__ == "__main__":
    # Create a dummy data directory and some files for testing
    test_data_dir = "temp_test_data"
    os.makedirs(test_data_dir, exist_ok=True)

    with open(os.path.join(test_data_dir, "test_bio_notes.txt"), "w") as f:
        f.write(
            "Bioinformatics is an interdisciplinary field that develops methods and software tools for understanding biological data. "
            "It combines computer science, statistics, mathematics, and engineering to analyze and interpret biological data. "
            "Key areas include sequence analysis, genomics, proteomics, and transcriptomics."
        )

    with open(os.path.join(test_data_dir, "test_code.py"), "w") as f:
        f.write(
            "def calculate_gc_content(sequence):\n"
            "    gc_count = sequence.count('G') + sequence.count('C')\n"
            "    return (gc_count / len(sequence)) * 100 if sequence else 0\n"
            "\n"
            "dna_sequence = 'AGCTCGAGCT'\n"
            "print(f'GC Content: {calculate_gc_content(dna_sequence)}%')"
        )

    # Note: For PDF testing, we need a real PDF file.

    processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)

    print("\n--- Loading Documents Test ---")
    # Ensure DATA_PATH is set correctly in config.py or use the test_data_dir here

    loaded_docs = processor.load_documents_from_directory(test_data_dir)

    print("\n--- Splitting Documents Test ---")
    if loaded_docs:
        chunks = processor.split_documents(loaded_docs)
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1} (Length: {len(chunk.page_content)}):")
            print(f"  Content: '{chunk.page_content[:200]}...'")
            print(f"  Source: {chunk.metadata.get('source', 'N/A')}")
    else:
        print("No documents loaded to split.")

    # Clean up dummy data
    os.remove(os.path.join(test_data_dir, "test_bio_notes.txt"))
    os.remove(os.path.join(test_data_dir, "test_code.py"))
    os.rmdir(test_data_dir)
    print(f"\nCleaned up dummy test data in '{test_data_dir}' directory.")
