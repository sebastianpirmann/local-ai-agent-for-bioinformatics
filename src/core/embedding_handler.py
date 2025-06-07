from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.embeddings import Embeddings
from typing import List
from config import OLLAMA_EMBEDDING_MODEL


class EmbeddingHandler:
    """
    Handles the creation of text embeddings using the local Ollama Embedding Model.
    """

    def __init__(self, model_name: str = OLLAMA_EMBEDDING_MODEL):
        """
        Initializes the EmbeddingHandler with a specific Ollama embedding model.
        Args:
            model_name (str): The name of the Ollama embedding model to use.
        """
        self.model = OllamaEmbeddings(model=model_name)
        print(f"EmbeddingHandler initialized with model: {model_name}")

    def get_embedding_model(self) -> Embeddings:
        """
        Returns the initialized embeddings model.
        Returns:
            Embeddings: The LangChain embeddings model instance.
        """
        return self.model

    def create_embedding(self, text: str) -> List[float]:
        """
        Creates a single embedding vector for a given text.
        Args:
            text (str): The text to embed.
        Returns:
            List[float]: A list of floats representing the embedding vector.
        """
        try:
            embedding = self.model.embed_query(text)
            return embedding
        except Exception as e:
            print(f"Error creating embedding for text: '{text[:50]}...' Error: {e}")
            return []

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Creates embedding vectors for a list of texts.
        Args:
            texts (List[str]): A list of texts to embed.
        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        try:
            embeddings = self.model.embed_documents(texts)
            return embeddings
        except Exception as e:
            print(f"Error creating embeddings for multiple texts. Error: {e}")
            return [
                [] for _ in texts
            ]  # Return empty lists for each text in case of error


# Example usage for testing purposes
if __name__ == "__main__":
    embedding_handler = EmbeddingHandler()

    # Test with a single text
    print("\n--- Single Embedding Test ---")
    text_to_embed = "Bioinformatics involves the application of computer technology to the management and analysis of biological data."
    embedding = embedding_handler.create_embedding(text_to_embed)
    print(f"Embedding for '{text_to_embed[:60]}...': {embedding[:10]}...")
    print(f"Embedding dimension: {len(embedding)}")

    # Test with multiple texts
    print("\n--- Multiple Embeddings Test ---")
    texts_to_embed = [
        "Genomic sequencing is a key technique.",
        "PCR is used to amplify DNA.",
        "DNA amplification through polymerase chain reaction is crucial.",
    ]
    embeddings = embedding_handler.create_embeddings(texts_to_embed)
    print(f"Number of embeddings created: {len(embeddings)}")
    if embeddings:
        print(f"First embedding (first 10 values): {embeddings[0][:10]}...")
        print(f"Second embedding (first 10 values): {embeddings[1][:10]}...")
        print(f"Third embedding (first 10 values): {embeddings[2][:10]}...")
