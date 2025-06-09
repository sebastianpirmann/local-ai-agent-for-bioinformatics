import os
import sys
from src.agents.bio_rag_agent import BioRAGAgent
from config import CHROMA_DB_PATH


def ensure_knowledge_base_exists():
    """
    Checks if the ChromaDB knowledge base directory exists and is not empty.
    If not, prompts the user to build it.
    """
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        print("\n-----------------------------------------------------")
        print("WARNING: Knowledge Base (ChromaDB) not found or is empty!")
        print("Please build it first by running:")
        print("  python3 -m src.knowledge_base.build_knowledge_base")
        print("Ensure you have documents in your 'data/' directory.")
        print("-----------------------------------------------------\n")
        return False
    return True


def main():
    """
    Main function to run the Bio-AI Assistant CLI.
    """
    # First, ensure the knowledge base is built
    if not ensure_knowledge_base_exists():
        sys.exit(1)  # Exit if KB is not found

    print("Initializing Bio-AI Assistant. This might take a moment...")
    try:
        agent = BioRAGAgent()
    except ValueError as e:
        print(f"Error initializing agent: {e}")
        print("Please check your Ollama installation and ensure models were pulled.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during agent initialization: {e}")
        sys.exit(1)

    print("\nBio-AI Assistant Ready! Type your questions below.")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        try:
            question = input("\nYour question: ").strip()

            if question.lower() in ["exit", "quit"]:
                print("Exiting Bio-AI Assistant. Goodbye!")
                break

            if not question:
                print("Please enter a question.")
                continue

            answer = agent.query_agent(question)
            print(f"Assistant: {answer}")

        except KeyboardInterrupt:
            print("\nExiting Bio-AI Assistant. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred during query: {e}")
            print("Please check if Ollama is still running and models are available.")


if __name__ == "__main__":
    main()
