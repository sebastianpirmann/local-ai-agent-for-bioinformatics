from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from config import OLLAMA_LLM_MODEL


class LLMHandler:
    """
    Handles interactions with the local Ollama Large Language Model (LLM).
    """

    def __init__(self, model_name: str = OLLAMA_LLM_MODEL):
        """
        Initializes the LLMHandler with a specific Ollama model.
        Args:
            model_name (str): The name of the Ollama model to use.
        """
        self.model_name = model_name
        try:
            self.model = ChatOllama(model=self.model_name)
        except Exception as e:
            print(f"Error initializing Ollama LLM model '{self.model_name}': {e}")
            raise
        print(f"LLMHandler initialized with model: {self.model_name}")

    def get_llm(self) -> BaseChatModel:
        """
        Returns the initialized LLM model.
        Returns:
            BaseChatModel: The LangChain LLM model instance.
        """
        return self.model

    def generate_response(self, prompt: str, system_message: str = None) -> str:
        """
        Generates a response from the LLM based on the given prompt and an optional system message.
        Args:
            prompt (str): The user's prompt or question.
            system_message (str, optional): A system message to guide the LLM's behavior. Defaults to None.
        Returns:
            str: The generated response from the LLM.
        """
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))
        messages.append(HumanMessage(content=prompt))

        try:
            response = self.model.invoke(messages)
            return response.content
        except Exception as e:
            print(f"Error generating response from LLM: {e}")
            return "An error occurred while generating the response."


# Example usage (for testing purposes, will be removed later)
if __name__ == "__main__":
    llm_handler = LLMHandler()

    # Test with a simple prompt
    print("\n--- Simple Prompt Test ---")
    response = llm_handler.generate_response("What is the capital of France?")
    print(f"LLM Response: {response}")

    # Test with a system message
    print("\n--- System Message Test ---")
    system_msg = "You are a helpful assistant specialized in bioinformatics. Provide concise answers."
    response = llm_handler.generate_response("Explain PCR.", system_message=system_msg)
    print(f"LLM Response (Bioinfo focus): {response}")
