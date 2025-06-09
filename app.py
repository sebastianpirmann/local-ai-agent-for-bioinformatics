import streamlit as st
import os
import sys

# Add the src directory to the Python path
# This is necessary when running app.py from the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from src.agents.bio_rag_agent import BioRAGAgent
from config import (
    CHROMA_DB_PATH,
    DATA_PATH,
    CONTEXT_MODE,
    OLLAMA_LLM_MODEL,
)


# Helper to check if knowledge base exists
def ensure_knowledge_base_exists_ui():
    if not os.path.exists(CHROMA_DB_PATH) or not os.listdir(CHROMA_DB_PATH):
        st.error(
            f"Knowledge Base (ChromaDB) not found or is empty at `{CHROMA_DB_PATH}`!"
            "\nPlease build it first by running: `python3 -m src.knowledge_base.build_knowledge_base`"
            f"\nEnsure you have documents in your `data/` directory (`{DATA_PATH}`)."
        )
        st.stop()  # Stop execution of the Streamlit app
    return True


# Initialize the RAG Agent only once per session
@st.cache_resource
def initialize_agent():
    ensure_knowledge_base_exists_ui()
    st.info("Initializing Bioinformatics-AI Assistant. This might take a moment...")
    try:
        agent = BioRAGAgent()
        st.success("Bioinformatics-AI Assistant Initialized!")
        return agent
    except Exception as e:
        st.error(f"Error initializing agent: {e}")
        st.warning(
            "Please ensure Ollama is running and models are pulled (e.g., mistral, nomic-embed-text)."
        )
        st.stop()  # Stop execution if initialization fails


# Streamlit UI
st.set_page_config(page_title="Bio-AI Assistant", page_icon="")
st.title("Bioinformatics-AI Assistant")
st.markdown(
    """
    Ask the Assistant for help with Bioinformatics stuff. 
    """
)

# Display current configuration
st.sidebar.header("Configuration")
st.sidebar.write(f"**LLM Model:** `{OLLAMA_LLM_MODEL}`")
st.sidebar.write(f"**Context Mode:** `{CONTEXT_MODE}`")
st.sidebar.write(f"**Knowledge Base Path:** `{CHROMA_DB_PATH}`")
st.sidebar.write(f"**Data Directory:** `{DATA_PATH}`")


# Initialize agent (this will run only once thanks to @st.cache_resource)
agent = initialize_agent()

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask the Bioinformatics Assistant for help"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Assistant thinking..."):
        # Get response from the RAG agent
        response = agent.query_agent(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

st.sidebar.markdown("---")
st.sidebar.markdown("Made by Sebastian Pirmann")
