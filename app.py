import os

import streamlit as st
from dotenv import load_dotenv

from src.data_loader import DocumentLoader
from src.indexer import DocumentIndexer
from src.query_engine import SmartQueryEngine

load_dotenv()

@st.cache_resource
def load_system():
    """
    Load and initialize the document processing system.

    This function loads documents from the FastAPI documentation,
    creates a searchable index, and initializes a query engine.

    The result is cached using Streamlit's caching mechanism to avoid
    reloading the documents and recreating the index on every run.

    Returns:
        SmartQueryEngine: An initialized query engine ready to answer questions.
    """
    # Load documents
    loader = DocumentLoader("data/raw/fastapi_docs")
    documents = loader.load_documents()

    # Create index
    indexer = DocumentIndexer()
    index = indexer.create_index(documents)

    # Create query engine
    query_engine = SmartQueryEngine(index)

    return query_engine

def main():
    """
    Main application function for the Tech Docs Q&A Assistant.

    Sets up the Streamlit UI, initializes the query system, and handles
    user interactions. Users can input questions about FastAPI documentation
    and receive AI-generated answers with confidence scores and sources.
    """
    st.title("📚 Tech Docs Q&A Assistant")
    st.write("Ask questions about FastAPI documentation!")

    # Initialize system
    query_engine = load_system()

    # Query input
    question = st.text_input("Your question:", placeholder="How do I create a FastAPI endpoint?")

    if question:
        with st.spinner("Searching documentation..."):
            result = query_engine.query(question)

        # Display answer
        st.subheader("Answer")
        st.write(result["answer"])

        # Display confidence
        st.metric("Confidence Score", f"{result['confidence']:.2f}")

        # Display sources
        st.subheader("Sources")
        for i, source in enumerate(result["sources"]):
            with st.expander(f"Source {i+1} (Score: {source['score']:.2f})"):
                st.write(f"**Section:** {source['section']}")
                st.write(f"**File:** {source['source']}")
                st.write(f"**Content:** {source['content']}")

if __name__ == "__main__":
    main()
