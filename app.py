import streamlit as st
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex

from src.data_loader import DocumentLoader
from src.indexer import DocumentIndexer
from src.query_engine import SmartQueryEngine

load_dotenv()


@st.cache_resource
def initialize_system():
    """
    Initialize the RAG system with smart caching and persistence.

    This function first attempts to load an existing index from disk. If no index
    exists, it loads documents from the FastAPI documentation, creates a new
    searchable index, and persists it for future use.

    The result is cached using Streamlit's caching mechanism to avoid
    reloading on every run. After the first initialization, subsequent
    app starts will be much faster.

    Returns:
        SmartQueryEngine: An initialized query engine ready to answer questions.
    """
    indexer = DocumentIndexer()

    # Try to load existing index first
    index = indexer.load_existing_index()

    if index is None:
        # Only load documents and create index if none exists
        st.info("🔄 First-time setup: Loading and indexing documents...")
        st.write("This may take a few minutes, but subsequent runs will be much faster!")

        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Load documents
            status_text.text("📖 Loading documents from FastAPI docs...")
            progress_bar.progress(20)

            loader = DocumentLoader("data/raw/fastapi_docs")
            documents = loader.load_documents()

            status_text.text(f"✅ Loaded {len(documents)} documents")
            progress_bar.progress(40)

            # Create and persist index
            status_text.text("🔍 Creating search index (this may take a while)...")
            progress_bar.progress(60)

            index = indexer.create_new_index(documents)

            progress_bar.progress(100)
            status_text.text("✅ Setup complete! Index saved for future use.")

            # Show success message
            st.success(
                f"🎉 Successfully indexed {len(documents)} documents! "
                "Future app starts will be much faster."
            )

        except Exception as e:
            st.error(f"❌ Error during setup: {str(e)}")
            st.error("Please check your data directory and try again.")
            st.stop()
    else:
        # Index loaded successfully
        st.success("✅ Loaded existing index - ready to answer questions!")

    # VectorStoreIndex型にキャスト（from_indexは使わず型チェックのみ）
    if not isinstance(index, VectorStoreIndex):
        st.error("Loaded index is not a VectorStoreIndex. Please rebuild the index.")
        st.stop()

    # Create query engine
    query_engine = SmartQueryEngine(index)
    return query_engine


def sidebar_info():
    """Display sidebar information and controls."""
    st.sidebar.title("🔧 System Info")

    # Check if index exists
    indexer = DocumentIndexer()
    if indexer.index_exists():
        st.sidebar.success("✅ Index: Ready")
    else:
        st.sidebar.warning("⚠️ Index: Not found")

    # System controls
    st.sidebar.subheader("Controls")

    if st.sidebar.button("🔄 Rebuild Index"):
        st.sidebar.warning("Clearing cache and rebuilding...")
        # Clear the cached system
        st.cache_resource.clear()
        # Force a rerun to reinitialize
        st.rerun()

    if st.sidebar.button("🗑️ Clear Cache"):
        st.cache_resource.clear()
        st.sidebar.success("Cache cleared!")

    # Usage tips
    st.sidebar.subheader("💡 Tips")
    st.sidebar.write(
        "- First run takes a few minutes to setup\n"
        "- Subsequent runs load instantly\n"
        "- Use 'Rebuild Index' if documents change\n"
        "- Higher confidence scores = better answers"
    )


def main():
    """
    Main application function for the Tech Docs Q&A Assistant.

    Sets up the Streamlit UI, initializes the query system with smart caching,
    and handles user interactions.
    Users can input questions about FastAPI documentation and receive
    AI-generated answers with confidence scores and sources.
    """
    st.set_page_config(page_title="Tech Docs Q&A", page_icon="📚", layout="wide")

    st.title("📚 Tech Docs Q&A Assistant")
    st.markdown(
        "Ask questions about **FastAPI documentation** and get AI-powered answers "
        "with source citations and confidence scores."
    )

    # Add sidebar
    sidebar_info()

    # Initialize system with smart caching
    try:
        query_engine = initialize_system()
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        st.error("Please check your setup and try again.")
        return

    # Create two columns for better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input with better UX
        question = st.text_area(
            "Your question:",
            placeholder="Example: How do I create a FastAPI endpoint with path parameters?",
            height=100,
        )

        # Search button
        search_clicked = st.button("🔍 Search Documentation", type="primary")

    with col2:
        # Example questions
        st.subheader("💡 Example Questions")
        example_questions = [
            "How do I create a basic FastAPI app?",
            "What are path parameters in FastAPI?",
            "What are query parameters in FastAPI?",
            "How do I handle JSON request body?",
            "Does FastAPI provide swagger?",
        ]

        for example in example_questions:
            if st.button(example, key=f"example_{hash(example)}"):
                question = example
                search_clicked = True

    # Process query
    if (question and search_clicked) or (question and st.session_state.get("auto_search", False)):
        with st.spinner("🔍 Searching documentation..."):
            try:
                result = query_engine.query(question)

                # Display answer
                st.subheader("💬 Answer")
                st.write(result["answer"])

                # Display metrics in columns
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.metric("Confidence Score", f"{result['confidence']:.2f}")
                with metric_col2:
                    st.metric("Sources Found", len(result["sources"]))

                # Display sources
                if result["sources"]:
                    st.subheader("📄 Sources")
                    for i, source in enumerate(result["sources"]):
                        with st.expander(
                            f"Source {i + 1}: {source['section']} "
                            f"(Relevance: {source['score']:.2f})"
                        ):
                            st.write(f"**File:** `{source['source']}`")
                            st.write(f"**Section:** {source['section']}")
                            st.write("**Content Preview:**")
                            st.write(source["content"])
                else:
                    st.warning("No relevant sources found. Try rephrasing your question.")

            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                st.error("Please try again or rephrase your question.")


if __name__ == "__main__":
    main()
