from pathlib import Path

import chromadb
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

# from llama_index.embeddings.openai import OpenAIEmbedding

# Module-level flag to prevent re-configuration
_SETTINGS_CONFIGURED = False


class DocumentIndexer:
    """Creates vector indexes using ChromaDB and Hugging Face embeddings with persistence."""

    def __init__(self, persist_dir="./storage", collection_name="tech_docs"):
        """Initialize the DocumentIndexer with ChromaDB client and Hugging Face embeddings.

        Args:
            persist_dir (str): Directory to persist the index data.
            collection_name (str): The name of the ChromaDB collection to use or create.
        """
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")

        # Configure LlamaIndex settings (only once)
        self._configure_settings()

    def _configure_settings(self):
        """Configure LlamaIndex settings once per module load."""
        global _SETTINGS_CONFIGURED

        if not _SETTINGS_CONFIGURED:
            # Settings.llm = Ollama(model="deepseek-r1:latest", temperature=0.1)
            # Settings.llm = Ollama(model="llama3.2:latest", temperature=0.1)
            Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
            Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            _SETTINGS_CONFIGURED = True
            print("⚙️ LlamaIndex settings configured")

    def index_exists(self):
        """Check if a persisted index already exists.

        Returns:
            bool: True if index exists, False otherwise.
        """
        return (self.persist_dir / "index_store.json").exists()

    def load_existing_index(self):
        """Load an existing index from disk.

        Returns:
            VectorStoreIndex or None: The loaded index, or None if loading fails.
        """
        if not self.index_exists():
            print("ℹ️ No existing index found")
            return None

        try:
            # Load vector store
            chroma_collection = self.chroma_client.get_collection(self.collection_name)
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

            # Load index from storage
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=str(self.persist_dir)
            )
            index = load_index_from_storage(storage_context)
            print("✅ Successfully loaded existing index from storage")
            return index

        except Exception as e:
            print(f"❌ Error loading existing index: {e}")
            print("🔄 Will create a new index instead")
            return None

    def create_new_index(self, documents):
        """Create a new index from documents and persist it.

        Args:
            documents: List of documents to index.

        Returns:
            VectorStoreIndex: The newly created index.
        """
        print(f"🔄 Creating new index from {len(documents)} documents...")

        # Create vector store
        chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Create storage context
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Create index
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        # Persist to disk
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        index.storage_context.persist(persist_dir=self.persist_dir)

        print("✅ Successfully created and persisted new index")
        return index

    def get_or_create_index(self, documents=None):
        """Smart loading: use existing index if available, create new one if not.

        Args:
            documents: List of documents to index (only needed if creating new index).

        Returns:
            VectorStoreIndex: The loaded or newly created index.

        Raises:
            ValueError: If no existing index found and no documents provided.
        """
        # Try to load existing index first
        index = self.load_existing_index()

        if index is not None:
            return index

        # Create new index if none exists
        if documents is None:
            raise ValueError(
                "No existing index found and no documents provided. "
                "Please provide documents to create a new index."
            )

        return self.create_new_index(documents)

    def create_index(self, documents):
        """Legacy method for backward compatibility.

        Args:
            documents: List of documents to index.

        Returns:
            VectorStoreIndex: The created or loaded index.
        """
        return self.get_or_create_index(documents)
