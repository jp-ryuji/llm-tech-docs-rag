import chromadb
from llama_index.core import Settings, VectorStoreIndex
# from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore


# pylint: disable=too-few-public-methods
class DocumentIndexer:
    """Creates vector indexes using ChromaDB and Hugging Face embeddings."""

    def __init__(self, collection_name="tech_docs"):
        """Initialize the DocumentIndexer with ChromaDB client and Hugging Face embeddings.

        Args:
            collection_name (str): The name of the ChromaDB collection to use or create.
        """
        # Configure LlamaIndex settings
        Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
        # Comment out OpenAI embedding model
        # Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        # Enable free Hugging Face embedding model
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

        # Setup vector store
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.chroma_collection = self.chroma_client.get_or_create_collection(collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)

    def create_index(self, documents):
        """Creates a vector store index from a list of documents.

        Args:
            documents: A list of documents to be indexed.

        Returns:
            The created VectorStoreIndex instance.
        """
        index = VectorStoreIndex.from_documents(
            documents,
            vector_store=self.vector_store
        )
        return index
