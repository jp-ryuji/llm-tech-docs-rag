from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever


# pylint: disable=too-few-public-methods
class SmartQueryEngine:
    """A smart query engine that processes natural language questions against indexed documents.

    This class implements a retrieval-augmented generation (RAG) approach using LlamaIndex
    components to retrieve relevant document sections and generate informed responses.
    It includes similarity filtering to improve response quality.
    """

    def __init__(self, index: VectorStoreIndex):
        """Initialize the SmartQueryEngine with a vector store index.

        Args:
            index (VectorStoreIndex): The vector store index to query against.
        """
        self.index = index
        self._setup_query_engine()

    def _setup_query_engine(self):
        """Configure the retriever, post-processor, and query engine components.

        Sets up a retriever with top-k similarity search, a similarity post-processor
        to filter low-confidence results, and a retriever query engine that combines
        these components.
        """
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )

        # Add post-processor for filtering low-confidence results
        postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)

        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor]
        )

    def query(self, question: str) -> dict:
        """Process a natural language question and return a structured response.

        Args:
            question (str): The natural language question to process.

        Returns:
            dict: A dictionary containing:
                - answer (str): The generated response to the question
                - sources (list): List of source documents with metadata
                - confidence (float): Average similarity score of source documents
        """
        response = self.query_engine.query(question)

        # Format response with sources
        sources = []
        for node in response.source_nodes:
            sources.append({
                "content": node.text[:200] + "...",
                "source": node.metadata.get("source", "Unknown"),
                "score": node.score,
                "section": node.metadata.get("section", "general")
            })

        return {
            "answer": str(response),
            "sources": sources,
            "confidence": self._calculate_confidence(response.source_nodes)
        }

    def _calculate_confidence(self, source_nodes) -> float:
        """Calculate the average similarity score of the source nodes as a confidence measure.

        Args:
            source_nodes (list): List of source nodes from the query response.

        Returns:
            float: Average similarity score rounded to 2 decimal places, or 0.0 if no nodes.
        """
        if not source_nodes:
            return 0.0
        avg_score = sum(node.score for node in source_nodes) / len(source_nodes)
        return round(avg_score, 2)
