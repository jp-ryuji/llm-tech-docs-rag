# LLM Tech Documentation RAG System

This project implements a Retrieval-Augmented Generation (RAG) system for technical documentation using Large Language Models.

## Features

- Document loading from markdown files
- Vector indexing with ChromaDB
- Question answering over technical documentation
- Evaluation framework for assessing response quality
- Cost-effective embeddings using Hugging Face models
- Interactive web interface using Streamlit

## Directory Structure

```plaintext
src/ # Source code for the RAG system
  ├── data_loader.py         # Loads documents from the data directory
  ├── indexer.py             # Creates and manages the vector index
  ├── query_engine.py        # Handles queries against the indexed documents
  └── evaluator.py           # Evaluates the quality of responses

data/ # Data storage for documentation files

chroma_db/ # ChromaDB Vector Database

storage/ # LlamaIndex Index Metadata
  ├── index_store.json         # Index structure and metadata
  ├── docstore.json            # Document metadata and chunks
  ├── image__vector_store.json # Vector store configuration
  └── graph_store.json         # Knowledge graph data (if used)

app.py # Streamlit web application for interactive Q&A
.env.example # Sample environment variables file
```

## Prerequisites

Before setting up the project, ensure you have the following installed:

1. **uv** - Python package manager and project manager:

   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Alternatively, if you have Python installed:

   ```bash
   pip install uv
   ```

## Setup

After cloning the repository:

1. Install project dependencies:

   ```bash
   uv sync
   ```

2. Activate the virtual environment:

   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Set up environment variables:

   ```bash
   cp .env.example .env
   ```

   Edit the `.env` file to include your API keys and configuration variables.

## Usage

### Manual Execution

To run the interactive Streamlit web application:

```bash
uv streamlit run app.py
```

The web interface provides a user-friendly way to ask questions about the documentation and view confidence scores with source information.

## Development

### Dependency Management

This project uses `pyproject.toml` for dependency management with uv.

### Additional Development Information

For more detailed development information, please refer to [Development Documentation](docs/development.md).
