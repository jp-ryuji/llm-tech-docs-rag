# Development

## How to add data under data/raw (example of FastAPI docs)

```bash
git clone https://github.com/tiangolo/fastapi.git temp_fastapi
cp -r temp_fastapi/docs data/raw/fastapi_docs
rm -rf temp_fastapi
```

## `chroma_db/` (ChromaDB Vector Database) and `storage/` (LlamaIndex Index Metadata) directories

### `chroma_db/` - ChromaDB Vector Database

What it stores:

- The actual vector embeddings (numerical representations of your text)
- Vector similarity search indices
- ChromaDB-specific metadata and collections

Files:

```plaintext
chroma_db/
├── chroma.sqlite3         # SQLite database with vectors
├── index/                 # HNSW index files for fast similarity search
└── [collection-uuid]/     # Collection-specific data
```

Purpose: ChromaDB handles the heavy lifting of vector storage and similarity search.

### `storage/` - LlamaIndex Index Metadata

What it stores:

- Index structure and metadata
- Document relationships and graph connections
- Query engine configurations
- Index statistics and mappings

Files:

```plaintext
storage/
├── index_store.json       # Index structure and metadata
├── docstore.json          # Document metadata and chunks
├── vector_store.json      # Vector store configuration
└── graph_store.json       # Knowledge graph data (if used)
```

Purpose: LlamaIndex needs this to reconstruct the index object and understand how documents are organized and connected.

### Why You Need Both

The relationship:

```plaintext
Your Query
    ↓
LlamaIndex (uses storage/)
    ↓ "Find similar documents"
ChromaDB (uses chroma_db/)
    ↓ "Returns relevant chunks"
LlamaIndex (processes results)
    ↓
Final Answer
```

### Best Practices

1. Backup both directories when deploying
1. Keep them synchronized - if you rebuild one, rebuild both
1. Version them together - they're tightly coupled
1. Don't commit to Git - they're generated files (add to .gitignore)
