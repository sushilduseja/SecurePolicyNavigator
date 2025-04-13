# Secure Policy Navigator

A RAG-based Compliance Assistant for Security Policies & Standards that helps users navigate and understand complex compliance documentation through natural language queries.

## Features

- Natural language querying of security compliance documents
- Local LLM integration with Microsoft's Phi-2 model
- Efficient document retrieval using RAG architecture
- Clean, intuitive Streamlit interface
- Local vector storage with ChromaDB
- CPU-optimized for environments without GPU
- Interactive policy graph visualization
- Compliance checklist generation
- Multi-document scope control

## Project Structure

```
secure-policy-navigator/
├── data/               # Sample compliance documents
├── scripts/           # Utility scripts (e.g., document ingestion)
├── src/               # Source code
│   ├── components/    # Reusable UI components
│   ├── core/         # Core initialization and setup
│   ├── rag/          # RAG pipeline implementation
│   └── utils/        # Utility functions
├── tests/            # Test files
├── .env              # Environment variables
├── requirements.txt  # Project dependencies
└── README.md        # Project documentation
```

## Setup

1. Ensure Python 3.9+ is installed
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy `.env.example` to `.env` and configure:
   ```
   EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
   LLM_MODEL="microsoft/phi-2"
   DEVICE="cpu"  # Use "cpu" for systems without GPU
   ```

## Document Ingestion

Before running the application, ingest your compliance documents:

```bash
python scripts/ingest_documents.py
```

This will:
- Load documents from the `data/` directory
- Split them into chunks
- Generate embeddings
- Store them in the local ChromaDB vector store

## Running the Application

Start the application with:
```bash
streamlit run src/app.py
```

The interface provides:
- Natural language query interface
- Document scope selection
- Interactive policy graph
- Compliance checklist generator
- Collaboration notes section

## Development

### Adding Documents

1. Place new compliance documents in the `data/` directory (supports .md and .txt)
2. Run the ingestion script to update the vector store
3. Documents are automatically available for querying

### Core Components

- `src/rag/pipeline.py`: RAG pipeline implementation
- `src/utils/vector_store.py`: ChromaDB integration
- `src/utils/document_loader.py`: Document processing
- `src/app.py`: Streamlit interface
- `src/core/`: Core initialization and setup

### Testing

Run the test suite with:
```bash
pytest tests/
```

## License

MIT License