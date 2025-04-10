from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
VECTOR_STORE_DIR = ROOT_DIR / ".chroma"

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_TOKENS = 4096

# Model settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "mistral:7b-instruct"  # Default Ollama model

# Vector store settings
COLLECTION_NAME = "compliance_docs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True) 