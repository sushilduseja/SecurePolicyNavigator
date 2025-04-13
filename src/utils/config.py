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
LLM_MODEL = "facebook/opt-125m"  # Changed to a smaller model for testing
MAX_NEW_TOKENS = 256
MAX_INPUT_LENGTH = 4096
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K = 50
REPETITION_PENALTY = 1.2

# LLM Configuration
LLM_CONFIG = {
    "do_sample": True,  # Enable sampling
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repetition_penalty": REPETITION_PENALTY,
    "max_new_tokens": MAX_NEW_TOKENS,  # Keep only max_new_tokens
    "truncation": True,  # Add explicit truncation
    "max_length": MAX_INPUT_LENGTH,
    "return_full_text": False,  # Only return new tokens
    "pad_token_id": None,  # Will be set at runtime
    "eos_token_id": None   # Will be set at runtime
}

# Vector store settings
COLLECTION_NAME = "compliance_docs"

# Debug settings
DEBUG = False

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)