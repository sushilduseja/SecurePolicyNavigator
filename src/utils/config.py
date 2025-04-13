import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

# Project paths
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / os.getenv("DATA_PATH", "data")
VECTOR_STORE_DIR = ROOT_DIR / os.getenv("VECTOR_STORE_PATH", ".chroma")

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
# Max results to retrieve from vector store (UI controls this)
DEFAULT_K = int(os.getenv("DEFAULT_K", 3))

# Model settings
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# Use 'cuda' for GPU or 'cpu'. Ensure torch is installed with CUDA support if using 'cuda'.
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
# Use smaller model optimized for instruction following
LLM_MODEL = os.getenv("LLM_MODEL", "microsoft/phi-2")
# Chain type for RetrievalQA ('stuff', 'map_reduce', 'refine')
# 'stuff' is simplest but fails if context > LLM limit.
# 'map_reduce' and 'refine' handle larger contexts but use more LLM calls.
RAG_CHAIN_TYPE = os.getenv("RAG_CHAIN_TYPE", "stuff")

# LLM Generation settings
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", 256))  # Reduced from 512
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
TOP_P = float(os.getenv("TOP_P", 0.9))
TOP_K = int(os.getenv("TOP_K", 50))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 1.2))

# LLM Configuration dictionary
LLM_CONFIG = {
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "top_k": TOP_K,
    "repetition_penalty": REPETITION_PENALTY,
    "max_new_tokens": MAX_NEW_TOKENS,
    "do_sample": True,
    "pad_token_id": 50256  # Common pad token id for most models
}

# Vector store settings
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "compliance_docs")

# Debug and Logging settings
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
# Configure logging format
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Ensure directories exist
try:
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Vector store directory: {VECTOR_STORE_DIR}")
except OSError as e:
    logger.error(f"Error creating directories: {e}")

# Log key configurations
logger.info(f"Embedding Model: {EMBEDDING_MODEL} on {EMBEDDING_DEVICE}")
logger.info(f"LLM Model: {LLM_MODEL}")
logger.info(f"RAG Chain Type: {RAG_CHAIN_TYPE}")
logger.info(f"Chunk Size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")