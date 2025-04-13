"""Script to ingest compliance documents into ChromaDB vector store."""
import sys
from pathlib import Path
import logging

# Set up logging based on environment variable or default
log_level = logging.getLevelName(logging.INFO) # Default INFO
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path to allow src imports
project_root = Path(__file__).resolve().parent.parent # Use resolve() for robustness
sys.path.insert(0, str(project_root))
logger.info(f"Added project root to sys.path: {project_root}")

# ****** CORRECTED IMPORT ******
from src.utils.document_loader import load_and_split_documents
# ******************************
from src.utils.vector_store import VectorStoreManager
from src.utils.config import DATA_DIR # Import DATA_DIR for clarity

def main():
    """Main function to ingest documents."""
    try:
        # Initialize vector store manager
        logger.info("Initializing vector store manager...")
        # Ensure VectorStoreManager doesn't try to load LLM during ingestion
        # If it does, ingestion script might need separate config or simplified init
        vector_store = VectorStoreManager()
        # Explicitly initialize the store (load existing or create new)
        vector_store.initialize()

        # Load and split documents
        data_dir_path = DATA_DIR # Use configured data directory
        logger.info(f"Loading and splitting documents from {data_dir_path}")
        # ****** USE CORRECT FUNCTION NAME ******
        documents = load_and_split_documents(str(data_dir_path))
        # **************************************
        logger.info(f"Loaded and split into {len(documents)} document chunks")

        # Add documents (chunks) to the vector store
        if documents:
            vector_store.add_documents(documents)
            logger.info("Document chunks successfully ingested into vector store")
        else:
            logger.warning("No documents were found or processed to ingest.")

    except ImportError as ie:
         logger.error(f"Import Error: {ie}. Please ensure all dependencies are installed and PYTHONPATH is correct.", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()