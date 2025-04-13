"""Script to ingest compliance documents into ChromaDB vector store."""
import sys
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.utils.document_loader import load_documents
from src.utils.vector_store import VectorStoreManager

def main():
    """Main function to ingest documents."""
    try:
        # Initialize vector store manager
        logger.info("Initializing vector store...")
        vector_store = VectorStoreManager()
        
        # Load documents
        data_dir = project_root / "data"
        logger.info(f"Loading documents from {data_dir}")
        documents = load_documents(str(data_dir))
        logger.info(f"Loaded {len(documents)} documents")
        
        # Initialize vector store with documents
        if documents:
            vector_store.initialize(documents)
            logger.info("Documents successfully ingested into vector store")
        else:
            logger.warning("No documents were found to ingest")
        
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
