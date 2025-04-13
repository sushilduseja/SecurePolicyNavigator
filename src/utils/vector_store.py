import logging
from typing import List, Dict, Any, Optional
import shutil
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
import chromadb
from chromadb.config import Settings
from langchain_community.vectorstores.utils import filter_complex_metadata

from src.utils.config import (
    VECTOR_STORE_DIR,
    EMBEDDING_MODEL,
    EMBEDDING_DEVICE,
    COLLECTION_NAME,
    DEBUG
)
from src.utils.error_handler import VectorStoreError

logger = logging.getLogger(__name__)

class VectorStoreManager:
    def __init__(self):
        """Initialize vector store manager and embeddings."""
        self.persist_directory = str(VECTOR_STORE_DIR)
        self.collection_name = COLLECTION_NAME
        self.vector_store: Optional[Chroma] = None
        self.is_initialized = False
        self.embeddings = None
        self.client = None

        # Create vector store directory if it doesn't exist
        VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)

        logger.info(f"Initializing embeddings ({EMBEDDING_MODEL}) on device: {EMBEDDING_DEVICE}")
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': EMBEDDING_DEVICE},
                encode_kwargs={'normalize_embeddings': True}
            )
            if not self.embeddings:
                raise VectorStoreError("Failed to initialize embeddings")
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace embeddings: {e}", exc_info=DEBUG)
            raise VectorStoreError("Could not initialize embeddings model.") from e

    def initialize(self, documents: Optional[List[Document]] = None) -> None:
        """Initialize the vector store."""
        if self.is_initialized and self.vector_store is not None:
            logger.info("Vector store already initialized.")
            return

        logger.info("Initializing VectorStoreManager...")
        try:
            # Initialize ChromaDB with basic settings
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )

            # Create a new client instance
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=settings
            )

            # Get or create collection with metadata
            collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "embedding_function": EMBEDDING_MODEL}
            )

            # Initialize Chroma with the collection
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
            )

            if documents:
                logger.info(f"Adding {len(documents)} documents to vector store...")
                self.add_documents(documents)

            self.is_initialized = True
            logger.info(f"Vector store initialized successfully with collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}", exc_info=DEBUG)
            self.vector_store = None
            self.client = None
            self.is_initialized = False
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}") from e

    def add_documents(self, documents: List[Document]) -> None:
        """Adds new documents (chunks) to the vector store."""
        if not self.vector_store:
            logger.error("Cannot add documents: vector store object is not available.")
            raise VectorStoreError("Vector store object is not available (was None when add_documents called).")

        if not documents:
            logger.warning("No documents provided to add.")
            return

        valid_docs = [doc for doc in documents if isinstance(doc, Document) and doc.page_content]
        if len(valid_docs) != len(documents):
            logger.warning(f"Filtered out {len(documents) - len(valid_docs)} invalid or empty documents.")

        if not valid_docs:
            logger.warning("No valid documents remaining to add.")
            return

        logger.info(f"Adding {len(valid_docs)} document chunks to vector store...")
        try:
            # Filter complex metadata - handle each metadata dict directly
            for doc in valid_docs:
                if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
                    filtered_metadata = {}
                    for key, value in doc.metadata.items():
                        # Only keep simple types
                        if isinstance(value, (str, int, float, bool)):
                            filtered_metadata[key] = value
                        elif isinstance(value, list) and len(value) > 0:
                            # Convert lists to string representation
                            filtered_metadata[key] = str(value[0])
                    doc.metadata = filtered_metadata

            # Generate unique IDs based on source and chunk ID
            ids = []
            for i, doc in enumerate(valid_docs):
                source = str(doc.metadata.get('source', '')).replace('\\', '_').replace('/', '_')
                chunk_id = str(doc.metadata.get('chunk_id', i))
                unique_id = f"doc_{i}_{source}_{chunk_id}"[:256]  # Limit ID length
                ids.append(unique_id)

            self.vector_store.add_documents(valid_docs, ids=ids)
            logger.info(f"Successfully added {len(valid_docs)} chunks to vector store.")
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}", exc_info=DEBUG)
            raise VectorStoreError("Failed to add documents to the vector store.") from e

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """
        Performs similarity search on the vector store.
        """
        if not self.is_initialized or not self.vector_store:
            logger.error("Cannot perform search: vector store not initialized.")
            raise VectorStoreError("Vector store is not initialized.")

        if not query or not query.strip():
            logger.warning("Empty query provided for similarity search.")
            return []

        logger.debug(f"Performing similarity search for query: '{query}', k={k}, filter={filter}")
        try:
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter
            )
            logger.debug(f"Found {len(results)} results for query: '{query}'")
            return results
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=DEBUG)
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Gets statistics about the vector store collection."""
        if not self.is_initialized or not self.vector_store:
            return {"status": "not_initialized"}

        try:
            doc_count = self.vector_store._collection.count()
            return {
                "status": "initialized",
                "store_type": "Chroma",
                "collection_name": self.collection_name,
                "document_count": doc_count,
                "embedding_model": EMBEDDING_MODEL,
                "persist_directory": self.persist_directory,
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {e}", exc_info=DEBUG)
            return {"status": "error", "message": str(e)}

    def reset(self) -> None:
        """Deletes the vector store from disk and resets the manager state."""
        logger.warning(f"Resetting vector store: Deleting directory {self.persist_directory}")
        self.vector_store = None
        self.is_initialized = False
        try:
            if VECTOR_STORE_DIR.exists():
                shutil.rmtree(VECTOR_STORE_DIR)
            VECTOR_STORE_DIR.mkdir(exist_ok=True, parents=True)
            logger.info("Vector store directory deleted and recreated.")
        except OSError as e:
            logger.error(f"Error deleting vector store directory: {e}", exc_info=DEBUG)
            raise VectorStoreError("Failed to delete vector store directory during reset.") from e
        except Exception as e:
             logger.error(f"Error resetting vector store: {e}", exc_info=DEBUG)
             raise VectorStoreError("Failed to reset vector store.") from e

    def persist(self):
        """Persists the vector store to disk (No longer needed for Chroma >= 0.4)."""
        logger.warning("Explicit persist() called, but ChromaDB >= 0.4 persists automatically. This call is deprecated.")
        return True