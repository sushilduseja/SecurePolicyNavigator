import logging
from typing import Union

logger = logging.getLogger(__name__)

# --- Custom Exception Classes ---

class BaseNavigatorError(Exception):
    """Base exception for all application-specific errors."""
    pass

class ModelError(BaseNavigatorError):
    """Exception for LLM or embedding model related errors."""
    pass

class PipelineError(BaseNavigatorError):
    """Exception for errors within the RAG pipeline logic."""
    pass

class DocumentLoadError(BaseNavigatorError):
    """Exception for errors during document loading or processing."""
    pass

class VectorStoreError(BaseNavigatorError):
    """Exception for errors related to vector store operations."""
    pass

# --- Error Handling Function ---

def handle_model_error(error: Exception) -> str:
    """
    Provides a user-friendly message for common model/pipeline errors.
    Logs the original error.
    """
    error_str = str(error)
    logger.error(f"Handling error: {error_str}", exc_info=True) # Log full traceback

    # Check for specific known error patterns (use sparingly, prefer exception types)
    if "LayerNormKernelImpl" in error_str or "CUDA" in error_str.upper():
        return "Model computation error (possibly CUDA/GPU related). Check logs for details."
    elif "out of memory" in error_str.lower():
        return "Model execution failed (Out of Memory). Try reducing context documents or using a smaller model."
    elif "ConnectionError" in error_str:
         return "Could not connect to model service (check network and model endpoint)."
    elif isinstance(error, ModelError):
        return f"Model Error: {error_str}"
    elif isinstance(error, PipelineError):
        return f"Pipeline Error: {error_str}"
    elif isinstance(error, VectorStoreError):
        return f"Vector Store Error: {error_str}"
    elif isinstance(error, DocumentLoadError):
        return f"Document Loading Error: {error_str}"
    else:
        # Generic fallback
        return f"An unexpected error occurred: {error_str}"