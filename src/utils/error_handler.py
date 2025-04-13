from typing import Union, Any, Callable
import logging

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Base exception for model-related errors."""
    pass

class ModelOperationError(ModelError):
    """Exception for model operation errors."""
    pass

class DocumentLoadError(Exception):
    """Exception for document loading errors."""
    pass

class VectorStoreError(Exception):
    """Exception for vector store operations."""
    pass

def handle_model_error(error: Union[Exception, Callable]) -> Union[str, Exception]:
    """Handle common model operation errors."""
    # If error is a callable, execute it to get the actual error
    error_obj = error() if callable(error) else error
    error_str = str(error_obj)
    
    if "LayerNormKernelImpl" in error_str:
        error_msg = ("Model precision error: Switching to float32 precision.")
        logger.error(error_msg)
        return ModelOperationError(error_msg)
    
    # Generic model errors
    if "model" in error_str.lower():
        return ModelError(f"Model error: {error_str}")
        
    return error_obj
