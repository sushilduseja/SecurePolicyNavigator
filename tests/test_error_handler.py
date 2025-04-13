import pytest
from src.utils.error_handler import (
    handle_model_error,
    ModelError,
    DocumentLoadError,
    VectorStoreError
)

def test_model_error():
    """Test ModelError exception."""
    with pytest.raises(ModelError) as exc_info:
        raise ModelError("Test model error")
    assert str(exc_info.value) == "Test model error"

def test_document_load_error():
    """Test DocumentLoadError exception."""
    with pytest.raises(DocumentLoadError) as exc_info:
        raise DocumentLoadError("Test document load error")
    assert str(exc_info.value) == "Test document load error"

def test_vector_store_error():
    """Test VectorStoreError exception."""
    with pytest.raises(VectorStoreError) as exc_info:
        raise VectorStoreError("Test vector store error")
    assert str(exc_info.value) == "Test vector store error"

def test_handle_model_error():
    """Test model error handling function."""
    class TestError(Exception):
        pass
    
    def failing_function():
        raise TestError("Internal model error")
    
    try:
        handle_model_error(failing_function)
    except TestError as error:
        assert isinstance(error, TestError)
        assert "internal model error" in str(error).lower()
