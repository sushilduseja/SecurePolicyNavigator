from pathlib import Path
import tempfile
import os

# Test configuration - Use temporary directory for tests
TEST_DIR = Path(tempfile.mkdtemp()) / "secure_policy_navigator_test"
TEST_VECTOR_STORE_DIR = TEST_DIR / ".vector_store"
TEST_DATA_DIR = TEST_DIR / "data"

# Test environment settings
TEST_SETTINGS = {
    "EMBEDDING_MODEL": "mock_embeddings",  # Use mock instead of real model
    "VECTOR_STORE_DIR": str(TEST_VECTOR_STORE_DIR),
    "COLLECTION_NAME": "test_collection",
    "CHUNK_SIZE": 100,  # Reduced for testing
    "CHUNK_OVERLAP": 20  # Reduced for testing
}

# Create test directories
for dir in [TEST_DIR, TEST_VECTOR_STORE_DIR, TEST_DATA_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# Override default vector store directory for tests
import sys
from unittest.mock import patch
import builtins

# Patch VECTOR_STORE_DIR in config
patch.dict('sys.modules').start()
