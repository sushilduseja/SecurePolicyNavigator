import pytest
import os
import tempfile
from pathlib import Path
import shutil
from unittest.mock import MagicMock, patch
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Create test directories
TEST_DIR = Path(tempfile.gettempdir()) / "secure_policy_navigator_test"
TEST_VECTOR_STORE_DIR = TEST_DIR / ".chroma"
TEST_DATA_DIR = TEST_DIR / "data"

# Test environment settings
TEST_SETTINGS = {
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "VECTOR_STORE_DIR": str(TEST_VECTOR_STORE_DIR),
    "COLLECTION_NAME": "test_collection",
    "CHUNK_SIZE": 500,
    "CHUNK_OVERLAP": 50
}

class MockSentenceTransformer:
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return [0.1] * 384
        return [[0.1] * 384 for _ in texts]

# Add mock before importing any project modules
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['sentence_transformers'].SentenceTransformer = MockSentenceTransformer

# Mock langchain_community.embeddings.HuggingFaceEmbeddings
sys.modules['langchain_community.embeddings'] = MagicMock()
sys.modules['langchain_community.embeddings.HuggingFaceEmbeddings'] = MagicMock()

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def sample_documents(test_data_dir):
    """Create sample Document objects for testing."""
    from langchain_core.documents import Document
    
    # Create and return Document objects directly
    return [
        Document(
            page_content="Access Control Policy\nPassword policy requires minimum 12 characters\nMulti-Factor Authentication required for all systems",
            metadata={"source": "access_control.md", "file_type": "markdown", "section": "Access Control"}
        ),
        Document(
            page_content="GDPR Policy\nData Subject Rights include access, erasure, and portability\nData breaches must be reported within 72 hours",
            metadata={"source": "gdpr.md", "file_type": "markdown", "section": "Data Protection"}
        ),
        Document(
            page_content="Incident Response Plan\nSecurity incidents must be reported within 24 hours\nContainment procedures must be documented",
            metadata={"source": "incident_response.md", "file_type": "markdown", "section": "Incident Response"}
        ),
        Document(
            page_content="ISO 27001 Controls\nInformation security policies\nAccess control requirements\nCryptography standards",
            metadata={"source": "iso27001.md", "file_type": "markdown", "section": "Access Control"}
        ),
        Document(
            page_content="Security Policy 2025 Updates\nNew requirements for cloud security\nEnhanced monitoring requirements",
            metadata={"source": "security_policy_2025.md", "file_type": "markdown"}
        )
    ]

class MockRunnableLLM(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_instance = True
    
    def __instancecheck__(self, instance):
        return True

@pytest.fixture
def mock_llm_response():
    """Mock LLM response with realistic security policy answers."""
    mock = MockRunnableLLM()
    mock.return_value = {
        "answer": "Based on the security policy, passwords must be at least 12 characters long and include uppercase, lowercase, numbers, and symbols.",
        "sources": ["access_control.md"]
    }
    return mock

@pytest.fixture
def mock_load_documents(monkeypatch):
    """Mock the document loader to avoid file system operations in tests."""
    mock_loader = MagicMock()
    
    # Configure the mock to return sample documents when called
    def mock_load_function(path):
        return sample_documents()
    
    mock_loader.side_effect = mock_load_function
    
    # Patch the load_documents function
    monkeypatch.setattr('src.utils.document_loader.load_documents', mock_loader)
    
    return mock_loader

@pytest.fixture
def mock_embeddings():
    """Create deterministic mock embeddings."""
    class MockEmbeddings:
        def __init__(self):
            import numpy as np
            self.dimension = 384
        
        def embed_query(self, text):
            """Return a fixed-size embedding for any query."""
            import numpy as np
            return np.ones(self.dimension, dtype=np.float32) * 0.1
            
        def embed_documents(self, texts):
            """Return fixed-size embeddings for any list of texts."""
            import numpy as np
            return [np.ones(self.dimension, dtype=np.float32) * 0.1 for _ in range(len(texts))]
    
    return MockEmbeddings()

@pytest.fixture
def mock_vector_store(mock_embeddings):
    """Create a mock Chroma vector store."""
    # Create mock retriever
    mock_retriever = MagicMock()
    mock_retriever.get_relevant_documents = MagicMock(
        return_value=[MagicMock(page_content="Mock document from retriever")]
    )

    # Create mock docs with proper metadata for testing
    mock_docs = [
        MagicMock(
            page_content="Password policy requires minimum 12 characters",
            metadata={"source": "access_control.md", "file_type": "markdown"}
        ),
        MagicMock(
            page_content="GDPR data subject rights include access and erasure",
            metadata={"source": "gdpr.md", "file_type": "markdown"}
        ),
        MagicMock(
            page_content="ISO 27001 controls for information security",
            metadata={"source": "iso27001.md", "file_type": "markdown", "section": "Access Control"}
        )
    ]

    # Create mock store
    mock_store = MagicMock()
    mock_store.add_documents = MagicMock()
    
    # Configure similarity_search to handle filters
    def mock_similarity_search(query, k=3, filter=None):
        if not query:
            return []
            
        results = mock_docs
        
        # Apply filter if provided
        if filter:
            filtered_results = []
            for doc in results:
                matches = True
                for key, value in filter.items():
                    if doc.metadata.get(key) != value:
                        matches = False
                        break
                if matches:
                    filtered_results.append(doc)
            results = filtered_results
            
        # Limit to k results
        return results[:k]
    
    mock_store.similarity_search = MagicMock(side_effect=mock_similarity_search)
    mock_store.as_retriever = MagicMock(return_value=mock_retriever)
    mock_store.get = MagicMock(return_value=mock_docs)
    mock_store.persist = MagicMock()

    return mock_store

@pytest.fixture(scope="session", autouse=True)
def cleanup_chroma():
    """Cleanup Chroma vector store after each session."""
    yield
    chroma_dir = Path("./chroma")
    if chroma_dir.exists():
        for _ in range(3):  # Retry 3 times
            try:
                shutil.rmtree(chroma_dir)
                break
            except PermissionError:
                time.sleep(0.5)

@pytest.fixture(scope="session", autouse=True)
def test_env():
    """Set up test environment."""
    # Clean up any existing test directories
    if TEST_DIR.exists():
        try:
            shutil.rmtree(TEST_DIR)
        except PermissionError:
            pass  # Ignore permission errors on cleanup
    
    # Create fresh test directories
    TEST_DIR.mkdir(exist_ok=True)
    TEST_VECTOR_STORE_DIR.mkdir(exist_ok=True)
    TEST_DATA_DIR.mkdir(exist_ok=True)
    
    yield
    
    # Cleanup after tests
    try:
        if TEST_DIR.exists():
            shutil.rmtree(TEST_DIR)
    except PermissionError:
        pass  # Ignore permission errors on cleanup
