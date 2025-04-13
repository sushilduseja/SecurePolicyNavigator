import pytest
import os
from pathlib import Path
from src.utils.document_loader import get_document_list, load_documents
from src.utils.error_handler import DocumentLoadError

@pytest.fixture
def test_data_dir(tmp_path):
    """Create a temporary directory with test documents."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create sample policy documents with realistic content
    docs = {
        "gdpr.md": """# GDPR Data Protection Policy
        
        ## Article 1: Data Subject Rights
        1. Right to Access: Data subjects have the right to obtain confirmation of whether their personal data is being processed.
        2. Right to Rectification: Data subjects can request the correction of inaccurate personal data.
        3. Right to Erasure: Also known as 'right to be forgotten'.
        4. Right to Data Portability: Data subjects can request their data in a structured format.
        
        ## Article 2: Data Processing
        1. Lawfulness of Processing
        2. Consent Requirements
        3. Purpose Limitation
        
        ## Article 3: Data Protection Measures
        1. Encryption Requirements
        2. Access Controls
        3. Audit Logging""",
        
        "iso27001.md": """# ISO 27001 Security Policy
        
        ## Section 1: Access Control Policy
        1. Password Requirements
           - Minimum 12 characters
           - Must include numbers, symbols, and mixed case
           - Changed every 90 days
        2. Multi-Factor Authentication
        3. Session Management
        
        ## Section 2: Asset Management
        1. Asset Inventory
        2. Asset Classification
        3. Asset Handling
        
        ## Section 3: Incident Response
        1. Incident Classification
        2. Response Procedures
        3. Recovery Plans""",
        
        "nist_csf.md": """# NIST Cybersecurity Framework
        
        ## Function 1: Identify
        1. Asset Management
        2. Business Environment
        3. Risk Assessment
        
        ## Function 2: Protect
        1. Access Control
        2. Data Security
        3. Protective Technology
        
        ## Function 3: Detect
        1. Anomalies and Events
        2. Security Monitoring
        3. Detection Processes""",
        
        "test.txt": "This is a test file that should be ignored",
        "test.pdf": "PDF content that should be ignored",
        "test.doc": "DOC content that should be ignored"
    }
    
    for filename, content in docs.items():
        (data_dir / filename).write_text(content)
    
    return data_dir

def test_get_document_list(tmp_path):
    """Test getting list of documents."""
    # Create test files
    (tmp_path / "test1.md").write_text("Test 1")
    (tmp_path / "test2.txt").write_text("Test 2")
    (tmp_path / "test3.md").write_text("Test 3")
    (tmp_path / "test4.txt").write_text("Test 4")
    (tmp_path / "test5.pdf").write_text("Test 5")  # Unsupported format
    
    # Test with default file types
    docs = get_document_list(str(tmp_path))
    assert len(docs) == 4  # Should find 2 .md and 2 .txt files
    
    # Test with specific file type
    docs = get_document_list(str(tmp_path), file_types=['md'])
    assert len(docs) == 2  # Should only find .md files

def test_load_documents(tmp_path):
    """Test loading documents."""
    # Create test files
    (tmp_path / "test1.md").write_text("# Test 1\nContent 1")
    (tmp_path / "test2.txt").write_text("Test 2\nContent 2")
    
    # Load documents
    docs = load_documents(str(tmp_path))
    assert len(docs) == 2
    
    # Verify document contents and metadata
    for doc in docs:
        assert doc.page_content
        assert "source" in doc.metadata
        assert "file_type" in doc.metadata
        if doc.metadata["source"].endswith(".md"):
            assert doc.metadata["file_type"] == "markdown"
        else:
            assert doc.metadata["file_type"] == "text"

def test_load_documents_unsupported_format(tmp_path):
    """Test loading documents with unsupported format."""
    # Create test file with unsupported extension
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test.xyz").write_text("Test content")

    # Attempt to load documents
    with pytest.raises(DocumentLoadError) as exc_info:
        load_documents(str(test_dir))
    assert str(exc_info.value) == "No documents found"

def test_load_documents_with_subdirectories(test_data_dir):
    """Test loading documents from nested directory structure."""
    # Create subdirectories with documents
    subdir = test_data_dir / "policies" / "security"
    subdir.mkdir(parents=True)
    
    (subdir / "access_control.md").write_text("""
    # Access Control Policy
    1. Authentication Requirements
    2. Authorization Levels
    3. Access Review Process
    """)
    
    docs = load_documents(str(test_data_dir))
    # Expect 3 original .md + 1 new .md + 1 .txt = 5
    assert len(docs) == 5
    
    # Verify nested document was loaded
    doc_contents = [doc.page_content for doc in docs]
    assert any("Access Control Policy" in content for content in doc_contents)

def test_load_documents_with_special_characters(test_data_dir):
    """Test loading documents with special characters in content and filenames."""
    special_filename = test_data_dir / "special-chars-$#@!.md"
    special_content = """# Special Characters Test
    Line with symbols: !@#$%^&*()
    Line with quotes: "single' quotes"
    Line with unicode: 🔒 💻 🔑
    """
    special_filename.write_text(special_content, encoding='utf-8')
    
    docs = load_documents(str(test_data_dir))
    # Expect 4 original .md + 1 special .md = 5
    assert len(docs) == 5
    
    # Verify special content was loaded
    doc_contents = [doc.page_content for doc in docs]
    assert any("Special Characters Test" in content for content in doc_contents)
    assert any("symbols: !@#$%^&*()" in content for content in doc_contents)

def test_load_documents_with_large_files(test_data_dir):
    """Test loading large documents."""
    large_file = test_data_dir / "large_policy.md"
    # Create a 1MB file
    with open(large_file, 'w') as f:
        f.write("# Large Policy Document\n")
        f.write("Content line\n" * 50000)
    
    docs = load_documents(str(test_data_dir))
    # Expect 3 original .md + 1 large .md + 1 .txt = 5
    assert len(docs) == 5
    
    # Verify large file was loaded
    doc_contents = [doc.page_content for doc in docs]
    assert any("Large Policy Document" in content for content in doc_contents)

def test_load_documents_with_invalid_encoding(test_data_dir):
    """Test handling of files with invalid encoding."""
    invalid_file = test_data_dir / "invalid_encoding.md"
    # Create a file with invalid UTF-8 bytes
    with open(invalid_file, 'wb') as f:
        f.write(b"# Invalid UTF-8 Content\n")
        f.write(b"\xFF\xFE Invalid bytes")
    
    # Should skip invalid file but load others
    docs = load_documents(str(test_data_dir))
    # Expect 3 original .md + 1 .txt = 4 (invalid file skipped)
    assert len(docs) == 4
    
def test_load_documents_with_empty_files(test_data_dir):
    """Test handling of empty files."""
    empty_file = test_data_dir / "empty.md"
    empty_file.write_text("")
    
    docs = load_documents(str(test_data_dir))
    # Expect 3 original .md + 1 empty .md + 1 .txt = 5
    assert len(docs) == 5
    
def test_load_documents_with_symlinks(test_data_dir):
    """Test handling of symbolic links."""
    if os.name != 'nt':  # Skip on Windows
        symlink_dir = test_data_dir / "symlink_dir"
        os.symlink(test_data_dir, symlink_dir)
        
        docs = load_documents(str(symlink_dir))
        assert len(docs) == 3  # Should load through symlink

def test_load_documents_concurrent_access(test_data_dir):
    """Test concurrent document loading."""
    import threading
    
    results = []
    def load_thread():
        try:
            docs = load_documents(str(test_data_dir))
            results.append(len(docs))
        except Exception as e:
            results.append(e)
    
    threads = [threading.Thread(target=load_thread) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # All threads should successfully load 4 documents
    assert all(r == 4 for r in results)

def test_load_documents_empty_dir(tmp_path):
    """Test loading documents from empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    with pytest.raises(DocumentLoadError) as exc_info:
        load_documents(str(empty_dir))
    assert "No documents found" in str(exc_info.value)

def test_load_documents_invalid_path():
    """Test loading documents from invalid path."""
    with pytest.raises(DocumentLoadError) as exc_info:
        load_documents("invalid/path")
    assert "Directory does not exist" in str(exc_info.value)

def test_load_documents_unsupported_format(tmp_path):
    """Test loading documents with unsupported format."""
    test_dir = tmp_path / "test"
    test_dir.mkdir()
    (test_dir / "test.xyz").write_text("Test content")
    
    docs = load_documents(str(test_dir))
    assert len(docs) == 0  # Should not load unsupported formats
