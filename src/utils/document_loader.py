import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain.schema import Document
from src.utils.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.error_handler import DocumentLoadError  # Import DocumentLoadError

class SimpleMarkdownLoader:
    """A simple markdown loader that reads files as plain text with basic markdown handling."""
    
    def __init__(self, file_path):
        self.file_path = file_path
    
    def _clean_markdown(self, text: str) -> str:
        """Remove markdown formatting while preserving content structure."""
        # Remove headers while preserving content
        text = re.sub(r'#{1,6}\s+(.+)', r'\1', text)
        
        # Remove emphasis markers
        text = re.sub(r'[*_]{1,2}([^*_]+)[*_]{1,2}', r'\1', text)
        
        # Remove code blocks while preserving content
        text = re.sub(r'```[^\n]*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
        
        # Remove inline code
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Convert lists to plain text
        text = re.sub(r'^\s*[-*+]\s+', '• ', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '• ', text, flags=re.MULTILINE)
        
        # Preserve line breaks for readability
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def load(self) -> List[Document]:
        """Load and process markdown file."""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            # Clean markdown formatting
            cleaned_text = self._clean_markdown(text)
            
            # Create metadata
            metadata = {
                "source": str(self.file_path),
                "file_type": "markdown",
                "filename": Path(self.file_path).name
            }
            
            return [Document(page_content=cleaned_text, metadata=metadata)]
            
        except Exception as e:
            print(f"Error loading markdown file {self.file_path}: {str(e)}")
            return []

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:-]', '', text)
    return text.strip()

def process_document(doc: Document) -> Document:
    """Process a single document by cleaning text and adding metadata."""
    # Clean the text content while preserving important characters
    cleaned_text = doc.page_content.strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    # Ensure metadata contains required fields
    metadata = doc.metadata.copy()
    if "source" in metadata:
        metadata["source"] = str(metadata["source"])
    
    # Add computed metadata
    metadata["word_count"] = len(cleaned_text.split())
    metadata["char_count"] = len(cleaned_text)
    
    return Document(page_content=cleaned_text, metadata=metadata)

def process_documents(documents: List[Document]) -> List[Document]:
    """Process a list of documents by cleaning text and adding metadata."""
    return [process_document(doc) for doc in documents]

def load_documents(data_dir: Optional[str] = None) -> List[Document]:
    """Load and process documents from directory."""
    if data_dir is None:
        data_dir = str(DATA_DIR)
    
    if not os.path.exists(data_dir):
        raise DocumentLoadError("Directory does not exist")
        
    print(f"Loading documents from {data_dir}")
    documents = []
    data_path = Path(data_dir)
    
    # Load supported file types
    supported_extensions = ["*.md", "*.txt"]  # Support both markdown and text files
    supported_files = []
    for ext in supported_extensions:
        supported_files.extend(list(data_path.glob(f"**/{ext}")))
    
    if not supported_files:
        # Check if there are any files at all (including unsupported types)
        all_files = list(data_path.glob("**/*.*"))
        if all_files:
            print(f"Found {len(all_files)} files, but none with supported extensions ({', '.join(supported_extensions)})")
            # Just return empty list rather than raising an error
            return []
        else:
            raise DocumentLoadError("No documents found")
    
    for file_path in supported_files:
        try:
            if file_path.suffix == '.md':
                loader = SimpleMarkdownLoader(str(file_path))
                file_type = "markdown"  # Changed from 'md' to 'markdown'
            else:  # .txt files
                loader = TextLoader(str(file_path), encoding='utf-8')
                file_type = "text"
                
            docs = loader.load()
            
            # Normalize paths and metadata
            for doc in docs:
                doc.metadata.update({
                    "source": str(file_path.resolve()),
                    "file_type": file_type,
                    "filename": file_path.name
                })
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    
    processed_docs = process_documents(documents)
    print(f"Loaded {len(processed_docs)} documents")
    return processed_docs

def get_document_list(
    data_dir: Optional[str] = None,
    file_types: Optional[List[str]] = None
) -> List[str]:
    """
    Get a list of available document names in the specified directory.
    
    Args:
        data_dir: The directory to search in. Defaults to DATA_DIR from config.
        file_types: List of file extensions to include (without dots).
                   If None, includes all supported file types.
    
    Returns:
        List of document filenames
    """
    if data_dir is None:
        data_dir_path = DATA_DIR
    else:
        data_dir_path = Path(data_dir)

    if not data_dir_path.is_dir():
        return []

    if file_types is None:
        file_types = ['md', 'txt']  # Default supported types
    
    # Convert file types to proper glob patterns
    patterns = [f"**/*.{ext}" for ext in file_types]
    
    documents = []
    for pattern in patterns:
        documents.extend([f.name for f in data_dir_path.glob(pattern)])
    
    return sorted(list(set(documents)))