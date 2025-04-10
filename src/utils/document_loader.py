from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP

def load_documents() -> List[Dict[str, Any]]:
    """
    Load documents from the data directory and split them into chunks.
    Returns a list of document chunks with metadata.
    """
    # Load all text files from the data directory
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    return chunks

def get_document_list() -> List[str]:
    """
    Get a list of available document names in the data directory.
    """
    return [f.name for f in DATA_DIR.glob("*.txt")] 