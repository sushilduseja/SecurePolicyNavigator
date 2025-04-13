import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging

# Use Unstructured loaders for better handling of complex markdown and other formats
# Ensure 'unstructured' and potentially related libraries (like 'markdown', 'libmagic') are installed
try:
    from langchain_community.document_loaders import (
        DirectoryLoader,
        UnstructuredMarkdownLoader,
        TextLoader,
    )
    HAS_UNSTRUCTURED = True
except ImportError:
    HAS_UNSTRUCTURED = False
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    logging.warning("Unstructured library not found. Markdown parsing will be basic.")
    logging.warning("Install 'unstructured' (and potentially 'unstructured[md]') for better markdown support.")

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from src.utils.config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils.error_handler import DocumentLoadError

logger = logging.getLogger(__name__)

def _create_loader(file_path: Path) -> Any:
    """Creates the appropriate document loader based on file extension."""
    ext = file_path.suffix.lower()
    if ext == ".md" and HAS_UNSTRUCTURED:
        # Unstructured handles complex markdown better
        return UnstructuredMarkdownLoader(str(file_path), mode="elements")
    elif ext in [".txt", ".md"]: # Fallback for .md if unstructured is not available
        # Use TextLoader for plain text and basic markdown fallback
        try:
            return TextLoader(str(file_path), encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to load {file_path} with utf-8 encoding, trying fallback: {e}")
            # Try a fallback encoding if utf-8 fails
            try:
                return TextLoader(str(file_path), encoding="latin-1")
            except Exception as e_fallback:
                logger.error(f"Failed to load {file_path} with fallback encoding: {e_fallback}")
                raise DocumentLoadError(f"Could not load file {file_path.name} due to encoding issues.") from e_fallback
    else:
        # Potentially add loaders for other types (PDF, DOCX) using unstructured here
        logger.warning(f"Unsupported file type: {ext} for file {file_path.name}. Skipping.")
        return None


def load_and_split_documents(data_dir: Optional[str] = None) -> List[Document]:
    """
    Loads documents from the specified directory, splits them into chunks,
    and returns a list of Document objects (chunks).
    """
    target_dir = Path(data_dir) if data_dir else DATA_DIR
    if not target_dir.is_dir():
        logger.error(f"Data directory not found: {target_dir}")
        raise DocumentLoadError(f"Directory does not exist: {target_dir}")

    logger.info(f"Loading documents from: {target_dir}")
    loaded_documents = []
    processed_files = 0
    skipped_files = 0

    supported_extensions = ["*.md", "*.txt"]
    all_files = []
    for ext_pattern in supported_extensions:
        all_files.extend(list(target_dir.rglob(ext_pattern)))

    if not all_files:
        logger.warning(f"No documents with supported extensions ({', '.join(supported_extensions)}) found in {target_dir}")
        return []

    logger.info(f"Found {len(all_files)} potential documents.")

    for file_path in all_files:
        if file_path.is_file():
            # ****** ADDED LOGGING ******
            logger.debug(f"Processing file: {file_path.name}")
            # **************************
            try:
                loader = _create_loader(file_path)
                if loader:
                    # ****** ADDED LOGGING ******
                    logger.info(f"Loading '{file_path.name}' using {type(loader).__name__}")
                    # **************************
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = str(file_path.resolve())
                        doc.metadata["filename"] = file_path.name
                    loaded_documents.extend(docs)
                    processed_files += 1
                else:
                    skipped_files += 1
            except DocumentLoadError as dle:
                logger.error(f"Skipping file due to load error: {dle}")
                skipped_files += 1
            except Exception as e:
                logger.error(f"Unexpected error loading file {file_path}: {e}", exc_info=DEBUG)
                skipped_files += 1
        else:
            logger.warning(f"Skipping non-file item: {file_path}")
            skipped_files += 1

    if not loaded_documents:
        logger.warning("No documents were successfully loaded.")
        return []

    logger.info(f"Successfully loaded content from {processed_files} files. Skipped {skipped_files} files.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    logger.info(f"Splitting {len(loaded_documents)} loaded document sections into chunks...")
    chunks = text_splitter.split_documents(loaded_documents)
    logger.info(f"Created {len(chunks)} document chunks.")

    # ****** ADDED LOGGING ******
    # Log chunk counts per file for debugging retrieval issues
    source_chunk_counts = {}
    for chunk in chunks:
        source = chunk.metadata.get("filename", "Unknown")
        source_chunk_counts[source] = source_chunk_counts.get(source, 0) + 1
    logger.debug(f"Chunk counts per source file: {source_chunk_counts}")
    # **************************


    final_chunks = []
    for i, chunk in enumerate(chunks):
        chunk.page_content = re.sub(r"\s+", " ", chunk.page_content).strip()
        chunk.metadata["chunk_id"] = i
        if chunk.page_content:
            final_chunks.append(chunk)

    logger.info(f"Returning {len(final_chunks)} non-empty document chunks.")
    return final_chunks

def get_document_list(data_dir: Optional[str] = None) -> List[str]:
    """
    Get a list of unique source document filenames available in the data directory.
    """
    target_dir = Path(data_dir) if data_dir else DATA_DIR
    if not target_dir.is_dir():
        logger.warning(f"Cannot list documents, directory not found: {target_dir}")
        return []

    supported_extensions = ['md', 'txt'] # Add more if loaders are added
    patterns = [f"**/*.{ext}" for ext in supported_extensions]

    documents = set()
    for pattern in patterns:
        for f in target_dir.rglob(pattern):
            if f.is_file():
                # Store the relative path from the data directory for consistency
                try:
                    relative_path = f.relative_to(target_dir)
                    documents.add(str(relative_path))
                except ValueError:
                    # If the file is not within the target_dir somehow, use absolute path
                    documents.add(str(f.resolve()))


    logger.info(f"Found {len(documents)} unique source documents in {target_dir}")
    return sorted(list(documents))