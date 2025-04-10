from typing import List, Dict, Any
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from ..utils.config import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    VECTOR_STORE_DIR,
    COLLECTION_NAME
)
from ..utils.document_loader import load_documents

class RAGPipeline:
    def __init__(self):
        """Initialize the RAG pipeline components."""
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.llm = Ollama(model=LLM_MODEL)
        self.vector_store = None
        self.qa_chain = None
        
    def initialize(self):
        """Initialize the vector store and QA chain."""
        # Load and process documents
        documents = load_documents()
        
        # Create or load vector store
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(VECTOR_STORE_DIR),
            collection_name=COLLECTION_NAME
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 3}
            )
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline with a question.
        Returns the answer and relevant source documents.
        """
        if not self.qa_chain:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        # Get answer from QA chain
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": []  # TODO: Implement source tracking
        } 