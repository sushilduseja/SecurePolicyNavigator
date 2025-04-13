"""RAG Pipeline implementation for the Policy Navigator."""
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import torch

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    pipeline,
    GenerationConfig
)
from langchain.chains import RetrievalQA
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.prompts import PromptTemplate

from src.utils.config import (
    LLM_MODEL,
    LLM_CONFIG,
    RAG_CHAIN_TYPE,
    DEFAULT_K,
    DEBUG
)
from src.utils.vector_store import VectorStoreManager
from src.utils.error_handler import handle_model_error, ModelError

logger = logging.getLogger(__name__)

# Template for improved response structuring
RESPONSE_TEMPLATE = """Answer the question based on the provided context.
Focus on relevant security policy and compliance information.
If the answer cannot be derived from the context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer: Let me help you with that.
"""

class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline for policy queries.
    """
    
    def __init__(self):
        """Initialize RAG pipeline components."""
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.retriever = None
        self._initialize_pipeline()

    def _initialize_pipeline(self):
        """Initialize LLM and QA chain components."""
        try:
            # Initialize vector store first
            if not self.vector_store_manager.is_initialized:
                self.vector_store_manager.initialize()
            self.vector_store = self.vector_store_manager.vector_store

            # Initialize LLM components with CPU-only mode and explicit FP32
            tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
            
            # Force FP32 and CPU, disable all mixed precision
            torch.set_default_dtype(torch.float32)  # Set default dtype to float32
            model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL,
                torch_dtype=torch.float32,  # Force FP32
                device_map="cpu",  # Force CPU
                trust_remote_code=True,
                load_in_8bit=False,  # Disable quantization
                load_in_4bit=False,  # Disable quantization
                use_cache=True
            )

            # Ensure model and all parameters are in eval mode and FP32
            model = model.eval()
            model = model.float()  # Convert all parameters to float32
            
            # Verify all parameters are float32
            for param in model.parameters():
                param.data = param.data.float()

            # Configure generation settings
            generation_config = GenerationConfig(
                do_sample=True,
                temperature=LLM_CONFIG["temperature"],
                top_p=LLM_CONFIG["top_p"],
                top_k=LLM_CONFIG["top_k"],
                repetition_penalty=LLM_CONFIG["repetition_penalty"],
                max_new_tokens=LLM_CONFIG["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else LLM_CONFIG["pad_token_id"]
            )

            # Create text generation pipeline with explicit FP32 settings
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                max_new_tokens=LLM_CONFIG["max_new_tokens"],
                batch_size=1,  # Process one at a time for lower memory usage
                device_map="cpu",  # Force CPU
                torch_dtype=torch.float32,  # Force FP32
                framework="pt",  # Explicitly set framework to PyTorch
            )

            # Create LangChain HF pipeline
            self.llm = HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={"temperature": LLM_CONFIG["temperature"]}
            )

            # Create improved prompt template with better instruction following
            prompt = PromptTemplate(
                template="""Given the following context from security and compliance documents, answer the question accurately and concisely. If the answer cannot be directly derived from the provided context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer: Let me help you with that based on the provided context.""",
                input_variables=["context", "question"]
            )

            # Initialize retriever with consistent settings
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": DEFAULT_K}
            )

            # Create QA chain with optimized settings
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type=RAG_CHAIN_TYPE,
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={
                    "prompt": prompt,
                    "document_variable_name": "context"
                }
            )

            logger.info("RAG Pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}", exc_info=DEBUG)
            raise ModelError(f"RAG Pipeline initialization failed: {str(e)}") from e

    def process_query(self, query: str, k: Optional[int] = None, 
                     filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a natural language query using the RAG pipeline.
        
        Args:
            query: The user's question
            k: Number of documents to retrieve (optional)
            filters: Metadata filters for document retrieval (optional)
            
        Returns:
            Dict containing answer and source documents
        """
        if not query or not query.strip():
            return {"answer": "Please provide a valid question.", "sources": []}

        try:
            logger.info(f"Processing query: '{query}', k={k}, filters={filters}")
            
            # Update retriever parameters if specified
            if k is not None:
                self.retriever.search_kwargs["k"] = k
            if filters:
                self.retriever.search_kwargs["filter"] = filters

            # Get relevant documents
            documents = self.retriever.invoke(query)
            logger.info(f"Retrieved {len(documents)} documents for query '{query}'.")

            # Process query with QA chain
            result = self.qa_chain.invoke({"query": query})
            
            # Extract answer and sources
            answer = result.get("result", "No answer generated.")
            source_documents = result.get("source_documents", [])
            
            # Format sources for response
            sources = []
            for doc in source_documents:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source = doc.metadata['source']
                    if isinstance(source, Path):
                        source = str(source)
                    sources.append(source)

            return {
                "answer": answer,
                "sources": list(set(sources))  # Deduplicate sources
            }

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}", exc_info=DEBUG)
            error_message = handle_model_error(e)
            return {
                "answer": f"Error: {error_message}",
                "sources": []
            }

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant document chunks for a query."""
        if not self.vector_store:
            return ""
            
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception as e:
            logger.error(f"Error retrieving context: {e}", exc_info=DEBUG)
            return ""

    def reset(self):
        """Reset the pipeline state."""
        self.vector_store_manager.reset()
        self.vector_store = None
        self._initialize_pipeline()