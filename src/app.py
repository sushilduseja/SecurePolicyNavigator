# ## FILE: src/app.py

import os
import sys
import traceback
import logging
import asyncio
import time
from pathlib import Path
import re # Ensure re is imported

# Attempt to set up environment early
try:
    # Assuming src.core.initialize handles essential setup
    from src.core import initialize
    initialize()
    print("Core initialization complete.") # Use print here as logging might not be fully set up yet
except ImportError:
    print("Warning: src.core could not be imported. Ensure it exists and is configured.")
except Exception as e:
    print(f"Error during core initialization: {e}")

# Configure logging (potentially re-configured by core.initialize)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import Streamlit and other libraries after potential setup
import streamlit as st
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from typing import List, Dict, Any

# Local imports (relative paths should work if src is in PYTHONPATH)
from src.utils.config import DEBUG, DATA_DIR, DEFAULT_K
from src.rag.pipeline import RAGPipeline
from src.utils.error_handler import handle_model_error, ModelError
from src.utils.document_loader import get_document_list

# Set page config (should be the first Streamlit command)
st.set_page_config(
    page_title="Secure Policy Navigator",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keep as is or enhance)
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #00acee; /* Twitter blue */
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
        border: none;
        border-radius: 0.5rem;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #008abe; /* Darker blue */
    }
    .source-box {
        background-color: #f0f2f6;
        border: 1px solid #dfe1e5;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .checklist-item { /* This class might not be used anymore with markdown checkbox */
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: #f8f9fa;
        border: 1px solid #eee;
        border-radius: 0.5rem;
    }
    /* Add styles for agraph if needed */
    </style>
    """, unsafe_allow_html=True)

# --- Initialization ---
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'checklist_items' not in st.session_state:
    st.session_state.checklist_items = {} # { requirement: bool_checked }
if 'collab_notes' not in st.session_state:
    st.session_state.collab_notes = ""
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = {"nodes": [], "edges": []}
if 'available_docs' not in st.session_state:
    st.session_state.available_docs = []
if 'active_docs' not in st.session_state:
     st.session_state.active_docs = [] # Store selected doc paths


def initialize_pipeline() -> None:
    """Initialize the RAG pipeline and load document list."""
    if st.session_state.initialized:
        return
    try:
        with st.spinner("🔧 Initializing Compliance Engine... Please wait."):
            st.session_state.rag_pipeline = RAGPipeline()
            # Load available documents after pipeline init (uses config)
            st.session_state.available_docs = get_document_list()
            # Default to all documents being active initially
            st.session_state.active_docs = st.session_state.available_docs
            st.session_state.initialized = True
        st.success("✅ Compliance Engine Ready!")
        # Short delay before potentially rerunning to update UI elements
        time.sleep(1)
        st.rerun() # Rerun to update UI elements that depend on initialization
    except ModelError as me:
        logger.error(f"Pipeline initialization failed: {me}", exc_info=DEBUG)
        st.error(f"❌ Initialization Failed: {me}. Please check model paths/availability and logs.")
        st.session_state.initialized = False
        st.session_state.rag_pipeline = None
    except Exception as e:
        logger.error(f"Unexpected error during initialization: {e}", exc_info=DEBUG)
        st.error(f"❌ Unexpected Initialization Error: {e}. Check logs for details.")
        st.session_state.initialized = False
        st.session_state.rag_pipeline = None

# --- UI Components ---

def display_sidebar():
    """Renders the sidebar controls."""
    with st.sidebar:
        st.header("🛠️ Controls")

        if not st.session_state.initialized:
            st.warning("Pipeline not initialized. Please wait or check logs.")
            # Optionally add a manual init button as fallback
            # if st.button("Retry Initialization"):
            #    initialize_pipeline()

        st.header("⚙️ Query Settings")
        # Similarity threshold isn't directly used by Chroma's default search
        # Keep slider for potential future use with threshold search or re-ranking
        # similarity_threshold = st.slider(
        #     "Similarity Score Threshold", 0.0, 1.0, 0.70, step=0.05,
        #     help="Minimum relevance score (if supported by search type)."
        # )
        max_results = st.slider(
            "Max Context Documents (k)", 1, 15, DEFAULT_K, # Increased max k slightly
            help="Maximum number of document chunks to use for context."
        )
        st.session_state.max_results = max_results # Store in session state

        st.header("📚 Document Scope")
        if st.session_state.initialized and st.session_state.available_docs:
            # Use available_docs for options, active_docs for default selection
            selected_docs = st.multiselect(
                "Active Documents",
                options=st.session_state.available_docs,
                default=st.session_state.active_docs, # Use the stored selection
                help="Select documents to include in the search scope."
            )
            # Update session state if selection changes
            if selected_docs != st.session_state.active_docs:
                st.session_state.active_docs = selected_docs
                st.info("Document scope updated.") # No need to rerun immediately
                # Rerunning here can interrupt user input, let the next query use the new scope

        elif st.session_state.initialized:
            st.info("No source documents found in the data directory.")

        # Add a button to reset the vector store (use with caution)
        st.divider()
        if st.button("⚠️ Reset Vector Store", help="Deletes all indexed data. Requires re-ingestion."):
            if st.session_state.rag_pipeline:
                try:
                    with st.spinner("Resetting vector store..."):
                        st.session_state.rag_pipeline.reset_vector_store()
                    st.success("Vector store reset. Run ingestion script to re-index.")
                    # Update available docs as they might be gone after reset?
                    st.session_state.available_docs = get_document_list()
                    st.session_state.active_docs = st.session_state.available_docs
                    # Clear checklist and chat history after reset
                    st.session_state.checklist_items = {}
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to reset vector store: {e}")
            else:
                st.error("Pipeline not initialized, cannot reset.")


def display_chat_interface():
    """Renders the main chat query interface."""
    st.subheader("💬 Policy Query Interface")

    # Display chat history
    for author, message in st.session_state.chat_history:
         with st.chat_message(author):
            # Use markdown for potentially better formatting in chat history
            st.markdown(message, unsafe_allow_html=False) # Keep unsafe_allow_html=False for security

    # Chat input
    user_question = st.chat_input("Ask about security policies...")

    if user_question:
        st.session_state.chat_history.append(("user", user_question))
        with st.chat_message("user"):
            st.markdown(user_question) # Display user question using markdown

        with st.chat_message("assistant"):
            message_placeholder = st.empty() # Placeholder for streaming or final answer
            message_placeholder.markdown("🔍 Analyzing policies...") # Initial message
            try:
                # Prepare filters based on selected documents
                doc_filters = None
                if st.session_state.active_docs:
                    doc_filters = {
                        "source": {
                            "$in": [str(Path(DATA_DIR) / doc_path) for doc_path in st.session_state.active_docs]
                        }
                    }

                # Get max_results from session state
                k_results = st.session_state.get('max_results', DEFAULT_K)

                response = st.session_state.rag_pipeline.process_query(
                    user_question,
                    k=k_results,
                    filters=doc_filters
                )
                answer = response.get("answer", "Sorry, I couldn't generate a response.")
                sources = response.get("sources", [])

                # Combine answer and sources for display
                full_response = answer
                if sources:
                    source_list = "\n\n**Sources:**\n" + "\n".join(f"- {Path(src).name}" for src in sources)
                    full_response = f"{answer}\n{source_list}"

                message_placeholder.markdown(full_response) # Update placeholder with final response
                st.session_state.chat_history.append(("assistant", full_response)) # Store combined response

            except Exception as e:
                logger.error(f"Error during query processing: {e}", exc_info=DEBUG)
                error_msg = f"An error occurred: {handle_model_error(e)}"
                message_placeholder.error(error_msg) # Show error in placeholder
                st.session_state.chat_history.append(("assistant", f"Error: {error_msg}"))


def display_policy_graph():
    """
    Visualizes policies as nodes. Dependency analysis is a placeholder.
    """
    st.subheader("📊 Policy Document Overview")
    st.info("""
        This graph shows the documents currently indexed in the vector store.
        **Note:** Automatic dependency analysis (showing links *between* policies based on content)
        is not yet implemented. This view helps visualize the available document set.
    """)

    if not st.session_state.initialized or not st.session_state.rag_pipeline:
        st.warning("Pipeline must be initialized to display document graph.")
        return

    nodes = []
    edges = [] # Explicitly empty until real relationship data exists

    try:
        # Attempt to get document metadata from the vector store for richer info
        if hasattr(st.session_state.rag_pipeline.vector_store_manager.vector_store, 'get'):
            all_docs_data = st.session_state.rag_pipeline.vector_store_manager.vector_store.get(include=['metadatas'])
            metadatas = all_docs_data.get('metadatas', [])
            unique_sources = {}
            if metadatas:
                for meta in metadatas:
                    source_path = meta.get('source')
                    filename = meta.get('filename', Path(source_path).name if source_path else 'Unknown')
                    if source_path and source_path not in unique_sources:
                         unique_sources[source_path] = {'filename': filename, 'source': source_path}
            logger.info(f"Found {len(unique_sources)} unique source documents in vector store for graph.")
            for source_path, meta_info in unique_sources.items():
                node_id = source_path
                node_label = meta_info['filename']
                nodes.append(Node(id=node_id, label=node_label, shape="box", title=f"Source: {source_path}"))
        else:
            logger.warning("Vector store does not support .get() or failed. Falling back to available_docs list.")
            doc_paths = st.session_state.available_docs
            for doc_path_str in doc_paths:
                doc_path = Path(doc_path_str)
                node_id = str(doc_path)
                node_label = doc_path.name
                nodes.append(Node(id=node_id, label=node_label, shape="box", title=f"Relative Path: {node_id}"))

        if not nodes:
            st.warning("No documents found in the vector store to display.")
            return

        # --- Placeholder for Relationship Extraction ---
        # TODO: Implement relationship extraction logic here.
        # --- End Placeholder ---

        with st.expander("Graph Display Options"):
            graph_height = st.slider("Graph Height", 400, 1000, 600)
            use_physics = st.toggle("Enable Physics Simulation", True)
            if use_physics:
                grav_const = st.slider("Gravitational Constant", -100000, 0, -30000, step=1000)
                spring_const = st.slider("Spring Constant", 0.0, 1.0, 0.05, step=0.01)
                spring_len = st.slider("Spring Length", 50, 500, 200)
                physics_config = {
                    'enabled': True,
                    'barnesHut': {
                        'gravitationalConstant': grav_const, 'centralGravity': 0.1,
                        'springLength': spring_len, 'springConstant': spring_const,
                        'damping': 0.09, 'avoidOverlap': 0.1
                        }
                    }
            else:
                 physics_config = {'enabled': False}

        config = Config(
            width="100%", height=graph_height, directed=True,
            nodeHighlightBehavior=True, highlightColor="#F7A7A6",
            collapsible=False,
            node={'labelProperty': 'label', 'size': 180, 'color': '#007bff', 'font': {'size': 14}},
            link={'highlightColor': '#8B008B', 'color': '#adb5bd'},
            physics=physics_config,
        )

        try:
            logger.debug(f"Rendering graph with {len(nodes)} nodes and {len(edges)} edges.")
            agraph(nodes=nodes, edges=edges, config=config)
        except Exception as e:
            logger.error(f"Error rendering graph with streamlit-agraph: {e}", exc_info=DEBUG)
            st.error(f"Could not render the policy graph visualization: {e}")

    except Exception as e:
        logger.error(f"Error preparing graph data: {e}", exc_info=DEBUG)
        st.error(f"Failed to prepare data for the policy graph: {e}")


# --- Constants and Helper for Checklist ---
CHECKLIST_GENERATION_PROMPT = """
Analyze the following context from security policy and compliance documents.
Identify and list the key actionable compliance requirements, obligations, or mandatory procedures mentioned.
Present the output *only* as a bulleted list. Each requirement must be on a new line, starting with '- '.
Do not include any introductory or concluding sentences, just the list.

Context:
{context}

Requirements List:
"""

def parse_llm_checklist_output(llm_output: str) -> Dict[str, bool]:
    """
    Parses the LLM's text output to extract checklist items.
    Assumes items start with common list markers on new lines.
    """
    items = {}
    lines = llm_output.strip().split('\n')
    logger.debug(f"Parsing checklist LLM output ({len(lines)} lines)")

    for line in lines:
        match = re.match(r'^\s*[-*•\d.)]+\s+(.*)', line)
        if match:
            item_text = match.group(1).strip()
            if item_text and len(item_text) > 10: # Basic filter
                 if item_text not in items:
                     items[item_text] = False
                     logger.debug(f"Extracted checklist item: {item_text}")
            else:
                logger.debug(f"Skipping short/empty matched line: {line}")
        else:
            logger.debug(f"Skipping non-list item line: {line}")
    return items

# --- Checklist Display Function (Improved Version) ---
def display_compliance_checklist():
    """Generates an interactive compliance checklist (experimental)."""
    st.subheader("✅ Compliance Checklist Generator (Experimental)")
    st.warning("""
        **Note:** Checklist items are generated by the AI based on its interpretation of the documents.
        This process may be incomplete or inaccurate. Always verify against the source documents.
        Formatting issues may still occur.
    """)

    if not st.session_state.initialized or not st.session_state.rag_pipeline:
        st.info("Pipeline must be initialized to generate checklist.")
        return

    # Button to trigger generation/update
    if st.button("Generate/Update Checklist"):
        st.session_state.checklist_items = {} # Clear previous items
        with st.spinner("🔍 Analyzing policies for compliance requirements... (This may take a moment)"):
            try:
                # Prepare filters (use all active docs for checklist)
                doc_filters = None
                if st.session_state.active_docs:
                    abs_doc_paths = [str((DATA_DIR / doc_path).resolve()) for doc_path in st.session_state.active_docs]
                    doc_filters = {"source": {"$in": abs_doc_paths}}
                    logger.info(f"Generating checklist using documents: {st.session_state.active_docs}")
                else:
                     logger.warning("Generating checklist with no document filter applied.")

                # Use more context (higher k) for checklist generation
                checklist_k = min(st.session_state.get('max_results', DEFAULT_K) * 3, 15) # Example: triple k, max 15
                logger.info(f"Using k={checklist_k} for checklist generation.")

                # --- Get relevant context first ---
                context_query = "Extract all sections related to compliance requirements, rules, procedures, and obligations."
                relevant_docs = st.session_state.rag_pipeline.vector_store_manager.similarity_search(
                    query=context_query,
                    k=checklist_k,
                    filter=doc_filters
                )
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])

                if not context_text:
                    st.warning("Could not retrieve relevant context for checklist generation.")
                    st.session_state.checklist_items = {}
                    # Use return instead of st.stop()
                    return

                # --- Call LLM with specific checklist prompt and retrieved context ---
                if hasattr(st.session_state.rag_pipeline, 'llm') and st.session_state.rag_pipeline.llm:
                    formatted_prompt = CHECKLIST_GENERATION_PROMPT.format(context=context_text) # Removed question=""
                    # TODO: Implement proper token counting and truncation if needed
                    max_context_len = 3000 # Conservative estimate
                    if len(formatted_prompt) > max_context_len:
                         logger.warning(f"Checklist prompt length ({len(formatted_prompt)}) exceeds limit, truncating context.")
                         truncated_context = context_text[:max_context_len - len(CHECKLIST_GENERATION_PROMPT.replace('{context}',''))]
                         formatted_prompt = CHECKLIST_GENERATION_PROMPT.format(context=truncated_context)

                    logger.debug("Invoking LLM directly for checklist generation.")
                    llm_response = st.session_state.rag_pipeline.llm.invoke(formatted_prompt)
                    logger.debug(f"Raw LLM checklist response: {llm_response}")
                    processed_items = parse_llm_checklist_output(llm_response)
                else:
                     st.error("LLM instance not available in RAG pipeline for direct invocation.")
                     processed_items = {}

                if not processed_items:
                     st.warning("Could not extract structured checklist items from the AI's response. The format might have been unexpected.")
                     # Optionally show the raw response for debugging
                     # st.text_area("Raw AI Response:", llm_response, height=200)

                st.session_state.checklist_items = processed_items

            except Exception as e:
                logger.error(f"Error generating checklist: {e}", exc_info=DEBUG)
                st.error(f"Failed to generate checklist: {handle_model_error(e)}")
                st.session_state.checklist_items = {} # Clear on error

    # Display checklist if items exist
    if st.session_state.checklist_items:
        items_list = list(st.session_state.checklist_items.items())
        for i, (req, checked) in enumerate(items_list):
            checkbox_key = f"check_{hash(req)}"
            # Display using markdown for better formatting
            new_state = st.checkbox(
                label=f"{req}", # Simpler label, rely on Streamlit's default wrapping
                value=checked,
                key=checkbox_key
            )
            if new_state != checked:
                st.session_state.checklist_items[req] = new_state
                st.rerun()

        completed = sum(st.session_state.checklist_items.values())
        total = len(st.session_state.checklist_items)
        if total > 0:
            st.progress(completed / total)
            st.caption(f"Completed {completed} of {total} identified requirements.")
        # Removed redundant else clause here

    elif 'checklist_items' in st.session_state and not st.session_state.checklist_items:
         st.info("Click 'Generate/Update Checklist' to attempt extraction.")


def display_collaboration_notes():
    """Real-time collaboration notes component."""
    st.subheader("👥 Collaboration Notes")
    notes = st.text_area(
        "Shared Notes (Supports Markdown)",
        value=st.session_state.collab_notes,
        height=300,
        key="collab_editor",
        help="Enter notes here. They are saved in the current session."
    )
    st.session_state.collab_notes = notes

    st.download_button(
        label="Export Notes",
        data=st.session_state.collab_notes,
        file_name="policy_discussion_notes.md",
        mime="text/markdown"
    )

# --- Main Application Logic ---
def main():
    st.title("🔒 Secure Policy Navigator")

    # Attempt to initialize the pipeline on first run
    if not st.session_state.initialized:
        initialize_pipeline()

    # Render sidebar
    display_sidebar()

    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Interface",
        "📊 Policy Graph",
        "✅ Compliance Checklist",
        "👥 Collaboration"
    ])

    # Render content based on initialization status
    if st.session_state.initialized:
        with tab1:
            display_chat_interface()
        with tab2:
            display_policy_graph()
        with tab3:
            display_compliance_checklist() # Calls the improved function
        with tab4:
            display_collaboration_notes()
    else:
        st.info("👈 Please wait for the pipeline to initialize or check sidebar/logs for errors.")


if __name__ == "__main__":
    # Ensure the app runs even if core initialization had issues (errors shown in UI)
    main()