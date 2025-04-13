import os
import sys
from pathlib import Path
import traceback
import logging
import asyncio
import nest_asyncio
import graphviz
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_async():
    """Initialize async environment."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        nest_asyncio.apply()
    except Exception as e:
        logger.error(f"Failed to initialize async: {e}")

def setup_env():
    """Set up the environment before any imports."""
    setup_async()
    os.environ["STREAMLIT_TELEMETRY"] = "0"
    os.environ["TORCH_CLASSES"] = "0"

setup_env()

import streamlit as st
from src.utils.config import DEBUG
from src.rag.pipeline import RAGPipeline
from src.utils.error_handler import handle_model_error
from src.utils.document_loader import get_document_list

# Set page config
st.set_page_config(
    page_title="Secure Policy Navigator",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stButton > button {
        width: 100%;
        background-color: #00acee;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        margin: 0.5rem 0;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .node {
        fill: #4CAF50 !important;
        stroke: #388E3C !important;
    }
    .edgePath path {
        stroke: #666 !important;
    }
    .checklist-item {
        padding: 0.5rem;
        margin: 0.25rem 0;
        background: #f8f9fa;
        border-radius: 0.5rem;
    }
    """, unsafe_allow_html=True)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'checklist_items' not in st.session_state:
    st.session_state.checklist_items = {}
if 'collab_notes' not in st.session_state:
    st.session_state.collab_notes = ""
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = {"nodes": [], "edges": []}

def initialize_pipeline() -> None:
    """Initialize the RAG pipeline."""
    try:
        st.session_state.rag_pipeline = RAGPipeline()
        st.session_state.initialized = True
        st.success("✅ Pipeline initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        st.error(f"❌ Initialization failed: {str(e)}")

def build_policy_graph(documents: List[Document]) -> None:
    """Safe document graph construction"""
    G = nx.DiGraph()
    
    for doc in documents:
        # Handle Document objects or raw strings
        if isinstance(doc, Document):
            metadata = doc.metadata
            content = doc.page_content
        else:
            metadata = {}
            content = str(doc)
        
        doc_name = metadata.get("source", "Unknown")
        title = metadata.get("title", doc_name.split("/")[-1])
        
        G.add_node(doc_name, 
                  title=title,
                  type=metadata.get("type", "policy"),
                  content=content)
        
        # Extract references from content
        references = re.findall(r'\[(.*?)\]', content)
        for ref in references:
            if ref in G.nodes:
                G.add_edge(doc_name, ref)
    
    st.session_state.graph_data = {
        "nodes": [Node(id=n, label=G.nodes[n]['title'], shape="box") for n in G.nodes],
        "edges": [Edge(source=e[0], target=e[1], type="CURVE_SMOOTH") for e in G.edges]
    }

def generate_compliance_checklist():
    """Generate interactive compliance checklist."""
    if not st.session_state.initialized:
        return
    
    st.subheader("Compliance Checklist Generator")
    with st.spinner("🔍 Analyzing policies for compliance requirements..."):
        try:
            if not st.session_state.checklist_items:
                # Get base requirements from RAG
                requirements = st.session_state.rag_pipeline.process_query(
                    "List all compliance requirements from the documents"
                )["answer"].split("\n")
                
                # Process into checklist items
                st.session_state.checklist_items = {
                    req.strip(): False 
                    for req in requirements if req.strip()
                }
            
            # Display checklist
            for i, (req, checked) in enumerate(st.session_state.checklist_items.items()):
                cols = st.columns([1, 20])
                with cols[0]:
                    st.checkbox(
                        "Completion Status",
                        value=checked,
                        key=f"check_{i}",
                        label_visibility="collapsed"
                    )
                with cols[1]:
                    st.markdown(f'<div class="checklist-item">{req}</div>', 
                              unsafe_allow_html=True)
            
            # Progress tracking
            completed = sum(st.session_state.checklist_items.values())
            total = len(st.session_state.checklist_items)
            st.progress(completed/total if total > 0 else 0)
            st.caption(f"Completed {completed} of {total} requirements")
            
        except Exception as e:
            st.error(f"Error generating checklist: {str(e)}")

def collaboration_notes():
    """Real-time collaboration component."""
    st.subheader("Collaboration Notes")
    notes = st.text_area(
        "Shared Notes (Supports Markdown)",
        value=st.session_state.collab_notes,
        height=300,
        key="collab_editor"
    )
    st.session_state.collab_notes = notes
    
    # Add export button
    st.download_button(
        label="Export Notes",
        data=notes,
        file_name="policy_discussion.md",
        mime="text/markdown"
    )

def policy_dependency_graph():
    """Visualize policy relationships."""
    st.subheader("Policy Dependency Graph")
    
    if not st.session_state.initialized:
        st.warning("Initialize pipeline first")
        return
    
    with st.spinner("🔍 Analyzing policy relationships..."):
        try:
            # Get document relationships
            documents = st.session_state.rag_pipeline.vector_store_manager.vector_store.get()
            build_policy_graph(documents)
            
            # Graph configuration
            config = Config(
                width=1000,
                height=600,
                directed=True,
                nodeHighlightBehavior=True,
                highlightColor="#F7A7A6",
                collapsible=True,
                node={'labelProperty': 'label'},
                link={'highlightColor': '#8B008B'},
                physics=True,
                hierarchical=True
            )
            
            # Render graph
            return_value = agraph(
                nodes=st.session_state.graph_data["nodes"],
                edges=st.session_state.graph_data["edges"],
                config=config
            )
            
            # Add graph controls
            with st.expander("Graph Controls"):
                st.slider("Node Spacing", 100, 300, 150, key="node_spacing")
                st.selectbox("Layout", ["Hierarchical", "Force-directed"], key="layout")
            
        except Exception as e:
            st.error(f"Error generating graph: {str(e)}")

def main():
    st.title("🔒 Secure Policy Navigator")
    
    # Sidebar controls
    with st.sidebar:
        st.header("🛠️ Pipeline Controls")
        
        if not st.session_state.initialized:
            st.button("🚀 Initialize Pipeline", on_click=initialize_pipeline)
        
        st.header("⚙️ Advanced Settings")
        with st.expander("Search Parameters"):
            similarity_threshold = st.slider(
                "Similarity Threshold", 0.0, 1.0, 0.75,
                help="Minimum match confidence for document retrieval"
            )
            max_results = st.number_input(
                "Max Results", 1, 10, 3,
                help="Maximum number of documents to consider"
            )
        
        st.header("📚 Document Management")
        if st.session_state.initialized:
            try:
                documents = get_document_list()
                st.multiselect(
                    "Active Documents",
                    documents,
                    default=documents,
                    key="active_docs"
                )
            except Exception as e:
                st.error(f"Document error: {e}")

    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "💬 Chat Interface", 
        "📊 Policy Graph", 
        "✅ Compliance Checklist",
        "👥 Collaboration"
    ])

    with tab1:
        st.subheader("Policy Query Interface")
        if st.session_state.initialized:
            user_question = st.chat_input("Ask about security policies...")
            
            if user_question:
                with st.spinner("🔍 Analyzing policies..."):
                    progress_bar = st.progress(0)
                    try:
                        progress_bar.progress(20)
                        response = st.session_state.rag_pipeline.process_query(
                            user_question,
                            similarity_threshold=similarity_threshold,
                            max_results=max_results
                        )
                        progress_bar.progress(80)
                        
                        # Display chat
                        st.chat_message("user").write(user_question)
                        assistant_msg = st.chat_message("assistant")
                        assistant_msg.write(response["answer"])
                        
                        # Show sources
                        if response.get("sources"):
                            with st.expander("🔍 Source Documents"):
                                for source in response["sources"]:
                                    st.markdown(f"- `{source}`")
                        
                        progress_bar.progress(100)
                        st.session_state.chat_history.append(
                            (user_question, response["answer"])
                        )
                    except Exception as e:
                        st.error(f"Error: {handle_model_error(e)}")
                    finally:
                        progress_bar.empty()
        else:
            st.info("👈 Initialize pipeline to begin")

    with tab2:
        policy_dependency_graph()

    with tab3:
        generate_compliance_checklist()

    with tab4:
        collaboration_notes()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}")
        st.error("Application error - check logs for details")