"""Core initialization module for Secure Policy Navigator."""

import os
import sys
from pathlib import Path
import logging
import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockTorchClasses:
    """Mock implementation of torch.classes to prevent runtime errors."""
    def __init__(self) -> None:
        self._modules = {}
        self.__path__ = []  # Mock __path__ as an empty list

    def __getattr__(self, name: str) -> Any:
        if name not in self._modules:
            self._modules[name] = MagicMock()
        return self._modules[name]

def patch_torch() -> None:
    """Set up torch environment before any torch imports."""
    try:
        import torch
        
        # Mock torch.classes with a valid __path__ attribute
        if not hasattr(torch, "classes"):
            torch.classes = MockTorchClasses()
        
        # Patch custom class wrapper
        if not hasattr(torch._C, "_get_custom_class_python_wrapper"):
            setattr(torch._C, "_get_custom_class_python_wrapper", lambda *args, **kwargs: None)
        
        logger.info("Torch environment patched successfully")
    except ImportError:
        logger.warning("Could not patch torch - import failed")

def setup_async() -> Optional[asyncio.AbstractEventLoop]:
    """Initialize async environment."""
    try:
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        import nest_asyncio
        nest_asyncio.apply()
        
        logger.info("Async environment initialized successfully")
        return loop
    except Exception as e:
        logger.error(f"Failed to initialize async environment: {e}")
        return None

def initialize() -> None:
    """Initialize the complete environment."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ["STREAMLIT_TELEMETRY"] = "0"
    os.environ["TORCH_CLASSES"] = "0"
    
    # Patch torch before any other imports
    patch_torch()
    
    # Initialize async environment
    setup_async()
    
    logger.info("Environment initialized successfully")
