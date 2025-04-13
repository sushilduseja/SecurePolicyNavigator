"""Core initialization module for Secure Policy Navigator."""

import os
import sys
from pathlib import Path
import logging
import asyncio
from typing import Any, Optional
from unittest.mock import MagicMock, PropertyMock

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MockTorchClassesPath:
    """Mock implementation of torch.classes __path__"""
    def __init__(self):
        self._path = [str(Path(torch.__file__).parent / "_classes"]

    def __iter__(self):
        return iter(self._path)

    def __contains__(self, item):
        return item in self._path

class MockTorchClasses:
    """Enhanced mock implementation of torch.classes"""
    def __init__(self):
        self.__path__ = MockTorchClassesPath()
        self._modules = {}

    def __getattr__(self, name: str) -> Any:
        if name not in self._modules:
            mock_module = MagicMock()
            mock_module.__path__ = MockTorchClassesPath()
            self._modules[name] = mock_module
        return self._modules[name]

def patch_torch() -> None:
    """Set up torch environment before any torch imports."""
    try:
        import torch
        
        if not hasattr(torch, "classes"):
            torch.classes = MockTorchClasses()
        
        # Add proper __path__ attribute
        torch.classes.__path__ = MockTorchClassesPath()
        
        # Patch custom class wrapper
        if not hasattr(torch._C, "_get_custom_class_python_wrapper"):
            torch._C._get_custom_class_python_wrapper = lambda *args, **kwargs: None
        
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
