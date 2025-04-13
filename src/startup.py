import os
import sys
import asyncio
from pathlib import Path
import logging
from unittest.mock import MagicMock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_torch():
    """Set up torch environment with proper mocking."""
    try:
        import torch
        from unittest.mock import MagicMock

        # Mock torch.classes with a valid __path__ attribute
        class MockPath:
            def __init__(self):
                self._path = []

            def __getattr__(self, name):
                return []

        class MockClasses:
            def __init__(self):
                self.__path__ = MockPath()

            def __getattr__(self, name):
                return MagicMock()

        if not hasattr(torch, "classes"):
            torch.classes = MockClasses()

        if not hasattr(torch._C, "_get_custom_class_python_wrapper"):
            torch._C._get_custom_class_python_wrapper = lambda *args, **kwargs: None

        logger.info("Torch environment configured successfully")
    except Exception as e:
        logger.warning(f"Could not configure torch: {e}")

def setup_async():
    """Initialize async environment."""
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    logger.info("Async environment configured successfully")

def setup_environment():
    """Initialize the complete environment."""
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Set environment variables
    os.environ["STREAMLIT_TELEMETRY"] = "0"
    os.environ["TORCH_CLASSES"] = "0"
    
    # Initialize components
    setup_async()
    setup_torch()
    logger.info("Environment setup complete")

def run_app():
    """Run the Streamlit application."""
    try:
        setup_environment()
        app_path = Path(__file__).parent / "app.py"
        if not app_path.exists():
            raise FileNotFoundError(f"App file not found: {app_path}")
        os.system(f"streamlit run {app_path}")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_app()
