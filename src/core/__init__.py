"""Core initialization module for Secure Policy Navigator."""

import os
import sys
import asyncio
from pathlib import Path
import logging
from unittest.mock import MagicMock # Re-added for minimal patch

# Configure logging early (might be reconfigured by config.py later)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Minimal Torch Patch (Workaround for Streamlit Watcher Issue) ---
# This attempts to prevent the specific RuntimeError related to accessing
# torch.classes.__path__._path during Streamlit's module inspection.
# WARNING: Still a workaround. Investigate dependency conflicts if possible.
def patch_torch_minimal():
    """Applies minimal patch for torch.classes.__path__ access issue."""
    try:
        import torch
        logger.debug("Applying minimal torch patch for __path__ access...")

        # Check if torch.classes exists and if its __path__ needs mocking
        if hasattr(torch, "classes") and hasattr(torch.classes, "__path__"):
            # Check if accessing _path directly causes an error or if it's missing
            try:
                # Attempt the problematic access to see if it errors
                _ = torch.classes.__path__._path
            except (RuntimeError, AttributeError):
                 logger.warning("torch.classes.__path__._path access issue detected. Applying mock.")
                 # Mock the __path__ object itself to have a dummy _path attribute
                 path_mock = MagicMock()
                 path_mock._path = [] # Provide the expected attribute as an empty list
                 torch.classes.__path__ = path_mock

        logger.debug("Minimal torch patching applied (if necessary).")

    except ImportError:
        logger.warning("PyTorch not found. Skipping torch patching.")
    except Exception as e:
        logger.error(f"Error during minimal torch patching: {e}", exc_info=True)
# --- End of Minimal Torch Patch ---


# --- Asyncio Setup ---
def setup_async():
    """Initializes the asyncio event loop policy, especially for Windows."""
    try:
        logger.debug("Setting up asyncio event loop policy...")
        if sys.platform == "win32":
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            logger.info("Set WindowsSelectorEventLoopPolicy for asyncio.")

        import nest_asyncio
        nest_asyncio.apply()
        logger.info("Applied nest_asyncio.")

    except ImportError:
         logger.warning("nest_asyncio not found. Skipping application.")
    except Exception as e:
        logger.error(f"Failed to setup async environment: {e}", exc_info=True)

# --- Main Initialization Function ---
def initialize():
    """Performs essential setup tasks before the application runs."""
    print("Running core initialization...")

    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added project root to sys.path: {project_root}")

    os.environ.setdefault("STREAMLIT_TELEMETRY", "0")

    # Apply Minimal Torch Patch
    patch_torch_minimal() # Call the minimal patch

    setup_async()

    try:
        from dotenv import load_dotenv
        dotenv_path = project_root / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path=dotenv_path)
            print(f"Loaded environment variables from: {dotenv_path}")
        else:
            print("No .env file found at project root.")
    except ImportError:
        print("python-dotenv not installed. Cannot load .env file.")
    except Exception as e:
        print(f"Error loading .env file: {e}")

    print("Core initialization sequence finished.")

# Run initialization when this module is imported
# initialize() # Commented out - let app.py call it explicitly if needed