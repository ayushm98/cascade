"""Hugging Face Spaces entry point for Cascade UI."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import and run the main app
from cascade.ui.app import main

if __name__ == "__main__":
    main()
