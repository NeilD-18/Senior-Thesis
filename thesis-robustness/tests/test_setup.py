#!/usr/bin/env python3
"""Test script to verify setup."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    # Import modules to trigger registration
    from src import datasets, models
    from src.common.registry import DATASETS, MODELS
    
    print("✓ Imports successful")
    print(f"✓ Registered datasets: {list(DATASETS.keys())}")
    print(f"✓ Registered models: {list(MODELS.keys())}")
    
    if len(DATASETS) == 0 or len(MODELS) == 0:
        print("\n⚠ Warning: No datasets or models registered. Check imports.")
        sys.exit(1)
    
    print("\n✓ Setup verification passed!")
    
except Exception as e:
    print(f"✗ Setup verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
