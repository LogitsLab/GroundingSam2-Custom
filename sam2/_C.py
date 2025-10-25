"""
Fallback module for SAM 2 _C extensions when C++ compilation is not available.
This provides dummy implementations to prevent ImportError.
"""

import torch
import warnings

warnings.warn("SAM 2 C++ extensions not available. Using fallback implementations.")

class DummyModule:
    """Dummy module to replace missing _C extensions."""
    
    @staticmethod
    def get_connected_componnets(mask):
        """Dummy function for connected components."""
        warnings.warn("Using dummy implementation for get_connected_componnets")
        return mask

# Create a dummy module that can be imported
_C = DummyModule()
