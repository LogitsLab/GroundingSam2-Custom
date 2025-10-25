"""
Fallback module for GroundingDINO _C extensions when C++ compilation is not available.
This provides dummy implementations to prevent ImportError.
"""

import torch
import warnings

warnings.warn("GroundingDINO C++ extensions not available. Using fallback implementations.")

class DummyModule:
    """Dummy module to replace missing _C extensions."""
    
    @staticmethod
    def ms_deform_attn_forward(*args, **kwargs):
        """Dummy forward function for multi-scale deformable attention."""
        warnings.warn("Using dummy implementation for ms_deform_attn_forward")
        return torch.zeros(1)
    
    @staticmethod
    def ms_deform_attn_backward(*args, **kwargs):
        """Dummy backward function for multi-scale deformable attention."""
        warnings.warn("Using dummy implementation for ms_deform_attn_backward")
        return torch.zeros(1)

# Create a dummy module that can be imported
_C = DummyModule()
