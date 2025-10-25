"""
Model configuration and loading utilities for GroundingSam2-Custom.

This module provides centralized model configurations, loading functions,
and device setup for both SAM 2 and GroundingDINO models.
"""

import torch
import warnings
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model

# Suppress specific warnings
warnings.filterwarnings('ignore', message='torch.meshgrid.*indexing')
warnings.filterwarnings('ignore', message='torch.utils.checkpoint.*use_reentrant')
warnings.filterwarnings('ignore', message='NumPy array.*writable')
warnings.filterwarnings('ignore', message='.*C\+\+ extensions not available.*')

# SAM 2 model configurations
SAM2_MODELS = {
    "tiny": {
        "name": "SAM 2.1 Tiny",
        "size": "156MB",
        "speed": "Fastest",
        "quality": "Good",
        "checkpoint": "./checkpoints/sam2.1_hiera_tiny.pt",
        "config": "configs/sam2.1/sam2.1_hiera_t.yaml"
    },
    "small": {
        "name": "SAM 2.1 Small", 
        "size": "184MB",
        "speed": "Fast",
        "quality": "Better",
        "checkpoint": "./checkpoints/sam2.1_hiera_small.pt",
        "config": "configs/sam2.1/sam2.1_hiera_s.yaml"
    },
    "base": {
        "name": "SAM 2.1 Base+",
        "size": "323MB", 
        "speed": "Medium",
        "quality": "High",
        "checkpoint": "./checkpoints/sam2.1_hiera_base_plus.pt",
        "config": "configs/sam2.1/sam2.1_hiera_b+.yaml"
    },
    "large": {
        "name": "SAM 2.1 Large",
        "size": "898MB",
        "speed": "Slower", 
        "quality": "Best",
        "checkpoint": "./checkpoints/sam2.1_hiera_large.pt",
        "config": "configs/sam2.1/sam2.1_hiera_l.yaml"
    }
}

# GroundingDINO model configurations
GDINO_MODELS = {
    "swint": {
        "name": "GroundingDINO SwinT",
        "size": "693MB",
        "speed": "Faster",
        "quality": "Good",
        "checkpoint": "gdino_checkpoints/groundingdino_swint_ogc.pth",
        "config": "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    },
    "swinb": {
        "name": "GroundingDINO SwinB",
        "size": "938MB", 
        "speed": "Slower",
        "quality": "Best",
        "checkpoint": "gdino_checkpoints/groundingdino_swinb_cogcoor.pth",
        "config": "grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    }
}

def get_model_paths(sam2_model, gdino_model):
    """Get model checkpoint and config paths based on user selection."""
    if sam2_model not in SAM2_MODELS:
        raise ValueError(f"Invalid SAM 2 model: {sam2_model}. Choose from {list(SAM2_MODELS.keys())}")
    if gdino_model not in GDINO_MODELS:
        raise ValueError(f"Invalid GroundingDINO model: {gdino_model}. Choose from {list(GDINO_MODELS.keys())}")
    
    return SAM2_MODELS[sam2_model], GDINO_MODELS[gdino_model]

def setup_device():
    """Setup and return the appropriate device (CUDA or CPU)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def setup_mixed_precision(device):
    """Setup mixed precision for better performance."""
    torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
    
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        # Turn on tfloat32 for Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def load_grounding_dino(gdino_model, device):
    """Load GroundingDINO model."""
    gdino_config = GDINO_MODELS[gdino_model]
    grounding_model = load_model(
        model_config_path=gdino_config["config"],
        model_checkpoint_path=gdino_config["checkpoint"],
        device=device
    )
    return grounding_model

def load_sam2(sam2_model, device):
    """Load SAM 2 model and predictor."""
    sam2_config = SAM2_MODELS[sam2_model]
    sam2_model_obj = build_sam2(sam2_config["config"], sam2_config["checkpoint"], device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model_obj)
    return sam2_predictor, sam2_config

def load_sam2_video(sam2_model, device):
    """Load SAM 2 video predictor."""
    sam2_config = SAM2_MODELS[sam2_model]
    video_predictor = build_sam2_video_predictor(sam2_config["config"], sam2_config["checkpoint"])
    return video_predictor, sam2_config

def load_models(sam2_model, gdino_model, device=None):
    """Load both GroundingDINO and SAM 2 models."""
    if device is None:
        device = setup_device()
    
    print("🔧 Loading models...")
    
    # Load GroundingDINO
    print(f"   Loading GroundingDINO ({gdino_model})...")
    grounding_model = load_grounding_dino(gdino_model, device)
    
    # Load SAM 2
    print(f"   Loading SAM 2 ({sam2_model})...")
    sam2_predictor, sam2_config = load_sam2(sam2_model, device)
    
    return grounding_model, sam2_predictor, sam2_config, device

def get_model_info(sam2_model, gdino_model):
    """Get model information for display."""
    sam2_info = SAM2_MODELS[sam2_model]
    gdino_info = GDINO_MODELS[gdino_model]
    
    return {
        "sam2": sam2_info,
        "gdino": gdino_info
    }

# Streamlit cached model loading
try:
    import streamlit as st
    
    @st.cache_resource
    def load_models_cached(sam2_model, gdino_model):
        """Load models with Streamlit caching for better performance."""
        return load_models(sam2_model, gdino_model)
        
except ImportError:
    # If streamlit is not available, provide a fallback
    def load_models_cached(sam2_model, gdino_model):
        """Fallback function when streamlit is not available."""
        return load_models(sam2_model, gdino_model)
