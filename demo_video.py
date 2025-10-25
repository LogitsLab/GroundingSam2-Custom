import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from core.models import load_models, setup_mixed_precision
from core.arguments import get_video_parser
from core.processing import process_video

"""
GroundingSam2-Custom Video Object Tracking Demo

This script demonstrates video object tracking using GroundingDINO for object detection
and SAM 2 for precise segmentation and tracking across video frames. Users can select
different model sizes based on their performance and accuracy requirements.

Usage:
    python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py --sam2-model large --gdino-model swint --text "hippopotamus." --video "path/to/video.mp4"
"""

def parse_arguments():
    """Parse command line arguments for model selection and parameters."""
    return get_video_parser().parse_args()

def main():
    """Main function to run the video tracking demo."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up parameters
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    VIDEO_PATH = args.video
    TEXT_PROMPT = args.text
    OUTPUT_VIDEO_PATH = args.output_video
    PROMPT_TYPE_FOR_VIDEO = args.prompt_type
    
    print(f"🚀 GroundingSam2-Custom Video Object Tracking Demo")
    print(f"📊 Model Configuration:")
    print(f"   SAM 2: {args.sam2_model}")
    print(f"   GroundingDINO: {args.gdino_model}")
    print(f"🎯 Text Prompt: '{TEXT_PROMPT}'")
    print(f"🎬 Input Video: {VIDEO_PATH}")
    print(f"📝 Prompt Type: {PROMPT_TYPE_FOR_VIDEO}")
    print(f"📁 Output Video: {OUTPUT_VIDEO_PATH}")
    print("-" * 60)

    # Load models
    grounding_model, sam2_predictor, sam2_config, device = load_models(
        args.sam2_model, args.gdino_model
    )
    
    print("🎬 Processing video...")
    
    # Process the video using the core processing function
    result = process_video(
        grounding_model, sam2_predictor, VIDEO_PATH, TEXT_PROMPT,
        BOX_THRESHOLD, TEXT_THRESHOLD, "./", OUTPUT_VIDEO_PATH,
        PROMPT_TYPE_FOR_VIDEO, True, device
    )
    
    if result:
        print("✅ Video tracking complete!")
        print(f"🎬 Output video: {result}")
    else:
        print("❌ No objects detected in the video")

if __name__ == "__main__":
    main()