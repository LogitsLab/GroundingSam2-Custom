import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from core.models import load_models, setup_mixed_precision
from core.arguments import get_image_parser
from core.processing import process_image

"""
GroundingSam2-Custom Image Segmentation Demo

This script demonstrates image segmentation using GroundingDINO for object detection
and SAM 2 for precise segmentation. Users can select different model sizes based on
their performance and accuracy requirements.

Usage:
    python grounded_sam2_local_demo.py --sam2-model large --gdino-model swint --text "car. tire." --image "path/to/image.jpg"
"""

def parse_arguments():
    """Parse command line arguments for model selection and parameters."""
    return get_image_parser().parse_args()

def main():
    """Main function to run the image segmentation demo."""
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up parameters
    TEXT_PROMPT = args.text
    IMG_PATH = args.image
    BOX_THRESHOLD = args.box_threshold
    TEXT_THRESHOLD = args.text_threshold
    OUTPUT_DIR = Path(args.output_dir)
    DUMP_JSON_RESULTS = not args.no_json
    MULTIMASK_OUTPUT = args.multimask
    
    print(f"🚀 GroundingSam2-Custom Image Segmentation Demo")
    print(f"📊 Model Configuration:")
    print(f"   SAM 2: {args.sam2_model}")
    print(f"   GroundingDINO: {args.gdino_model}")
    print(f"🎯 Text Prompt: '{TEXT_PROMPT}'")
    print(f"🖼️  Input Image: {IMG_PATH}")
    print(f"📁 Output Directory: {OUTPUT_DIR}")
    print("-" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load models
    grounding_model, sam2_predictor, sam2_config, device = load_models(
        args.sam2_model, args.gdino_model
    )
    
    print("🎯 Processing image...")
    
    # Process the image using the core processing function
    result = process_image(
        grounding_model, sam2_predictor, IMG_PATH, TEXT_PROMPT,
        BOX_THRESHOLD, TEXT_THRESHOLD, str(OUTPUT_DIR),
        DUMP_JSON_RESULTS, MULTIMASK_OUTPUT, True, device
    )
    
    if result:
        print("✅ Processing complete!")
        print(f"📁 Results saved to: {OUTPUT_DIR}")
        print(f"   • Segmented image: {result}")
        if DUMP_JSON_RESULTS:
            print(f"   • JSON results: {result.parent / f'{result.stem}_results.json'}")
    else:
        print("❌ No objects detected in the image")

if __name__ == "__main__":
    main()
