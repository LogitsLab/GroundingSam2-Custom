#!/usr/bin/env python3
"""
GroundingSam2-Custom Unified Demo Script

This script provides a unified interface for image segmentation and video object tracking
using GroundingDINO for object detection and SAM 2 for precise segmentation.

Features:
- Image segmentation: Process single images or folders of images
- Video object tracking: Track objects across video frames
- Batch processing: Process entire folders of images/videos
- Flexible model selection: Choose from different model sizes
- Multiple output formats: Images, videos, JSON results

Usage:
    # Image segmentation
    python demo.py --input image.jpg --text "car. person."
    
    # Video tracking
    python demo.py --input video.mp4 --text "hippopotamus." --mode video
    
    # Batch processing
    python demo.py --input /path/to/folder --text "car. truck." --mode batch
    
    # Custom models
    python demo.py --input image.jpg --text "dog." --sam2-model large --gdino-model swinb
"""

import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from tqdm import tqdm
from PIL import Image
from core.models import load_models
from core.arguments import get_unified_parser, detect_input_type, get_files_from_folder
from core.processing import process_image, process_video

def parse_arguments():
    """Parse command line arguments for the unified demo."""
    return get_unified_parser().parse_args()

def main():
    """Main function for the unified demo."""
    args = parse_arguments()
    
    # Detect input type if auto mode
    if args.mode == "auto":
        try:
            input_type = detect_input_type(args.input)
            args.mode = input_type
        except Exception as e:
            print(f"❌ Error detecting input type: {e}")
            return
    elif args.mode == "batch":
        input_type = "folder"
    else:
        input_type = args.mode
    
    print(f"🚀 GroundingSam2-Custom Unified Demo")
    print(f"📊 Model Configuration:")
    print(f"   SAM 2: {args.sam2_model}")
    print(f"   GroundingDINO: {args.gdino_model}")
    print(f"🎯 Text Prompt: '{args.text}'")
    print(f"📁 Input: {args.input} ({input_type})")
    print(f"📁 Output Directory: {args.output_dir}")
    print("-" * 60)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load models
    grounding_model, sam2_predictor, sam2_config, device = load_models(
        args.sam2_model, args.gdino_model
    )
    
    # Process based on input type
    if input_type == "image":
        result = process_image(
            grounding_model, sam2_predictor, args.input, args.text,
            args.box_threshold, args.text_threshold, args.output_dir,
            not args.no_json, args.multimask, True, device
        )
        
    elif input_type == "video":
        result = process_video(
            grounding_model, sam2_predictor, args.input, args.text,
            args.box_threshold, args.text_threshold, args.output_dir,
            args.output_video, args.prompt_type, True, device
        )
        
    elif input_type == "folder":
        # Batch processing
        print(f"📁 Processing folder: {args.input}")
        
        # Get all files
        image_files = get_files_from_folder(args.input, args.image_extensions, args.max_files)
        video_files = get_files_from_folder(args.input, args.video_extensions, args.max_files)
        
        print(f"   Found {len(image_files)} images and {len(video_files)} videos")
        
        # Process images
        if image_files:
            print("🖼️  Processing images...")
            for img_file in tqdm(image_files, desc="Processing images"):
                try:
                    process_image(
                        grounding_model, sam2_predictor, str(img_file), args.text,
                        args.box_threshold, args.text_threshold, args.output_dir,
                        not args.no_json, args.multimask, True, device
                    )
                except Exception as e:
                    print(f"   ❌ Error processing {img_file}: {e}")
        
        # Process videos
        if video_files:
            print("🎬 Processing videos...")
            for vid_file in tqdm(video_files, desc="Processing videos"):
                try:
                    output_name = f"{vid_file.stem}_tracking.mp4"
                    process_video(
                        grounding_model, sam2_predictor, str(vid_file), args.text,
                        args.box_threshold, args.text_threshold, args.output_dir,
                        output_name, args.prompt_type, True, device
                    )
                except Exception as e:
                    print(f"   ❌ Error processing {vid_file}: {e}")
    
    print("✅ Processing complete!")
    print(f"📁 Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
