"""
Argument parsing utilities for GroundingSam2-Custom.

This module provides standardized argument parsing for all demo scripts,
ensuring consistency across different interfaces.
"""

import argparse
from pathlib import Path

def get_base_parser(description, epilog=None):
    """Get base argument parser with common options."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog
    )
    
    # Model selection arguments
    parser.add_argument("--sam2-model", 
                       choices=["tiny", "small", "base", "large"], 
                       default="large",
                       help="SAM 2 model size (default: large)")
    
    parser.add_argument("--gdino-model", 
                       choices=["swint", "swinb"], 
                       default="swinb",
                       help="GroundingDINO model (default: swinb)")
    
    # Detection thresholds
    parser.add_argument("--box-threshold", 
                       type=float, 
                       default=0.25,
                       help="Box detection threshold (default: 0.25)")
    
    parser.add_argument("--text-threshold", 
                       type=float, 
                       default=0.25,
                       help="Text detection threshold (default: 0.25)")
    
    return parser

def get_image_parser():
    """Get argument parser for image segmentation."""
    parser = get_base_parser(
        description="GroundingSam2-Custom Image Segmentation Demo",
        epilog="""
Model Options:
  SAM 2 Models:
    tiny    - Fastest, lowest quality (156MB)
    small   - Fast, good quality (184MB) 
    base    - Balanced performance (323MB)
    large   - Best quality, slower (898MB)
    
  GroundingDINO Models:
    swint   - Faster detection (693MB)
    swinb   - Best accuracy (938MB)
        """
    )
    
    # Input parameters
    parser.add_argument("--text", 
                       default="car. tire.",
                       help="Text prompt for object detection (default: 'car. tire.')")
    
    parser.add_argument("--image", 
                       default="notebooks/images/truck.jpg",
                       help="Path to input image (default: 'notebooks/images/truck.jpg')")
    
    # Output options
    parser.add_argument("--output-dir", 
                       default="outputs/grounded_sam2_local_demo",
                       help="Output directory (default: 'outputs/grounded_sam2_local_demo')")
    
    parser.add_argument("--no-json", 
                       action="store_true",
                       help="Skip JSON results export")
    
    parser.add_argument("--multimask", 
                       action="store_true",
                       help="Enable multimask output")
    
    return parser

def get_video_parser():
    """Get argument parser for video tracking."""
    parser = get_base_parser(
        description="GroundingSam2-Custom Video Object Tracking Demo",
        epilog="""
Model Options:
  SAM 2 Models:
    tiny    - Fastest, lowest quality (156MB)
    small   - Fast, good quality (184MB) 
    base    - Balanced performance (323MB)
    large   - Best quality, slower (898MB)
    
  GroundingDINO Models:
    swint   - Faster detection (693MB)
    swinb   - Best accuracy (938MB)
    
  Prompt Types:
    point   - Use point prompts for tracking
    box     - Use bounding box prompts (default)
    mask    - Use mask prompts for tracking
        """
    )
    
    # Input parameters
    parser.add_argument("--text", 
                       default="hippopotamus.",
                       help="Text prompt for object detection (default: 'hippopotamus.')")
    
    parser.add_argument("--video", 
                       default="./assets/hippopotamus.mp4",
                       help="Path to input video (default: './assets/hippopotamus.mp4')")
    
    # Tracking options
    parser.add_argument("--prompt-type", 
                       choices=["point", "box", "mask"], 
                       default="box",
                       help="Prompt type for video tracking (default: box)")
    
    # Output options
    parser.add_argument("--output-video", 
                       default="./tracking_demo_output.mp4",
                       help="Output video path (default: './tracking_demo_output.mp4')")
    
    parser.add_argument("--frames-dir", 
                       default="./custom_video_frames",
                       help="Directory for extracted frames (default: './custom_video_frames')")
    
    parser.add_argument("--results-dir", 
                       default="./tracking_results",
                       help="Directory for tracking results (default: './tracking_results')")
    
    return parser

def get_unified_parser():
    """Get argument parser for unified demo."""
    parser = get_base_parser(
        description="GroundingSam2-Custom Unified Demo - Image Segmentation & Video Tracking",
        epilog="""
Examples:
  # Image segmentation
  python demo.py --input image.jpg --text "car. person."
  
  # Video tracking
  python demo.py --input video.mp4 --text "hippopotamus." --mode video
  
  # Batch processing folder
  python demo.py --input /path/to/folder --text "dog. cat." --mode batch
  
  # Custom models and settings
  python demo.py --input image.jpg --text "car." --sam2-model large --gdino-model swinb --box-threshold 0.3

Model Options:
  SAM 2 Models:
    tiny    - Fastest, lowest quality (156MB)
    small   - Fast, good quality (184MB) 
    base    - Balanced performance (323MB)
    large   - Best quality, slower (898MB)
    
  GroundingDINO Models:
    swint   - Faster detection (693MB)
    swinb   - Best accuracy (938MB)
        """
    )
    
    # Input parameters
    parser.add_argument("--input", "-i", 
                       required=True,
                       help="Input path: image file, video file, or folder path")
    
    parser.add_argument("--text", "-t", 
                       required=True,
                       help="Text prompt for object detection (e.g., 'car. person.')")
    
    parser.add_argument("--mode", 
                       choices=["auto", "image", "video", "batch"], 
                       default="auto",
                       help="Processing mode (default: auto-detect)")
    
    # Video-specific options
    parser.add_argument("--prompt-type", 
                       choices=["point", "box", "mask"], 
                       default="box",
                       help="Prompt type for video tracking (default: box)")
    
    # Output options
    parser.add_argument("--output-dir", "-o", 
                       default="outputs",
                       help="Output directory (default: 'outputs')")
    
    parser.add_argument("--output-video", 
                       default="tracking_output.mp4",
                       help="Output video filename for video mode (default: 'tracking_output.mp4')")
    
    parser.add_argument("--no-json", 
                       action="store_true",
                       help="Skip JSON results export")
    
    parser.add_argument("--multimask", 
                       action="store_true",
                       help="Enable multimask output for images")
    
    # Processing options
    parser.add_argument("--max-files", 
                       type=int, 
                       default=None,
                       help="Maximum number of files to process in batch mode")
    
    parser.add_argument("--image-extensions", 
                       nargs="+", 
                       default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                       help="Image file extensions for batch processing")
    
    parser.add_argument("--video-extensions", 
                       nargs="+", 
                       default=[".mp4", ".avi", ".mov", ".mkv"],
                       help="Video file extensions for batch processing")
    
    return parser

def detect_input_type(input_path):
    """Detect the type of input (image, video, or folder)."""
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    
    if input_path.is_file():
        # Check file extension
        ext = input_path.suffix.lower()
        if ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            return "image"
        elif ext in [".mp4", ".avi", ".mov", ".mkv", ".wmv"]:
            return "video"
        else:
            raise ValueError(f"Unsupported file type: {ext}")
    elif input_path.is_dir():
        return "folder"
    else:
        raise ValueError(f"Invalid input path: {input_path}")

def get_files_from_folder(folder_path, extensions, max_files=None):
    """Get list of files from folder with specified extensions."""
    folder_path = Path(folder_path)
    files = []
    
    for ext in extensions:
        files.extend(folder_path.glob(f"*{ext}"))
        files.extend(folder_path.glob(f"*{ext.upper()}"))
    
    files = sorted(files)
    
    if max_files:
        files = files[:max_files]
    
    return files
