#!/usr/bin/env python3
"""
GroundingSam2-Custom Streamlit App

Interactive web interface for image segmentation and video object tracking
using GroundingDINO for object detection and SAM 2 for precise segmentation.

Features:
- Upload images, videos, or select folders
- Model selection with performance indicators
- Customizable detection thresholds
- Predefined class checklists with custom additions
- Side-by-side input/output visualization
- JSON results display
- Real-time processing feedback

Usage:
    streamlit run streamlit_app.py
"""

import streamlit as st
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
import tempfile
import shutil
from pathlib import Path
from torchvision.ops import box_convert
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO
import time
import warnings

# Suppress specific warnings
warnings.filterwarnings('ignore', message='torch.meshgrid.*indexing')
warnings.filterwarnings('ignore', message='torch.utils.checkpoint.*use_reentrant')
warnings.filterwarnings('ignore', message='NumPy array.*writable')
warnings.filterwarnings('ignore', message='.*C\+\+ extensions not available.*')

# Import core modules
from core.models import SAM2_MODELS, GDINO_MODELS, load_models_cached
from core.streamlit_utils import (
    get_model_selection_ui, get_threshold_ui, get_display_options_ui, 
    get_class_selection_ui, get_custom_css, process_image_streamlit, 
    process_video_streamlit, PREDEFINED_CLASSES
)

# Page configuration
st.set_page_config(
    page_title="GroundingSam2-Custom",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown(get_custom_css(), unsafe_allow_html=True)

def main():
    """Main Streamlit app."""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 GroundingSam2-Custom</h1>', unsafe_allow_html=True)
    st.markdown("**Interactive Image Segmentation & Video Object Tracking**")
    
    # Sidebar with all controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        st.subheader("🤖 Model Selection")
        sam2_model, gdino_model = get_model_selection_ui()
        
        # Thresholds
        st.subheader("🎯 Detection Thresholds")
        box_threshold, text_threshold = get_threshold_ui()
        
        # Display options
        st.subheader("👁️ Display Options")
        show_boxes, show_json = get_display_options_ui()
        
        # Class selection
        st.subheader("🏷️ Object Classes")
        text_prompt = get_class_selection_ui()
        
        st.markdown("---")
    
    # Main content area - Input section
    st.markdown("---")
    st.header("📤 Input")
    
    # Input controls in main area
    col1, col2 = st.columns(2)
    
    with col1:
        # Output directory selection
        st.subheader("📁 Output Directory")
        output_dir = st.text_input(
            "Output Directory:",
            value="./output",
            help="Directory where results will be saved"
        )
    
    with col2:
        # Input type selection
        st.subheader("📁 Input Type")
        input_type = st.radio(
            "Select input type:",
            ["Image", "Video", "Folder"],
            help="Choose the type of input you want to process"
        )
    
    # File upload based on input type
    if input_type == "Image":
        uploaded_file = st.file_uploader(
            "Upload an image:",
            type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            help="Upload an image file for segmentation"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_path = tmp_file.name
            
            # Store uploaded file for later display (don't show here to avoid duplication)
            st.success(f"✅ Image uploaded: {uploaded_file.name}")
            
    elif input_type == "Video":
        uploaded_file = st.file_uploader(
            "Upload a video:",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for object tracking"
        )
        
        if uploaded_file is not None:
            # Save uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                input_path = tmp_file.name
            
            # Display video info
            st.video(uploaded_file)
            
    else:  # Folder
        folder_path = st.text_input(
            "Folder Path:",
            help="Enter the path to a folder containing images or videos"
        )
        
        if folder_path and os.path.exists(folder_path):
            # List files in folder
            files = []
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.mp4', '.avi', '.mov', '.mkv']:
                files.extend(Path(folder_path).glob(f"*{ext}"))
                files.extend(Path(folder_path).glob(f"*{ext.upper()}"))
            
            st.write(f"Found {len(files)} files in folder")
            if files:
                st.write("Files:", [f.name for f in files[:10]])  # Show first 10 files
                input_path = folder_path
    
    st.markdown("---")
    
    # Process button - centered and prominent
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button("🚀 Process", type="primary", use_container_width=True)
    
    if process_button:
        if 'input_path' in locals() and input_path:
            try:
                # Load models
                with st.spinner("Loading models..."):
                    grounding_model, sam2_predictor, sam2_config, device = load_models_cached(sam2_model, gdino_model)
                
                # Process based on input type
                if input_type == "Image":
                    with st.spinner("Processing image..."):
                        result_image, json_results = process_image_streamlit(
                            grounding_model, sam2_predictor, input_path, text_prompt,
                            box_threshold, text_threshold, show_boxes, device, output_dir
                        )
                    
                    if result_image is not None:
                        # Display results in a clean layout
                        st.markdown("### 🎯 Segmentation Results")
                        
                        # Create two columns for input and output comparison
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**📤 Input Image**")
                            # Show original image if available
                            if 'uploaded_file' in locals() and uploaded_file is not None:
                                st.image(uploaded_file, caption="Original", width="stretch")
                            else:
                                st.info("Original image not available for display")
                        
                        with col2:
                            st.markdown("**📤 Segmented Image**")
                            st.image(result_image, caption="Segmented", width="stretch")
                        
                        # Results information
                        st.markdown("---")
                        st.markdown("### 📊 Results Information")
                        
                        # Output directory structure
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**📁 Results saved to:** `{output_dir}`")
                            st.markdown("**📂 Output structure:**")
                            st.markdown(f"""
                            ```
                            {output_dir}/
                            ├── images/
                            │   └── {Path(input_path).stem}_segmented.jpg
                            ├── json_results/
                            │   └── {Path(input_path).stem}_results.json
                            └── logs/
                            ```""")
                        
                        with col2:
                            if show_json and json_results:
                                st.markdown("**📋 JSON Results:**")
                                # Make JSON results scrollable
                                import json
                                json_str = json.dumps(json_results, indent=2)
                                st.text_area(
                                    "JSON Results:",
                                    value=json_str,
                                    height=200,
                                    help="JSON results - scroll to see all data",
                                    key="json_results_compact"
                                )
                            else:
                                st.info("JSON results available in output directory")
                        
                        # Show JSON results in a dedicated section if selected
                        if show_json and json_results:
                            st.markdown("---")
                            st.markdown("### 📋 Detailed JSON Results")
                            st.markdown("**Detection Results:**")
                            
                            # Display JSON in a more readable format
                            if isinstance(json_results, dict) and 'annotations' in json_results:
                                st.markdown(f"**Total Objects Detected:** {len(json_results['annotations'])}")
                                
                                # Create a scrollable container for individual detections
                                with st.container():
                                    st.markdown("**Individual Object Details:**")
                                    
                                    # Create a scrollable area for the expandable sections
                                    detection_container = st.container()
                                    with detection_container:
                                        for i, annotation in enumerate(json_results['annotations']):
                                            with st.expander(f"Object {i+1}: {annotation.get('class_name', 'Unknown')} (Score: {annotation.get('score', 'N/A'):.3f})"):
                                                # Make the JSON content scrollable
                                                st.markdown("**Full Annotation Data:**")
                                                st.json(annotation)
                                                
                                                # Show key information in a more readable format
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.markdown(f"**Class:** {annotation.get('class_name', 'Unknown')}")
                                                    st.markdown(f"**Score:** {annotation.get('score', 'N/A'):.3f}")
                                                with col2:
                                                    bbox = annotation.get('bbox', [])
                                                    if bbox:
                                                        st.markdown(f"**Bounding Box:** [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                                                    else:
                                                        st.markdown("**Bounding Box:** Not available")
                                
                                # Show complete JSON in a scrollable container
                                st.markdown("**Complete JSON Results:**")
                                with st.container():
                                    # Create a scrollable text area for the complete JSON
                                    import json
                                    json_str = json.dumps(json_results, indent=2)
                                    st.text_area(
                                        "Full JSON Output:",
                                        value=json_str,
                                        height=300,
                                        help="Complete JSON results - scroll to see all data"
                                    )
                            else:
                                # For non-standard JSON format, show in scrollable container
                                with st.container():
                                    st.json(json_results)
                    else:
                        st.error("❌ No objects detected in the image")
                
                elif input_type == "Video":
                    with st.spinner("Processing video..."):
                        result_video, video_info = process_video_streamlit(
                            grounding_model, sam2_predictor, input_path, text_prompt,
                            box_threshold, text_threshold, show_boxes, device, output_dir
                        )
                    
                    if result_video:
                        st.markdown("### 🎬 Video Tracking Results")
                        
                        # Display the processed video
                        st.video(result_video)
                        
                        # Results information
                        st.markdown("---")
                        st.markdown("### 📊 Results Information")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**📁 Results saved to:** `{output_dir}`")
                            st.markdown("**📂 Output structure:**")
                            st.markdown(f"""
                            ```
                            {output_dir}/
                            ├── videos/
                            │   └── {Path(input_path).stem}_tracking.mp4
                            ├── tracking_frames/
                            │   └── annotated_frame_*.jpg
                            ├── json_results/
                            └── logs/
                            ```""")
                        
                        with col2:
                            if show_json and video_info:
                                st.markdown("**📋 Video Info:**")
                                # Make video info scrollable
                                import json
                                video_json_str = json.dumps(video_info, indent=2)
                                st.text_area(
                                    "Video Info:",
                                    value=video_json_str,
                                    height=200,
                                    help="Video information - scroll to see all data",
                                    key="video_info_compact"
                                )
                            else:
                                st.info("Video info available in output directory")
                        
                        # Show detailed video info if JSON is selected
                        if show_json and video_info:
                            st.markdown("---")
                            st.markdown("### 📋 Detailed Video Information")
                            
                            # Create scrollable container for video info
                            with st.container():
                                st.markdown("**Video Processing Results:**")
                                
                                # Show video info in a scrollable text area
                                import json
                                video_json_str = json.dumps(video_info, indent=2)
                                st.text_area(
                                    "Video Information:",
                                    value=video_json_str,
                                    height=200,
                                    help="Video processing results - scroll to see all data"
                                )
                                
                                # Also show as JSON for interactive viewing
                                st.markdown("**Interactive JSON View:**")
                                st.json(video_info)
                    else:
                        st.error("❌ No objects detected in the video")
                
                else:  # Folder
                    st.info("📁 Folder processing not implemented in this demo. Use the command-line demo.py script for batch processing.")
            
            except Exception as e:
                st.error(f"❌ Error processing: {str(e)}")
        else:
            st.warning("⚠️ Please upload a file or provide a valid folder path")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🎯 GroundingSam2-Custom | Powered by GroundingDINO + SAM 2</p>
        <p>For batch processing and advanced features, use the command-line interface: <code>python demo.py --help</code></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
