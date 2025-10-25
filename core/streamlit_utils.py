"""
Streamlit-specific utilities for GroundingSam2-Custom.

This module provides utilities for the Streamlit web interface,
including model configurations, UI components, and processing functions.
"""

import streamlit as st
import tempfile
import os
import json
import cv2
import numpy as np
from pathlib import Path
from core.models import SAM2_MODELS, GDINO_MODELS, load_models, setup_mixed_precision
from core.processing import process_image, process_video, run_detection, process_detection_results, run_sam2_segmentation, create_annotations, create_json_results

# Predefined classes for Streamlit interface
PREDEFINED_CLASSES = [
    "Sky", "Person", "Ground", "Vegetation", "Building", "Road Surface",
    "Wire", "Snow", "Trash Can", "Railroad", "Walkways", "Street Light",
    "Sign", "Traffic Light", "Fire Hydrant", "Pole", "Fence"
]

@st.cache_resource
def load_models_cached(sam2_model_key, gdino_model_key):
    """Load and cache the models for Streamlit."""
    return load_models(sam2_model_key, gdino_model_key)

def process_image_streamlit(grounding_model, sam2_predictor, image_path, text_prompt, 
                          box_threshold, text_threshold, show_boxes, device, output_dir="./output"):
    """Process a single image for segmentation in Streamlit."""
    # Run detection
    image_source, processed_image, boxes, confidences, labels = run_detection(
        grounding_model, image_path, text_prompt, box_threshold, text_threshold, device
    )
    
    if len(boxes) == 0:
        return None, "No objects detected"
    
    # Process detection results
    result = process_detection_results(image_source, boxes, confidences, labels)
    if result[0] is None:
        return None, "No objects detected"
    
    image_source, input_boxes, class_names, class_ids, labels = result
    
    # Setup mixed precision
    setup_mixed_precision(device)
    
    # Run SAM 2 segmentation
    masks, scores, logits = run_sam2_segmentation(sam2_predictor, image_source, input_boxes, False)
    
    # Load original image for visualization
    img = cv2.imread(image_path)
    
    # Create annotations
    annotated_frame = create_annotations(img, input_boxes, masks, class_ids, labels, show_boxes)
    
    # Create organized output directory structure
    base_output_dir = Path(output_dir)
    images_dir = base_output_dir / "images"
    json_dir = base_output_dir / "json_results"
    logs_dir = base_output_dir / "logs"
    
    # Create all directories
    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save segmented image
    output_name = Path(image_path).stem
    segmented_path = images_dir / f"{output_name}_segmented.jpg"
    cv2.imwrite(str(segmented_path), annotated_frame)
    
    # Create and save JSON results
    json_results = create_json_results(image_path, class_names, input_boxes, masks, scores, image_source)
    json_path = json_dir / f"{output_name}_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=4)
    
    return annotated_frame, json_results

def process_video_streamlit(grounding_model, sam2_predictor, video_path, text_prompt,
                           box_threshold, text_threshold, show_boxes, device, output_dir="./output"):
    """Process a video for object tracking in Streamlit."""
    # Create organized output directory structure
    base_output_dir = Path(output_dir)
    videos_dir = base_output_dir / "videos"
    tracking_frames_dir = base_output_dir / "tracking_frames"
    json_dir = base_output_dir / "json_results"
    logs_dir = base_output_dir / "logs"
    
    # Create all directories
    videos_dir.mkdir(parents=True, exist_ok=True)
    tracking_frames_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Process video using core function
    output_video_name = f"{Path(video_path).stem}_tracking.mp4"
    result_video = process_video(
        grounding_model, sam2_predictor, video_path, text_prompt,
        box_threshold, text_threshold, str(base_output_dir), output_video_name,
        "box", show_boxes, device
    )
    
    if result_video and result_video.exists():
        return str(result_video), {"status": "success", "output_dir": str(base_output_dir)}
    else:
        return None, "No objects detected in video"

def get_model_selection_ui():
    """Get model selection UI components for Streamlit."""
    st.subheader("🤖 Model Selection")
    
    # SAM 2 model selection
    sam2_model = st.selectbox(
        "SAM 2 Model:",
        list(SAM2_MODELS.keys()),
        index=3,  # Default to large
        format_func=lambda x: f"{SAM2_MODELS[x]['name']} ({SAM2_MODELS[x]['size']})"
    )
    
    # Display model info
    sam2_info = SAM2_MODELS[sam2_model]
    st.markdown(f"""
    <div class="model-info">
        <strong>{sam2_info['name']}</strong><br>
        Size: {sam2_info['size']}<br>
        Speed: {sam2_info['speed']}<br>
        Quality: {sam2_info['quality']}
    </div>
    """, unsafe_allow_html=True)
    
    # GroundingDINO model selection
    gdino_model = st.selectbox(
        "GroundingDINO Model:",
        list(GDINO_MODELS.keys()),
        index=1,  # Default to swinb
        format_func=lambda x: f"{GDINO_MODELS[x]['name']} ({GDINO_MODELS[x]['size']})"
    )
    
    # Display model info
    gdino_info = GDINO_MODELS[gdino_model]
    st.markdown(f"""
    <div class="model-info">
        <strong>{gdino_info['name']}</strong><br>
        Size: {gdino_info['size']}<br>
        Speed: {gdino_info['speed']}<br>
        Quality: {gdino_info['quality']}
    </div>
    """, unsafe_allow_html=True)
    
    return sam2_model, gdino_model

def get_threshold_ui():
    """Get threshold selection UI components for Streamlit."""
    st.subheader("🎚️ Detection Thresholds")
    
    box_threshold = st.slider(
        "Box Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for bounding box detection"
    )
    
    text_threshold = st.slider(
        "Text Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence for text matching"
    )
    
    return box_threshold, text_threshold

def get_display_options_ui():
    """Get display options UI components for Streamlit."""
    # Note: st.subheader is called in the main app, not here
    
    show_boxes = st.checkbox(
        "Show Bounding Boxes",
        value=True,
        help="Display bounding boxes around detected objects"
    )
    
    show_json = st.checkbox(
        "Show JSON Results",
        value=False,
        help="Display detailed JSON results"
    )
    
    return show_boxes, show_json

def get_class_selection_ui():
    """Get class selection UI components for Streamlit."""
    # Note: st.subheader is called in the main app, not here
    
    # Choose between predefined classes or custom text
    selection_method = st.radio(
        "Choose input method:",
        ["Use Predefined Classes", "Enter Custom Text"],
        help="Select how you want to specify object classes"
    )
    
    if selection_method == "Use Predefined Classes":
        st.markdown("**Select object classes:**")
        selected_classes = []
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            for i, class_name in enumerate(PREDEFINED_CLASSES[:9]):
                if st.checkbox(class_name, value=False, key=f"class_{i}"):
                    selected_classes.append(class_name)
        
        with col2:
            for i, class_name in enumerate(PREDEFINED_CLASSES[9:]):
                if st.checkbox(class_name, value=False, key=f"class_{i+9}"):
                    selected_classes.append(class_name)
        
        # Additional custom classes
        st.markdown("**Additional Custom Classes:**")
        custom_classes = st.text_area(
            "Enter additional classes (one per line):",
            height=80,
            help="Add custom object classes, one per line"
        )
        
        if custom_classes:
            custom_list = [cls.strip() for cls in custom_classes.split('\n') if cls.strip()]
            selected_classes.extend(custom_list)
        
        # Generate final prompt
        if selected_classes:
            text_prompt = ". ".join(selected_classes) + "."
            st.markdown(f"**Generated Prompt:** `{text_prompt}`")
        else:
            text_prompt = "person. car. dog."  # Default fallback
            st.markdown(f"**Generated Prompt:** `{text_prompt}` *(default - no classes selected)*")
    
    else:  # Enter Custom Text
        st.markdown("**Enter your custom text prompt:**")
        text_prompt = st.text_area(
            "Text Prompt:",
            value="person. car. dog.",
            height=100,
            help="Enter the text prompt for object detection (e.g., 'person. car. truck.')"
        )
        
        if not text_prompt.strip():
            text_prompt = "person. car. dog."  # Default fallback
            st.markdown(f"**Generated Prompt:** `{text_prompt}` *(default - empty input)*")
        else:
            st.markdown(f"**Generated Prompt:** `{text_prompt}`")
    
    return text_prompt

def get_custom_css():
    """Get custom CSS for Streamlit interface."""
    return """
    <style>
        /* Main header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }
        
        /* Model info cards */
        .model-info {
            background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%);
            padding: 1rem;
            border-radius: 0.75rem;
            margin: 1rem 0;
            color: #2c3e50;
            font-weight: 500;
            border: 1px solid #e1e5e9;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .model-info strong {
            color: #1f77b4;
            font-size: 1.1em;
        }
        
        /* Threshold sliders */
        .threshold-slider {
            margin: 1rem 0;
        }
        
        /* Class checklist */
        .class-checklist {
            max-height: 300px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: #ffffff;
        }
        
        /* Result containers */
        .result-container {
            border: 2px solid #1f77b4;
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin: 1rem 0;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        
        /* JSON container */
        .json-container {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 0.5rem;
            padding: 1rem;
            max-height: 400px;
            overflow-y: auto;
        }
        
        /* Scrollable JSON results */
        .scrollable-json {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #e1e5e9;
            border-radius: 0.5rem;
            padding: 1rem;
            background-color: #ffffff;
        }
        
        /* Individual object details */
        .object-detail {
            border: 1px solid #e1e5e9;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            background-color: #f8f9fa;
        }
        
        /* Text area styling for JSON */
        .stTextArea > div > div > textarea {
            font-family: 'Courier New', monospace;
            font-size: 0.9rem;
            line-height: 1.4;
        }
        
        /* Process button styling */
        .stButton > button {
            background: linear-gradient(135deg, #1f77b4 0%, #0d5a8a 100%);
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 2rem;
            font-size: 1.1rem;
            font-weight: bold;
            box-shadow: 0 4px 8px rgba(31, 119, 180, 0.3);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(31, 119, 180, 0.4);
        }
        
        /* Image comparison styling */
        .image-comparison {
            border: 2px solid #e1e5e9;
            border-radius: 0.75rem;
            padding: 1rem;
            margin: 1rem 0;
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        }
        
        /* Results information styling */
        .results-info {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            border-radius: 0.75rem;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid #dee2e6;
        }
        
        /* Sidebar section headers */
        .sidebar-section {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid #e1e5e9;
        }
        
        /* File upload area */
        .upload-area {
            border: 2px dashed #1f77b4;
            border-radius: 0.75rem;
            padding: 2rem;
            text-align: center;
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            margin: 1rem 0;
        }
        
        /* Status messages */
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .status-warning {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
    </style>
    """
