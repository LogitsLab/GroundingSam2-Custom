"""
Core processing functions for image segmentation and video tracking.

This module contains the main processing logic for both image segmentation
and video object tracking using GroundingDINO and SAM 2.
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
from grounding_dino.groundingdino.util.inference import load_image, predict
from utils.track_utils import sample_points_from_masks
from utils.video_utils import create_video_from_images

def run_detection(grounding_model, image_path, text_prompt, box_threshold, text_threshold, device):
    """Run GroundingDINO detection on an image."""
    image_source, image = load_image(image_path)
    boxes, confidences, labels = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device
    )
    return image_source, image, boxes, confidences, labels

def process_detection_results(image_source, boxes, confidences, labels):
    """Process detection results and convert to SAM 2 format."""
    if len(boxes) == 0:
        return None, None, None, None, None
    
    # Process the box prompt for SAM 2
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    
    confidences = confidences.numpy().tolist()
    class_names = labels
    class_ids = np.array(list(range(len(class_names))))
    
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence in zip(class_names, confidences)
    ]
    
    return image_source, input_boxes, class_names, class_ids, labels

def run_sam2_segmentation(sam2_predictor, image_source, input_boxes, multimask=False):
    """Run SAM 2 segmentation on detected boxes."""
    sam2_predictor.set_image(image_source)
    
    # Run SAM 2 segmentation
    masks, scores, logits = sam2_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=multimask,
    )
    
    # Process results
    if multimask:
        best = np.argmax(scores, axis=1)                     
        masks = masks[np.arange(masks.shape[0]), best]       
    
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    
    return masks, scores, logits

def create_annotations(image, input_boxes, masks, class_ids, labels, show_boxes=True):
    """Create annotated visualization with boxes, labels, and masks."""
    detections = sv.Detections(
        xyxy=input_boxes,
        mask=masks.astype(bool),
        class_id=class_ids
    )
    
    # Create annotated image
    annotated_frame = image.copy()
    
    if show_boxes:
        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
    
    label_annotator = sv.LabelAnnotator()
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    mask_annotator = sv.MaskAnnotator()
    annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
    
    return annotated_frame

def create_json_results(image_path, class_names, input_boxes, masks, scores, image_source):
    """Create JSON results in standard format."""
    def single_mask_to_rle(mask):
        if mask is None:
            return None
        try:
            # Ensure mask is 2D and convert to uint8
            if len(mask.shape) == 3:
                mask = mask.squeeze()
            mask = mask.astype(np.uint8)
            rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            return rle
        except Exception as e:
            print(f"Warning: Could not encode mask: {e}")
            return None
    
    # Handle masks properly
    mask_rles = []
    for mask in masks:
        if mask is not None:
            mask_rle = single_mask_to_rle(mask)
            mask_rles.append(mask_rle)
        else:
            mask_rles.append(None)
    h, w, _ = image_source.shape
    
    results = {
        "image_path": str(image_path),
        "annotations": [
            {
                "class_name": class_name,
                "bbox": box.tolist() if hasattr(box, 'tolist') else list(box),
                "segmentation": mask_rle,
                "score": float(score),
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": w,
        "img_height": h,
    }
    
    return results

def process_image(grounding_model, sam2_predictor, image_path, text_prompt, 
                 box_threshold, text_threshold, output_dir, dump_json=True, 
                 multimask=False, show_boxes=True, device="cuda"):
    """Process a single image for segmentation."""
    print(f"🎯 Processing image: {image_path}")
    
    # Run detection
    image_source, image, boxes, confidences, labels = run_detection(
        grounding_model, image_path, text_prompt, box_threshold, text_threshold, device
    )
    
    # Process detection results
    result = process_detection_results(image_source, boxes, confidences, labels)
    if result[0] is None:
        print(f"   No objects detected in {image_path}")
        return None
    
    image_source, input_boxes, class_names, class_ids, labels = result
    
    # Setup mixed precision
    from core.models import setup_mixed_precision
    setup_mixed_precision(device)
    
    # Run SAM 2 segmentation
    masks, scores, logits = run_sam2_segmentation(sam2_predictor, image_source, input_boxes, multimask)
    
    # Create output filename and organized directory structure
    output_name = Path(image_path).stem
    base_output_dir = Path(output_dir)
    
    # Create organized subdirectories
    images_dir = base_output_dir / "images"
    json_dir = base_output_dir / "json_results"
    logs_dir = base_output_dir / "logs"
    
    # Create all directories
    images_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Use images directory for segmented images
    output_path = images_dir
    
    # Load original image for visualization
    img = cv2.imread(str(image_path))
    
    # Create annotations
    annotated_frame = create_annotations(img, input_boxes, masks, class_ids, labels, show_boxes)
    
    # Save results
    cv2.imwrite(str(output_path / f"{output_name}_segmented.jpg"), annotated_frame)
    
    # Save JSON results if requested
    if dump_json:
        json_results = create_json_results(image_path, class_names, input_boxes, masks, scores, image_source)
        json_path = json_dir / f"{output_name}_results.json"
        with open(json_path, "w") as f:
            json.dump(json_results, f, indent=4)
    
    print(f"   ✅ Saved: {output_path / f'{output_name}_segmented.jpg'}")
    return output_path / f"{output_name}_segmented.jpg"

def process_video(grounding_model, sam2_predictor, video_path, text_prompt, 
                 box_threshold, text_threshold, output_dir, output_video_name,
                 prompt_type="box", show_boxes=True, device="cuda"):
    """Process a video for object tracking."""
    print(f"🎬 Processing video: {video_path}")
    
    # Setup video processing
    video_info = sv.VideoInfo.from_video_path(video_path)
    frame_generator = sv.get_video_frames_generator(video_path, stride=1, start=0, end=None)
    
    # Extract frames
    frames_dir = Path(output_dir) / "temp_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    print("   Extracting video frames...")
    with sv.ImageSink(
        target_dir_path=frames_dir, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Extracting frames"):
            sink.save_image(frame)
    
    # Get frame names
    frame_names = sorted([f for f in frames_dir.glob("*.jpg")])
    
    # Initialize video predictor
    from core.models import load_sam2_video
    video_predictor, sam2_config = load_sam2_video("large", device)  # Use large for video
    inference_state = video_predictor.init_state(video_path=str(frames_dir))
    
    # Process first frame for detection
    ann_frame_idx = 0
    img_path = frame_names[ann_frame_idx]
    image_source, image, boxes, confidences, labels = run_detection(
        grounding_model, str(img_path), text_prompt, box_threshold, text_threshold, device
    )
    
    # Process detection results
    result = process_detection_results(image_source, boxes, confidences, labels)
    if result[0] is None:
        print(f"   No objects detected in video")
        return None
    
    image_source, input_boxes, class_names, class_ids, labels = result
    OBJECTS = class_names
    
    # Setup mixed precision
    from core.models import setup_mixed_precision
    setup_mixed_precision(device)
    
    # Get masks for first frame
    sam2_predictor.set_image(image_source)
    masks, scores, logits = run_sam2_segmentation(sam2_predictor, image_source, input_boxes, False)
    
    # Register objects for tracking
    print("   Registering objects for tracking...")
    if prompt_type == "point":
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)
        for object_id, (label, points) in enumerate(zip(OBJECTS, all_sample_points), start=1):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    elif prompt_type == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    elif prompt_type == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask
            )
    
    # Run tracking
    print("   Running video tracking...")
    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    
    # Generate tracking visualizations
    print("   Generating tracking visualizations...")
    base_output_dir = Path(output_dir)
    
    # Create organized subdirectories for video processing
    videos_dir = base_output_dir / "videos"
    tracking_frames_dir = base_output_dir / "tracking_frames"
    json_dir = base_output_dir / "json_results"
    logs_dir = base_output_dir / "logs"
    
    # Create all directories
    videos_dir.mkdir(parents=True, exist_ok=True)
    tracking_frames_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = tracking_frames_dir
    
    ID_TO_OBJECTS = {i: obj for i, obj in enumerate(OBJECTS, start=1)}
    
    for frame_idx, segments in tqdm(video_segments.items(), desc="Processing frames"):
        img = cv2.imread(str(frame_names[frame_idx]))
        
        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)
        
        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),
            mask=masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )
        
        annotated_frame = img.copy()
        
        if show_boxes:
            box_annotator = sv.BoxAnnotator()
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        
        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(
            annotated_frame, 
            detections=detections, 
            labels=[ID_TO_OBJECTS[i] for i in object_ids]
        )
        
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        cv2.imwrite(str(results_dir / f"annotated_frame_{frame_idx:05d}.jpg"), annotated_frame)
    
    # Create output video
    print("   Creating output video...")
    output_video_path = videos_dir / output_video_name
    create_video_from_images(str(results_dir), str(output_video_path))
    
    # Clean up temporary frames
    import shutil
    shutil.rmtree(frames_dir)
    
    print(f"   ✅ Saved: {output_video_path}")
    return output_video_path
