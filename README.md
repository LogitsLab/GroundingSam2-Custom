# GroundingSam2-Custom

A comprehensive repository for GroundingDINO + SAM 2 inference with local models. Features image segmentation, video object tracking, unified demo interface, and interactive Streamlit web app.

## 🚀 Features

- **Image Segmentation**: Segment objects in images using natural language descriptions
- **Video Object Tracking**: Track objects across video frames with consistent IDs
- **Unified Demo Interface**: Single script for images, videos, and batch processing
- **Interactive Web App**: Streamlit interface with model selection and visualization
- **Model Download Script**: Download-only script for model weights
- **Local Models Only**: No API keys or external dependencies required
- **Multiple Model Sizes**: Choose from tiny, small, base+, or large models
- **Command-line Interface**: Flexible parameter control for all features

## 📋 Prerequisites

- Python 3.10+ (recommended)
- Conda environment (recommended)
- CUDA-capable GPU (optional, will fallback to CPU)
- ~3.1GB disk space for model weights

## 🛠️ Installation

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/LogitsLab/GroundingSam2-Custom.git
   cd GroundingSam2-Custom
   ```

2. **Create and activate conda environment**:
   ```bash
   conda create -n groundingsam python=3.10 -y
   conda activate groundingsam
   ```

3. **Run installation script**:
   ```bash
   bash install.sh
   ```

That's it! The script will automatically:
- Install all Python dependencies
- Download all model weights (3.1GB)
- Set up the environment
- Test the installation

## 🎯 Usage

### 1. Unified Demo Interface (Recommended)

```bash
# Image segmentation
python demo.py --input image.jpg --text "car. person."

# Video tracking
python demo.py --input video.mp4 --text "hippopotamus." --mode video

# Batch processing folder
python demo.py --input /path/to/folder --text "dog. cat." --mode batch

# Custom models and settings
python demo.py --input image.jpg --text "car." \
  --sam2-model large --gdino-model swinb \
  --box-threshold 0.3 --output-dir my_results
```

### 2. Interactive Web App

```bash
# Launch Streamlit app
bash run_streamlit.sh
# OR
streamlit run streamlit_app.py
```

Features:
- Upload images/videos or select folders
- Model selection with performance indicators
- Adjustable detection thresholds (default 0.25)
- 17 predefined object classes + custom classes
- Side-by-side input/output visualization
- JSON results display

### 3. Individual Demo Scripts

```bash
# Image segmentation
python grounded_sam2_local_demo.py --sam2-model large --gdino-model swinb

# Video tracking
python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py \
  --sam2-model large --gdino-model swinb --text "person. car."
```

### 4. Model Download Only

```bash
# Download models without installing dependencies
bash download_models.sh
```

## 📁 Directory Structure

```
GroundingSam2-Custom/
├── sam2/                          # SAM 2 core package
├── grounding_dino/                # GroundingDINO core package
├── utils/                         # Utility scripts
├── checkpoints/                  # SAM 2.1 model weights
│   ├── sam2.1_hiera_tiny.pt      # 156 MB
│   ├── sam2.1_hiera_small.pt     # 184 MB
│   ├── sam2.1_hiera_base_plus.pt # 323 MB
│   └── sam2.1_hiera_large.pt     # 898 MB (best quality)
├── gdino_checkpoints/            # GroundingDINO model weights
│   ├── groundingdino_swint_ogc.pth      # 693 MB
│   └── groundingdino_swinb_cogcoor.pth  # 938 MB (best quality)
├── outputs/                      # Generated outputs
├── install.sh                   # Complete installation script
├── download_models.sh            # Model download only
├── demo.py                       # Unified demo interface
├── streamlit_app.py             # Interactive web app
├── run_streamlit.sh             # Streamlit launcher
├── grounded_sam2_local_demo.py  # Image segmentation demo
└── grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py  # Video tracking demo
```

## 🎛️ Model Information

### SAM 2.1 Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `sam2.1_hiera_tiny.pt` | 156 MB | Fastest | Good | Quick prototyping |
| `sam2.1_hiera_small.pt` | 184 MB | Fast | Better | Balanced performance |
| `sam2.1_hiera_base_plus.pt` | 323 MB | Medium | High | Production use |
| `sam2.1_hiera_large.pt` | 898 MB | Slower | Best | Highest quality |

### GroundingDINO Models

| Model | Size | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `groundingdino_swint_ogc.pth` | 693 MB | Faster | Good | Quick detection |
| `groundingdino_swinb_cogcoor.pth` | 938 MB | Slower | Best | Highest accuracy |

## 🔧 Customization

### Using Different Models

Edit the demo scripts to use different model sizes:

```python
# For faster processing (lower quality)
sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_t.yaml"

# For best quality (slower)
sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
```

### Custom Images/Videos

Replace the sample files in the demo scripts with your own:
- Images: Update `image_path` variable
- Videos: Update `video_path` variable

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Use smaller models (tiny/small)
   - Reduce batch size
   - Process images at lower resolution

2. **Import Errors**
   - Ensure conda environment is activated: `conda activate groundingsam`
   - Reinstall packages: `pip install -e .` and `pip install -e grounding_dino`

3. **Model Download Failures**
   - Check internet connection
   - Re-run `bash install.sh`
   - Manually download models to `checkpoints/` and `gdino_checkpoints/`

4. **Slow Performance**
   - Use GPU if available: `torch.cuda.is_available()`
   - Use smaller models for faster processing
   - Reduce image/video resolution

### Verification

Test your installation:

```bash
python -c "import sam2; print('SAM-2 OK')"
python -c "import groundingdino; print('GroundingDINO OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 📄 License

This project combines multiple open-source components:

- **SAM 2**: Licensed under Apache 2.0 (see `LICENSE_sam2`)
- **GroundingDINO**: Licensed under Apache 2.0 (see `grounding_dino/LICENSE`)
- **This Repository**: Licensed under MIT (see `LICENSE`)

## 🙏 Credits

- **SAM 2**: [Meta AI](https://github.com/facebookresearch/segment-anything-2)
- **GroundingDINO**: [IDEA Research](https://github.com/IDEA-Research/GroundingDINO)
- **Original Grounded-SAM-2**: [LogitsLab](https://github.com/LogitsLab/Grounded-SAM-2)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/LogitsLab/GroundingSam2-Custom/issues)
- **Documentation**: Check the `notebooks/` directory for tutorials
- **Examples**: See `assets/` for sample files

## 🚀 Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/LogitsLab/GroundingSam2-Custom.git
cd GroundingSam2-Custom
conda create -n groundingsam python=3.10 -y
conda activate groundingsam

# 2. Install everything
bash install.sh

# 3. Run demos
python grounded_sam2_local_demo.py
python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py

# 4. Explore notebooks
jupyter notebook notebooks/
```

That's it! You're ready to start using GroundingSam2-Custom! 🎉
