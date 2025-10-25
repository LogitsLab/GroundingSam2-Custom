# GroundingSam2-Custom

A clean, simplified repository for GroundingDINO + SAM 2 inference with local models. This repository provides easy-to-use image segmentation and video object tracking capabilities without requiring API tokens or external services.

## 🚀 Features

- **Image Segmentation**: Segment objects in images using natural language descriptions
- **Video Object Tracking**: Track objects across video frames with consistent IDs
- **Local Models Only**: No API keys or external dependencies required
- **Multiple Model Sizes**: Choose from tiny, small, base+, or large models based on your needs
- **Jupyter Notebooks**: Interactive examples for learning and experimentation
- **Easy Installation**: One-command setup with automatic model downloads

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

### Image Segmentation

```bash
python grounded_sam2_local_demo.py
```

This will:
- Load a sample image
- Detect objects using GroundingDINO
- Segment them using SAM 2
- Save results to `outputs/grounded_sam2_local_demo/`

### Video Object Tracking

```bash
python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py
```

This will:
- Process a sample video
- Track objects across frames
- Generate tracking results
- Save output video to `tracking_results/`

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

Available notebooks:
- `image_predictor_example.ipynb` - Image segmentation tutorial
- `video_predictor_example.ipynb` - Video tracking tutorial

## 📁 Directory Structure

```
GroundingSam2-Custom/
├── sam2/                          # SAM 2 core package
├── grounding_dino/                # GroundingDINO core package
├── utils/                         # Utility scripts
├── notebooks/                     # Jupyter notebooks and examples
│   ├── image_predictor_example.ipynb
│   ├── video_predictor_example.ipynb
│   ├── images/                    # Sample images
│   └── videos/bedroom/           # Sample video frames
├── assets/                       # Example assets
│   ├── hippopotamus.mp4
│   └── tracking_car_1.jpg
├── checkpoints/                  # SAM 2.1 model weights (downloaded by install.sh)
│   ├── sam2.1_hiera_tiny.pt      # 156 MB
│   ├── sam2.1_hiera_small.pt    # 184 MB
│   ├── sam2.1_hiera_base_plus.pt # 323 MB
│   └── sam2.1_hiera_large.pt     # 898 MB (best quality)
├── gdino_checkpoints/            # GroundingDINO model weights (downloaded by install.sh)
│   ├── groundingdino_swint_ogc.pth      # 693 MB
│   └── groundingdino_swinb_cogcoor.pth  # 938 MB (best quality)
├── outputs/                      # Generated outputs
├── tracking_results/            # Video tracking results
├── install.sh                   # Installation script
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
