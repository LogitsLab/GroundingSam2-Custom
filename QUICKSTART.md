# 🚀 GroundingSam2-Custom Quick Start Guide

Get up and running with GroundingSam2-Custom in 5 minutes!

## ⚡ One-Command Installation

```bash
# 1. Clone the repository
git clone https://github.com/LogitsLab/GroundingSam2-Custom.git
cd GroundingSam2-Custom

# 2. Create conda environment
conda create -n groundingsam python=3.10 -y
conda activate groundingsam

# 3. Install everything (this downloads 3.1GB of models)
bash install.sh
```

That's it! The installation script handles everything automatically.

## 🎯 Test Your Installation

### Image Segmentation Demo

```bash
python grounded_sam2_local_demo.py
```

**Expected Output:**
- Processes a sample image
- Detects objects using GroundingDINO
- Segments them using SAM 2
- Saves results to `outputs/grounded_sam2_local_demo/`
- Creates annotated image with masks

### Video Tracking Demo

```bash
python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py
```

**Expected Output:**
- Processes a sample video
- Tracks objects across frames
- Generates tracking video
- Saves output to `tracking_results/`

## 📊 What You Get

After installation, you'll have:

- ✅ **4 SAM 2.1 Models** (156MB - 898MB each)
- ✅ **2 GroundingDINO Models** (693MB - 938MB each)
- ✅ **Total: 3.1GB** of pre-trained models
- ✅ **2 Demo Scripts** ready to run
- ✅ **2 Jupyter Notebooks** for learning
- ✅ **Sample Assets** for testing

## 🎛️ Model Options

### For Speed (Lower Quality)
- SAM 2: `sam2.1_hiera_tiny.pt` (156MB)
- GroundingDINO: `groundingdino_swint_ogc.pth` (693MB)

### For Quality (Slower)
- SAM 2: `sam2.1_hiera_large.pt` (898MB)
- GroundingDINO: `groundingdino_swinb_cogcoor.pth` (938MB)

## 🔧 Customization

### Use Your Own Images

Edit `grounded_sam2_local_demo.py`:

```python
# Change this line to your image
image_path = "path/to/your/image.jpg"

# Change the text prompt
text_prompt = "your description here"
```

### Use Your Own Videos

Edit `grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py`:

```python
# Change this line to your video
video_path = "path/to/your/video.mp4"

# Change the text prompt
text_prompt = "your description here"
```

## 📚 Learn More

### Jupyter Notebooks

```bash
jupyter notebook notebooks/
```

- `image_predictor_example.ipynb` - Image segmentation tutorial
- `video_predictor_example.ipynb` - Video tracking tutorial

### Directory Structure

```
GroundingSam2-Custom/
├── checkpoints/          # SAM 2.1 models (downloaded)
├── gdino_checkpoints/    # GroundingDINO models (downloaded)
├── outputs/             # Generated results
├── assets/              # Example files
├── notebooks/           # Learning tutorials
└── [demo scripts]       # Ready-to-run examples
```

## 🐛 Troubleshooting

### Installation Issues

```bash
# Re-run installation
bash install.sh

# Check conda environment
conda activate groundingsam
python --version  # Should be 3.10+
```

### Performance Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use smaller models for faster processing
# Edit demo scripts to use tiny/small models
```

### Memory Issues

- Use smaller models (tiny/small)
- Process images at lower resolution
- Close other applications

## ✅ Success Indicators

You know everything is working when:

1. ✅ Installation completes without errors
2. ✅ Image demo generates `outputs/grounded_sam2_local_demo/` folder
3. ✅ Video demo generates `tracking_results/` folder
4. ✅ No import errors when running demos
5. ✅ CUDA is available (if you have a GPU)

## 🎉 Next Steps

1. **Run the demos** to see the capabilities
2. **Try the notebooks** to learn the API
3. **Use your own images/videos** for custom projects
4. **Experiment with different models** for speed vs quality trade-offs

## 📞 Need Help?

- **Issues**: [GitHub Issues](https://github.com/LogitsLab/GroundingSam2-Custom/issues)
- **Documentation**: Check `README.md` for detailed information
- **Examples**: See `notebooks/` for interactive tutorials

---

**Happy coding! 🚀**
