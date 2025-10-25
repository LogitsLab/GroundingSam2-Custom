#!/bin/bash

# GroundingSam2-Custom Installation Script
# This script installs all dependencies and downloads model weights
# Assumes you are already in a conda environment with Python 3.10

set -e  # Exit on any error

echo "🚀 Starting GroundingSam2-Custom Installation..."
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "sam2" ]; then
    print_error "Please run this script from the GroundingSam2-Custom root directory"
    exit 1
fi

# Check if conda environment is active
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_warning "No conda environment detected. Please activate your conda environment first:"
    print_warning "  conda activate your_env_name"
    print_warning "Continuing anyway..."
else
    print_status "Conda environment detected: $CONDA_DEFAULT_ENV"
fi

print_step "1. Installing Python Dependencies"

# Upgrade pip and install build tools
print_status "Upgrading pip and build tools..."
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
print_status "Installing PyTorch 2.5.1 with CUDA 12.1..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
print_status "Installing core dependencies..."
pip install \
    numpy>=1.24.4 \
    tqdm>=4.66.1 \
    hydra-core>=1.3.2 \
    iopath>=0.1.10 \
    pillow>=9.4.0 \
    opencv-python>=4.7.0 \
    matplotlib>=3.9.1 \
    jupyter>=1.0.0 \
    transformers \
    supervision>=0.22.0 \
    pycocotools \
    addict \
    yapf \
    timm \
    scikit-image \
    pycocoevalcap

print_step "2. Installing SAM-2 Package"

# Install SAM-2 in editable mode
print_status "Installing SAM-2 package..."
pip install -e .

print_step "3. Installing GroundingDINO Package"

# Fix GroundingDINO setup.py to remove unsupported CUDA architecture
print_status "Fixing GroundingDINO setup.py for CUDA compatibility..."
if [ -f "grounding_dino/setup.py" ]; then
    # Backup original setup.py
    cp grounding_dino/setup.py grounding_dino/setup.py.backup
    
    # Remove the problematic compute_120 architecture
    sed -i 's/-gencode=arch=compute_120,code=sm_120//g' grounding_dino/setup.py
    
    print_status "Installing GroundingDINO package..."
    cd grounding_dino
    pip install --no-build-isolation -e .
    cd ..
else
    print_error "GroundingDINO directory not found!"
    exit 1
fi

print_step "4. Creating Directories"

# Create necessary directories
mkdir -p checkpoints
mkdir -p gdino_checkpoints
mkdir -p outputs
mkdir -p tracking_results

print_step "5. Downloading Model Weights"

# Use wget or curl to download
if command -v wget &> /dev/null; then
    CMD="wget"
elif command -v curl &> /dev/null; then
    CMD="curl -L -O"
else
    print_error "Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Download SAM 2.1 checkpoints
print_status "Downloading SAM 2.1 model weights..."
cd checkpoints

# SAM 2.1 checkpoints URLs
SAM2p1_BASE_URL="https://dl.fbaipublicfiles.com/segment_anything_2/092824"
sam2p1_hiera_t_url="${SAM2p1_BASE_URL}/sam2.1_hiera_tiny.pt"
sam2p1_hiera_s_url="${SAM2p1_BASE_URL}/sam2.1_hiera_small.pt"
sam2p1_hiera_b_plus_url="${SAM2p1_BASE_URL}/sam2.1_hiera_base_plus.pt"
sam2p1_hiera_l_url="${SAM2p1_BASE_URL}/sam2.1_hiera_large.pt"

print_status "Downloading sam2.1_hiera_tiny.pt (156 MB)..."
$CMD $sam2p1_hiera_t_url || { print_error "Failed to download sam2.1_hiera_tiny.pt"; exit 1; }

print_status "Downloading sam2.1_hiera_small.pt (184 MB)..."
$CMD $sam2p1_hiera_s_url || { print_error "Failed to download sam2.1_hiera_small.pt"; exit 1; }

print_status "Downloading sam2.1_hiera_base_plus.pt (323 MB)..."
$CMD $sam2p1_hiera_b_plus_url || { print_error "Failed to download sam2.1_hiera_base_plus.pt"; exit 1; }

print_status "Downloading sam2.1_hiera_large.pt (898 MB)..."
$CMD $sam2p1_hiera_l_url || { print_error "Failed to download sam2.1_hiera_large.pt"; exit 1; }

cd ..

# Download GroundingDINO checkpoints
print_status "Downloading GroundingDINO model weights..."
cd gdino_checkpoints

BASE_URL="https://github.com/IDEA-Research/GroundingDINO/releases/download/"
swint_ogc_url="${BASE_URL}v0.1.0-alpha/groundingdino_swint_ogc.pth"
swinb_cogcoor_url="${BASE_URL}v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth"

print_status "Downloading groundingdino_swint_ogc.pth (693 MB)..."
$CMD $swint_ogc_url || { print_error "Failed to download groundingdino_swint_ogc.pth"; exit 1; }

print_status "Downloading groundingdino_swinb_cogcoor.pth (938 MB)..."
$CMD $swinb_cogcoor_url || { print_error "Failed to download groundingdino_swinb_cogcoor.pth"; exit 1; }

cd ..

print_step "6. Testing Installation"

# Test the installation
print_status "Testing SAM-2 import..."
python -c "import sam2; print('✅ SAM-2 imported successfully')"

print_status "Testing GroundingDINO import..."
python -c "import groundingdino; print('✅ GroundingDINO imported successfully')"

print_status "Testing PyTorch CUDA..."
python -c "import torch; print(f'✅ PyTorch CUDA available: {torch.cuda.is_available()}')"

print_step "7. Installation Summary"

echo ""
echo "🎉 GroundingSam2-Custom Installation Complete!"
echo "=============================================="
echo ""
echo "📦 Installed Components:"
echo "  ✅ PyTorch 2.5.1 with CUDA 12.1"
echo "  ✅ SAM-2 package (editable install)"
echo "  ✅ GroundingDINO package (editable install)"
echo "  ✅ All Python dependencies"
echo ""
echo "📁 Model Weights Downloaded:"
echo "  ✅ SAM 2.1: tiny (156MB), small (184MB), base+ (323MB), large (898MB)"
echo "  ✅ GroundingDINO: swint_ogc (693MB), swinb_cogcoor (938MB)"
echo "  📊 Total: ~3.1GB of model weights"
echo ""
echo "🚀 How to Use:"
echo "  1. Image segmentation: python grounded_sam2_local_demo.py"
echo "  2. Video tracking: python grounded_sam2_tracking_demo_custom_video_input_gd1.0_local_model.py"
echo "  3. Jupyter notebooks: jupyter notebook notebooks/"
echo ""
echo "📂 Key Files:"
echo "  • Best SAM 2 model: ./checkpoints/sam2.1_hiera_large.pt"
echo "  • Best GroundingDINO: ./gdino_checkpoints/groundingdino_swinb_cogcoor.pth"
echo "  • Outputs: ./outputs/"
echo "  • Example assets: ./assets/"
echo ""
echo "🔧 Troubleshooting:"
echo "  • If you get CUDA errors, models will fallback to CPU"
echo "  • Check outputs/ directory for generated results"
echo "  • Use smaller models for lower memory usage"
echo ""

print_status "Installation completed successfully! 🎉"
print_status "You can now run the demo scripts to test the installation."
