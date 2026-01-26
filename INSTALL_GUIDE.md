# Gaussian Splatting Installation Guide

## Prerequisites

- CUDA-ready GPU with Compute Capability 7.0+ (24 GB VRAM for full quality training)
- CUDA Toolkit installed (check with `nvcc --version`)
- Conda for environment management
- C++ compiler (g++ on Linux)

## Installation Steps

### 1. Clone the repository with submodules

```bash
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
```

### 2. Create conda environment

```bash
conda create -n gaussian-splatting python=3.10
conda activate gaussian-splatting
```

### 3. Install PyTorch with CUDA support

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Verify installation:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If you get "file too short" errors, do a clean reinstall:
```bash
pip uninstall torch torchvision torchaudio nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 nvidia-cudnn-cu12 nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cusolver-cu12 nvidia-cusparse-cu12 nvidia-nccl-cu12 nvidia-nvtx-cu12 nvidia-nvjitlink-cu12 -y
pip cache purge
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install submodules

```bash
pip install --no-build-isolation submodules/diff-gaussian-rasterization
pip install --no-build-isolation submodules/simple-knn
pip install --no-build-isolation submodules/fused-ssim
```

### 5. Install additional dependencies

```bash
pip install plyfile tqdm
```

### 6. (Optional) Build SIBR Viewers

Install system dependencies:
```bash
sudo apt install -y libglew-dev libassimp-dev libboost-all-dev libgtk-3-dev libopencv-dev libglfw3-dev libavdevice-dev libavcodec-dev libeigen3-dev libxxf86vm-dev libembree-dev
```

If nvcc is not in PATH, set it:
```bash
export CUDACXX=/usr/local/cuda/bin/nvcc
```

Build the viewers:
```bash
cd SIBR_viewers
cmake -Bbuild . -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24 --target install
cd ..
```

## Usage

### Training

```bash
python train.py -s <path to COLMAP or NeRF Synthetic dataset>
```

### Viewing trained models

```bash
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m <path to trained model>
```

### Converting your own images

Install COLMAP:
```bash
sudo apt install colmap
```

Place images in `<your_folder>/input/` and run:
```bash
python convert.py -s <your_folder>
```

## Expected Dataset Format

Gaussian Splatting expects COLMAP format:

```
<scene>/
├── images/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.bin    # camera intrinsics
        ├── images.bin     # camera extrinsics per image
        └── points3D.bin   # sparse 3D point cloud
```

## Sample Data

Download sample dataset:
```bash
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip
python train.py -s tandt_db/tandt/truck
```

## Troubleshooting

- **CUDA version mismatch**: Install PyTorch matching your system CUDA version
- **"file too short" errors**: Corrupted download, do clean reinstall (see step 3)
- **No module named 'torch' during submodule install**: Use `--no-build-isolation` flag
- **nvcc not found**: Set `export CUDACXX=/usr/local/cuda/bin/nvcc` or install `nvidia-cuda-toolkit`
- **CUDA out of memory during training**: By default, all training images are loaded onto the GPU. For large datasets or GPUs with limited VRAM, keep images on CPU instead:
  ```bash
  python train.py -s <path> --data_device cpu
  ```
- **Process killed (OOM) with --data_device cpu**: System RAM exhausted. Combine CPU storage with lower resolution:
  ```bash
  python train.py -s <path> --data_device cpu --resolution 2
  ```
  Resolution values are divisors: `2` = half size, `4` = quarter size, `8` = eighth size

## Mesh Extraction with SuGaR

SuGaR (Surface-Aligned Gaussian Splatting) enables extracting traditional polygon meshes from trained Gaussian Splatting models. This is useful for:
- Exporting to 3D modeling software (Blender, Maya, etc.)
- 3D printing
- Integration with game engines
- Generating UV-mapped textured meshes

### Installing SuGaR Environment

SuGaR requires a different PyTorch/CUDA version than the base Gaussian Splatting environment, so it uses a separate conda environment.

1. Initialize the SuGaR submodule (if not already done):
   ```bash
   git submodule update --init --recursive
   ```

2. Create the SuGaR conda environment:
   ```bash
   conda env create -f environment_sugar.yml
   ```

3. Activate and verify:
   ```bash
   conda activate sugar
   python -c "import torch; print(torch.cuda.is_available())"
   ```

### Extracting Meshes

Basic usage:
```bash
conda activate sugar
python extract_mesh.py -m output/<model_name>
```

With options:
```bash
python extract_mesh.py -m output/<model_name> \
    -o output/<model_name>/mesh \
    --refinement_time medium \
    --export_obj
```

### Extract Mesh Options

| Option | Description |
|--------|-------------|
| `-m, --model_path` | Path to trained model (required) |
| `-s, --source_path` | Path to source COLMAP data (auto-detected) |
| `-o, --output_path` | Output directory (default: `<model>/mesh`) |
| `--iteration` | Checkpoint iteration (default: latest) |
| `--refinement_time` | `short` (~15min), `medium` (~1hr), `long` (~2hr) |
| `--export_obj` | Export textured OBJ in addition to PLY |
| `--low_poly` | Lower polygon count (faster, less detail) |
| `--high_poly` | Higher polygon count (slower, more detail) |

### Output Formats

- **PLY**: Point cloud / mesh format, viewable in MeshLab
- **OBJ**: Textured mesh with UV coordinates, compatible with most 3D software

### Example Workflow

```bash
# 1. Train a model (in gaussian-splatting environment)
conda activate gaussian-splatting
python train.py -s data/my_scene

# 2. Extract mesh (in sugar environment)
conda activate sugar
python extract_mesh.py -m output/my_scene --export_obj

# 3. View in MeshLab or import to Blender
```

### Mesh Extraction Troubleshooting

- **Out of memory**: Try `--low_poly` or use `--refinement_time short`
- **Missing source path**: Specify explicitly with `-s /path/to/colmap/data`
- **PyTorch3D errors**: Ensure you're using Python 3.9 and PyTorch 2.0.1 in the sugar environment
