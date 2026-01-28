# Gaussian Splatting Training Guide

This guide covers training Gaussian Splatting models, including how to work with large datasets and optimize for quality vs. speed.

## Quick Start

```bash
# Activate environment
conda activate gaussian-splatting

# Train on a dataset
python train.py -s <path_to_dataset>

# View the trained model
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/<model_name>
```

## Dataset Requirements

Gaussian Splatting expects COLMAP format:

```
<dataset>/
├── images/
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── sparse/
    └── 0/
        ├── cameras.txt (or .bin)
        ├── images.txt (or .bin)
        └── points3D.txt (or .bin)
```

## Training Options

### Resolution

By default, large images are automatically scaled down. Control this with `--resolution`:

| Flag | Behavior |
|------|----------|
| `--resolution -1` | Auto-scale (default) |
| `--resolution 1` | Full resolution |
| `--resolution 2` | Half resolution |
| `--resolution 4` | Quarter resolution |

**Example - Full resolution training:**
```bash
python train.py -s /path/to/data --resolution 1
```

### Memory Management

For large datasets or limited VRAM:

```bash
# Keep images on CPU (slower but uses less VRAM)
python train.py -s /path/to/data --data_device cpu

# Combine with lower resolution if still OOM
python train.py -s /path/to/data --data_device cpu --resolution 2
```

### Training Iterations

Default is 30,000 iterations. Adjust for quality vs. speed:

```bash
# Faster training (lower quality)
python train.py -s /path/to/data --iterations 15000

# Longer training (higher quality)
python train.py -s /path/to/data --iterations 50000
```

### Output Directory

```bash
# Custom output path
python train.py -s /path/to/data -m output/my_model_name
```

## Subsampling Large Datasets

For faster iteration or when working with very large datasets (1000+ images), you can create a subset.

### Using subsample_undistorted.py

This script creates a subset by keeping every Nth shot, preserving multi-lens camera groupings.

**Basic usage (keep 25% of shots):**
```bash
python subsample_undistorted.py \
    -s /path/to/undistorted \
    -o /path/to/undistorted_subset \
    -n 4
```

**Options:**

| Flag | Description |
|------|-------------|
| `-s, --source` | Source dataset path |
| `-o, --output` | Output path for subset |
| `-n, --keep_every` | Keep every Nth shot (default: 4) |
| `--offset` | Start offset for selection (default: 0) |

**Subsampling ratios:**

| `-n` value | Shots kept | Use case |
|------------|------------|----------|
| 2 | 50% | Moderate reduction |
| 4 | 25% | Good for testing |
| 6 | ~17% | Fast iteration |
| 10 | 10% | Quick previews |

**Example workflow:**
```bash
# Create 25% subset
python subsample_undistorted.py \
    -s /data/project/undistorted \
    -o /data/project/undistorted_25pct \
    -n 4

# Train on subset with full resolution
python train.py -s /data/project/undistorted_25pct --resolution 1 -m output/project_25pct

# If results look good, train on full dataset
python train.py -s /data/project/undistorted --resolution 1 -m output/project_full
```

## Multi-Lens Camera Data

When working with multi-lens cameras (e.g., drone with 2-3 cameras), each "shot" consists of multiple simultaneous images.

The `subsample_undistorted.py` script handles this automatically by:
1. Detecting shot numbers from filenames (e.g., `000921_0.png`, `000921_1.png`)
2. Keeping all lenses from selected shots together
3. Filtering COLMAP data to match

## Recommended Workflows

### Workflow 1: Quick Preview
```bash
# 10% of images, auto resolution
python subsample_undistorted.py -s /data/original -o /data/preview -n 10
python train.py -s /data/preview --iterations 10000
```

### Workflow 2: Quality Test
```bash
# 25% of images, full resolution
python subsample_undistorted.py -s /data/original -o /data/test -n 4
python train.py -s /data/test --resolution 1
```

### Workflow 3: Production Quality
```bash
# Full dataset, full resolution, longer training
python train.py -s /data/original --resolution 1 --iterations 50000
```

## Viewing Results

### SIBR Viewer
```bash
./SIBR_viewers/install/bin/SIBR_gaussianViewer_app -m output/<model_name>
```

**Controls:**
- WASD - Move camera
- Mouse - Look around
- Scroll - Adjust speed

### Render Images
```bash
python render.py -m output/<model_name>
```

Renders are saved to `output/<model_name>/train/` and `output/<model_name>/test/`.

### Compute Metrics
```bash
python metrics.py -m output/<model_name>
```

## Training Output Structure

```
output/<model_name>/
├── cfg_args                    # Training configuration
├── cameras.json                # Camera parameters
├── input.ply                   # Initial point cloud
├── point_cloud/
│   ├── iteration_7000/
│   │   └── point_cloud.ply     # Checkpoint
│   ├── iteration_30000/
│   │   └── point_cloud.ply     # Final model
│   └── ...
└── train/                      # Rendered training views (after render.py)
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Option 1: Lower resolution
python train.py -s /path/to/data --resolution 2

# Option 2: Keep images on CPU
python train.py -s /path/to/data --data_device cpu

# Option 3: Both
python train.py -s /path/to/data --data_device cpu --resolution 4
```

### Slow Training
- Reduce image count with `subsample_undistorted.py`
- Lower resolution with `--resolution 2` or higher
- Reduce iterations with `--iterations 15000`

### Poor Quality Results
- More images (less aggressive subsampling)
- Higher resolution `--resolution 1`
- More iterations `--iterations 50000`
- Check COLMAP reconstruction quality

### Missing Sparse Data After Subsampling
If the sparse folder is empty after subsampling, run COLMAP:
```bash
python convert.py -s /path/to/subset
```
