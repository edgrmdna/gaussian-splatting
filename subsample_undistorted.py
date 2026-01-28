#!/usr/bin/env python3
"""
Subsample an undistorted dataset for Gaussian Splatting training.

This script creates a subset of images by keeping every Nth shot, which is useful
for faster training iterations or when working with limited VRAM.

For multi-lens cameras (e.g., 2 or 3 lenses per shot), this script keeps all
images from selected shots together, ensuring consistent coverage.

Usage:
    python subsample_undistorted.py -s <source_path> -o <output_path> -n 4

Example:
    # Keep every 4th shot (keep 1, skip 3) = 25% of original data
    python subsample_undistorted.py \
        -s /path/to/undistorted \
        -o /path/to/undistorted_subset \
        -n 4
"""

import os
import sys
import shutil
from pathlib import Path
from argparse import ArgumentParser


def get_shot_number(filename):
    """
    Extract shot number from filename.

    Handles formats like:
        - '000921_0.png' -> '000921'
        - '000921_1.png' -> '000921'
        - 'IMG_0001.jpg' -> 'IMG_0001'
    """
    basename = os.path.basename(filename)
    name_no_ext = os.path.splitext(basename)[0]

    # Handle format XXXXXX_Y where Y is lens number
    if '_' in name_no_ext:
        return name_no_ext.rsplit('_', 1)[0]
    return name_no_ext


def find_images(images_dir):
    """Find all image files in directory (recursively)."""
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(Path(images_dir).rglob(ext))
    return image_files


def group_by_shot(image_files):
    """Group image files by shot number."""
    shots = {}
    for img_path in image_files:
        shot_num = get_shot_number(img_path.name)
        if shot_num not in shots:
            shots[shot_num] = []
        shots[shot_num].append(img_path)
    return shots


def filter_images_txt(src_path, dst_path, selected_images):
    """
    Filter COLMAP images.txt to only include selected images.

    Format: Each image has 2 lines:
        IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        POINTS2D[] as (X, Y, POINT3D_ID)
    """
    kept = 0
    with open(src_path, 'r') as fin, open(dst_path, 'w') as fout:
        lines = fin.readlines()
        i = 0
        while i < len(lines):
            line = lines[i]

            # Preserve comments
            if line.startswith('#'):
                fout.write(line)
                i += 1
                continue

            # Skip empty lines
            if not line.strip():
                i += 1
                continue

            # Parse image line
            parts = line.strip().split()
            if len(parts) >= 10:
                img_name = parts[9]  # NAME is the 10th field

                # Check if this image is in our selection
                if img_name in selected_images:
                    fout.write(line)
                    # Write the 2D points line too
                    if i + 1 < len(lines):
                        fout.write(lines[i + 1])
                    kept += 1

            i += 2  # Each image has 2 lines

    return kept


def main():
    parser = ArgumentParser(
        description="Subsample undistorted dataset for Gaussian Splatting",
        epilog="Example: python subsample_undistorted.py -s /data/undistorted -o /data/undistorted_subset -n 4"
    )
    parser.add_argument(
        "-s", "--source",
        required=True,
        help="Source undistorted dataset path"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output path for subsampled dataset"
    )
    parser.add_argument(
        "-n", "--keep_every",
        type=int,
        default=4,
        help="Keep every Nth shot (default: 4 = keep 1, skip 3 = 25%% of data)"
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start offset for shot selection (default: 0)"
    )

    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)

    # Validate source
    if not source.exists():
        print(f"ERROR: Source path does not exist: {source}")
        sys.exit(1)

    images_dir = source / "images"
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)

    # Find and group images
    print(f"Scanning images in: {images_dir}")
    image_files = find_images(images_dir)

    if not image_files:
        print("ERROR: No images found")
        sys.exit(1)

    shots = group_by_shot(image_files)
    sorted_shots = sorted(shots.keys())

    # Select shots
    selected_shots = sorted_shots[args.offset::args.keep_every]

    print(f"\nDataset statistics:")
    print(f"  Total images: {len(image_files)}")
    print(f"  Total shots: {len(sorted_shots)}")
    print(f"  Images per shot: {len(image_files) // len(sorted_shots)}")
    print(f"\nSubsampling (keep every {args.keep_every}th shot):")
    print(f"  Selected shots: {len(selected_shots)}")

    # Build list of selected image paths (relative to images_dir)
    selected_images_rel = set()
    selected_images_names = set()
    for shot in selected_shots:
        for img_path in shots[shot]:
            rel_path = str(img_path.relative_to(images_dir))
            selected_images_rel.add(rel_path)
            selected_images_names.add(img_path.name)

    print(f"  Selected images: {len(selected_images_rel)}")

    # Create output directories
    output_images = output / "images"
    output_sparse = output / "sparse" / "0"
    output_images.mkdir(parents=True, exist_ok=True)
    output_sparse.mkdir(parents=True, exist_ok=True)

    # Copy selected images
    print(f"\nCopying images to: {output_images}")
    for img_rel in sorted(selected_images_rel):
        src_img = images_dir / img_rel
        dst_img = output_images / img_rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_img, dst_img)
    print(f"  Copied {len(selected_images_rel)} images")

    # Handle COLMAP sparse data
    sparse_dir = source / "sparse" / "0"
    if not sparse_dir.exists():
        sparse_dir = source / "sparse"

    if not sparse_dir.exists():
        print(f"\nWARNING: No sparse directory found at {source / 'sparse'}")
        print("You will need to run COLMAP on the subsampled images:")
        print(f"  python convert.py -s {output}")
        return

    print(f"\nProcessing COLMAP data from: {sparse_dir}")

    # Copy cameras (unchanged)
    for cameras_file in ['cameras.txt', 'cameras.bin']:
        src_cameras = sparse_dir / cameras_file
        if src_cameras.exists():
            shutil.copy2(src_cameras, output_sparse / cameras_file)
            print(f"  Copied {cameras_file}")
            break

    # Filter images.txt or images.bin
    images_txt = sparse_dir / "images.txt"
    images_bin = sparse_dir / "images.bin"

    if images_txt.exists():
        # Build set of image names as they appear in images.txt
        # (may include subdirectory paths)
        kept = filter_images_txt(
            images_txt,
            output_sparse / "images.txt",
            selected_images_rel | selected_images_names
        )
        print(f"  Filtered images.txt: kept {kept} images")
    elif images_bin.exists():
        print("  WARNING: images.bin found but filtering not implemented")
        print("  You may need to run COLMAP on the subsampled images")

    # Copy points3D (for reference, though it contains points from all images)
    for points_file in ['points3D.txt', 'points3D.bin', 'points3D.ply']:
        src_points = sparse_dir / points_file
        if src_points.exists():
            shutil.copy2(src_points, output_sparse / points_file)
            print(f"  Copied {points_file}")

    # Summary
    print(f"\n{'='*60}")
    print("Subsampled dataset created successfully!")
    print(f"{'='*60}")
    print(f"\nOutput: {output}")
    print(f"Images: {len(selected_images_rel)} ({100/args.keep_every:.0f}% of original)")
    print(f"\nTo train with full resolution:")
    print(f"  python train.py -s {output} --resolution 1")
    print(f"\nTo train with auto-scaled resolution (less VRAM):")
    print(f"  python train.py -s {output}")


if __name__ == "__main__":
    main()
