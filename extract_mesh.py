#!/usr/bin/env python3
#
# Mesh extraction wrapper for SuGaR (Surface-Aligned Gaussian Splatting)
# Requires separate conda environment: conda env create -f environment_sugar.yml
#

import os
import sys
import glob
import subprocess
from argparse import ArgumentParser


def find_latest_iteration(model_path):
    """Find the latest checkpoint iteration in a model directory."""
    point_cloud_dir = os.path.join(model_path, "point_cloud")
    if not os.path.exists(point_cloud_dir):
        return None

    iterations = []
    for folder in os.listdir(point_cloud_dir):
        if folder.startswith("iteration_"):
            try:
                iterations.append(int(folder.split("_")[1]))
            except (ValueError, IndexError):
                continue

    return max(iterations) if iterations else None


def check_sugar_installation():
    """Check if SuGaR submodule is available."""
    sugar_path = os.path.join(os.path.dirname(__file__), "submodules", "SuGaR")
    if not os.path.exists(sugar_path):
        print("ERROR: SuGaR submodule not found.")
        print("Please run: git submodule update --init --recursive")
        return None

    # Check for required scripts
    train_script = os.path.join(sugar_path, "train.py")
    if not os.path.exists(train_script):
        print("ERROR: SuGaR train.py not found. Submodule may be empty.")
        print("Please run: git submodule update --init --recursive")
        return None

    return sugar_path


def find_source_path(model_path):
    """Try to find the source data path from the model's cfg_args."""
    cfg_path = os.path.join(model_path, "cfg_args")
    if os.path.exists(cfg_path):
        try:
            # cfg_args is a Python namespace serialized with repr()
            with open(cfg_path, 'r') as f:
                content = f.read()
                # Parse source_path from the namespace string
                if "source_path=" in content:
                    start = content.find("source_path=") + len("source_path=")
                    # Handle quoted paths
                    if content[start] == "'":
                        end = content.find("'", start + 1)
                        return content[start + 1:end]
                    elif content[start] == '"':
                        end = content.find('"', start + 1)
                        return content[start + 1:end]
        except Exception:
            pass
    return None


def main():
    parser = ArgumentParser(description="Extract mesh from trained Gaussian Splatting model using SuGaR")

    parser.add_argument("-m", "--model_path", required=True,
                        help="Path to trained Gaussian Splatting model")
    parser.add_argument("-s", "--source_path", default=None,
                        help="Path to source data (COLMAP dataset). Auto-detected if not specified.")
    parser.add_argument("-o", "--output_path", default=None,
                        help="Output directory for mesh (default: <model_path>/mesh)")
    parser.add_argument("--iteration", type=int, default=None,
                        help="Checkpoint iteration to use (default: latest)")
    parser.add_argument("--refinement_time", choices=["short", "medium", "long"],
                        default="short",
                        help="Refinement duration: short (~15min), medium (~1hr), long (~2hr)")
    parser.add_argument("--export_obj", action="store_true",
                        help="Also export textured OBJ mesh")
    parser.add_argument("--low_poly", action="store_true",
                        help="Generate lower polygon count mesh (faster, less detail)")
    parser.add_argument("--high_poly", action="store_true",
                        help="Generate higher polygon count mesh (slower, more detail)")
    parser.add_argument("--square_size", type=float, default=None,
                        help="Voxel size for marching cubes (smaller = more detail)")

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"ERROR: Model path does not exist: {args.model_path}")
        sys.exit(1)

    model_path = os.path.abspath(args.model_path)

    # Check SuGaR installation
    sugar_path = check_sugar_installation()
    if sugar_path is None:
        sys.exit(1)

    # Find or validate iteration
    if args.iteration is None:
        args.iteration = find_latest_iteration(model_path)
        if args.iteration is None:
            print("ERROR: No checkpoint found in model. Please specify --iteration.")
            sys.exit(1)
        print(f"Using latest iteration: {args.iteration}")

    checkpoint_path = os.path.join(model_path, "point_cloud", f"iteration_{args.iteration}")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # Find source path
    source_path = args.source_path
    if source_path is None:
        source_path = find_source_path(model_path)
        if source_path is None:
            print("ERROR: Could not auto-detect source data path.")
            print("Please specify with -s/--source_path")
            sys.exit(1)
        print(f"Auto-detected source path: {source_path}")

    if not os.path.exists(source_path):
        print(f"ERROR: Source path does not exist: {source_path}")
        sys.exit(1)

    source_path = os.path.abspath(source_path)

    # Set output path
    if args.output_path is None:
        args.output_path = os.path.join(model_path, "mesh")
    os.makedirs(args.output_path, exist_ok=True)
    output_path = os.path.abspath(args.output_path)

    print(f"\n{'='*60}")
    print("SuGaR Mesh Extraction")
    print(f"{'='*60}")
    print(f"Model path:      {model_path}")
    print(f"Source path:     {source_path}")
    print(f"Output path:     {output_path}")
    print(f"Iteration:       {args.iteration}")
    print(f"Refinement:      {args.refinement_time}")
    print(f"{'='*60}\n")

    # Build SuGaR command
    sugar_train = os.path.join(sugar_path, "train.py")

    cmd = [
        sys.executable, sugar_train,
        "-s", source_path,
        "-c", model_path,
        "-r", "sdf" if not args.low_poly else "density",
        "--refinement_time", args.refinement_time,
        "--export_ply", "True",
    ]

    # Add optional parameters
    if args.export_obj:
        cmd.extend(["--export_obj", "True"])

    if args.high_poly:
        cmd.extend(["--n_vertices_in_mesh", "1000000"])
    elif args.low_poly:
        cmd.extend(["--n_vertices_in_mesh", "200000"])

    if args.square_size is not None:
        cmd.extend(["--square_size", str(args.square_size)])

    print("Running SuGaR extraction...")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, cwd=sugar_path, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: SuGaR extraction failed with exit code {e.returncode}")
        print("\nTroubleshooting:")
        print("1. Ensure you're using the SuGaR conda environment:")
        print("   conda activate sugar")
        print("2. Verify PyTorch and CUDA are working:")
        print("   python -c \"import torch; print(torch.cuda.is_available())\"")
        print("3. Check GPU memory - SuGaR requires significant VRAM")
        sys.exit(1)
    except FileNotFoundError:
        print("ERROR: Could not run Python. Check your environment.")
        sys.exit(1)

    # Find and report output files
    print(f"\n{'='*60}")
    print("Extraction complete!")
    print(f"{'='*60}")

    # SuGaR outputs to its own directory structure, find the results
    sugar_output = os.path.join(sugar_path, "output")
    if os.path.exists(sugar_output):
        print(f"\nOutput files in SuGaR directory: {sugar_output}")
        for root, dirs, files in os.walk(sugar_output):
            for f in files:
                if f.endswith(('.ply', '.obj')):
                    print(f"  - {os.path.join(root, f)}")

    print(f"\nTo view the mesh, use a 3D viewer like MeshLab or Blender.")


if __name__ == "__main__":
    main()
