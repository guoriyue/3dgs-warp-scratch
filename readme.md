# 3D Gaussian Splatting with Warp

This repository contains an implementation of 3D Gaussian Splatting (3DGS) using NVIDIA's Warp framework for differentiable physics.

## Overview

3D Gaussian Splatting is a novel approach for fast, high-quality 3D scene reconstruction and rendering. This implementation leverages Warp for optimized parallel processing and differentiable rendering capabilities.

Key features:
- Differentiable rendering of 3D Gaussians with spherical harmonics
- Optimization-based training from multi-view images
- Point cloud densification and pruning during training
- High-quality novel view synthesis

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/warp-nerf-scratch.git
cd warp-nerf-scratch
```

2. Install required dependencies:
```bash
pip install numpy torch warp-lang matplotlib tqdm pillow plyfile
```

## Dataset Format

The training data should be organized as follows:

```
dataset_folder/
├── cameras.json     # Camera parameters
├── images/          # Training images
│   ├── image1.png
│   ├── image2.png
│   └── ...
```

The `cameras.json` file should contain a list of camera objects with the following properties:
- `id`: Camera identifier
- `position`: 3D position [x, y, z]
- `rotation`: Rotation matrix (3x3 flattened to array)
- `width`: Image width
- `height`: Image height
- `fx`, `fy`: Focal lengths

## Usage

### Training a 3D Gaussian Splatting Model

```bash
python train.py --dataset path/to/dataset --output path/to/output --iterations 30000
```

Options:
- `--dataset`: Path to the dataset (required)
- `--output`: Output directory for checkpoints and results (default: ./output)
- `--iterations`: Number of training iterations (default: 30000)
- `--learning_rate`: Learning rate for optimization (default: 0.01)
- `--num_points`: Initial number of Gaussian points (default: 5000)
- `--save_interval`: Save checkpoint every N iterations (default: 1000)

### Rendering from a trained model

```bash
python render.py --input_path path/to/output/point_cloud/iteration_X
```

Options:
- `--input_path`: Path to the saved point cloud
- `--output`: Output image filename (default: gaussian_render.png)
- `--width`: Image width (default: 1800)
- `--height`: Image height (default: 1800)

## Implementation Details

The implementation consists of the following key components:

1. `train.py`: Main training script
2. `render.py`: Script for rendering views from trained models
3. `wp_kernels.py`: Warp kernels for efficient rendering
4. `utils.py`: Utility functions for camera transformations and point cloud loading

### Training Process

The training process optimizes the parameters of a set of 3D Gaussians (position, scale, rotation, opacity, and spherical harmonics coefficients) to match a set of training images. The process involves:

1. Initialization of random Gaussians
2. For each iteration:
   - Randomly select a training view
   - Render the scene from that view
   - Compute loss against the ground truth image
   - Backpropagate and update Gaussian parameters
   - Periodically densify and prune the point cloud

## Acknowledgements

This implementation is inspired by:
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [NVIDIA Warp](https://github.com/NVIDIA/warp)

## License

[MIT License](LICENSE)
