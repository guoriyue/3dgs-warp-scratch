# 3D Gaussian Splatting with Nvidia Warp

A simple, educational implementation of 3D Gaussian Splatting using NVIDIA's Warp framework for differentiable physics and rendering.

## Why This Implementation?

### ✅ CPU & GPU with Zero Hassle

Thanks to NVIDIA Warp, the same code runs seamlessly on both CPU and GPU — no need to deal with CUDA setup, driver issues, or device-specific kernels. Just change one line in the config.

### 🧠 Learn Modern Graphics the Easy Way

Explore core concepts in differentiable rendering and parallel graphics programming — no need for expensive GPUs or thousands of lines of boilerplate.

### 📦 Minimalist & Educational

Forget overly complex libraries with thousands of lines. This repo is intentionally clean and easy to read — ideal for learning, experimenting, or building your own Gaussian Splatting variant.

## Quick Start

![The training video](lego_demo.gif)


### Installation

```bash
# Clone the repository
git clone https://github.com/guoriyue/3dgs-warp-scratch.git
cd 3dgs-warp-scratch

# Install dependencies
pip install warp-lang numpy matplotlib imageio tqdm
```

### Download Example Data

```bash
# Download the Lego dataset
bash download_example_data.sh
```


### Rendering

```bash
# Render 3 Gaussian points – a minimalist example
python render.py
```
You should see output similar to `example_render_.png`.

### Training

```bash
# Train on Lego dataset (CPU by default)
# For GPU training, change DEVICE in config.py to "cuda"
python train.py
```


## Project Structure

```
├── train.py                  # Main training loop
├── render.py                 # Rendering script to validate outputs
├── config.py                 # Configuration and training parameters
├── scheduler.py              # Learning rate scheduler
├── forward.py                # Forward rendering pass (reimplement of graphdeco-inria/gaussian-splatting)
├── backward.py               # Backward gradient pass (reimplement of graphdeco-inria/gaussian-splatting)
├── loss.py                   # Loss functions (includes depth loss, though unused in this repo)
├── utils/
│   ├── camera_utils.py       # Load camera intrinsics and extrinsics from training data
│   ├── point_cloud_utils.py  # Point cloud I/O utilities (e.g., saving to .ply)
│   ├── math_utils.py         # General math utilities (e.g., transformation matrices)
│   └── wp_utils.py           # Warp helpers for math ops and device transfer
└── data/                     # Contains NeRF-synthetic 'Lego' dataset

```

forward.py and backward.py are adapted from [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting), which provides a pure CUDA implementation.
Densification and pruning logic is adapted from [yzslab/gaussian-splatting-lightning](https://github.com/yzslab/gaussian-splatting-lightning), a PyTorch implementation.