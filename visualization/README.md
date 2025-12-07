# Visualization Guide

This directory contains visualization tools for UTNet model predictions.

## Files

- `visualize_model.py`: Main visualization script
- `keypoint_renderer.py`: 2D keypoint rendering utilities
- `mesh_renderer.py`: 3D mesh rendering utilities

## Usage

### Basic Usage

Visualize 2D keypoints and 3D mesh:

```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --output_dir visualization/results \
    --num_samples 10 \
    --split test
```

### Options

- `--config`: Path to config file (default: `config/config.yaml`)
- `--checkpoint`: Path to model checkpoint (required)
- `--output_dir`: Output directory for visualizations (default: `visualization/results`)
- `--num_samples`: Number of samples to visualize (default: 10)
- `--split`: Dataset split to use - `train`, `test`, or `val` (default: `test`)
- `--mode`: Visualization mode - `keypoints`, `mesh`, or `both` (default: `both`)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--gpu`: GPU card number to use (e.g., 0, 1, 2). If not specified, uses default GPU or device argument

### Examples

1. Visualize only 2D keypoints:
```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --mode keypoints \
    --num_samples 20
```

2. Visualize only 3D mesh:
```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --mode mesh \
    --num_samples 5
```

3. Visualize on validation set:
```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split val \
    --num_samples 15
```

4. Visualize on specific GPU card:
```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --gpu 1 \
    --num_samples 10
```

5. Visualize on CPU:
```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --device cpu \
    --num_samples 10
```

## Output

The script generates visualizations in the output directory:

- `keypoints_2d/`: 2D keypoint visualizations showing predictions (red) and ground truth (green)
- `mesh/`: 3D mesh visualizations showing rendered hand meshes with keypoints

## Requirements

- PyTorch
- OpenCV
- NumPy
- For mesh rendering: `pyrender` and `trimesh` (optional)

Install optional dependencies:
```bash
pip install pyrender trimesh
```


