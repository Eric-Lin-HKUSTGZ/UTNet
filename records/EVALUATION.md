# UTNet Evaluation Guide

This guide explains how to evaluate and visualize trained UTNet models.

## Quick Start

### 1. Evaluate Model Performance

Evaluate your trained model on the test set:

```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test
```

This will:
- Load the trained model from the checkpoint
- Evaluate on the test set
- Compute MPJPE and PA-MPJPE metrics
- Save results to `metrics/evaluation_results.json`

### 2. Visualize Predictions

Visualize model predictions:

```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --num_samples 10
```

This will:
- Load the trained model
- Generate visualizations for 10 samples
- Save 2D keypoint and 3D mesh visualizations

## Detailed Usage

### Evaluation

#### Basic Evaluation

```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --batch_size 8 \
    --output metrics/test_results.json
```

#### Evaluation with Saved Predictions

```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --save_predictions \
    --output metrics/test_results_with_preds.json
```

### Visualization

#### Visualize 2D Keypoints Only

```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --mode keypoints \
    --num_samples 20 \
    --output_dir visualization/results
```

#### Visualize 3D Mesh Only

```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --mode mesh \
    --num_samples 5 \
    --output_dir visualization/results
```

#### Visualize Both

```bash
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --mode both \
    --num_samples 10 \
    --output_dir visualization/results
```

#### Specify GPU Card

You can specify which GPU card to use:

```bash
# Use GPU 1
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --gpu 1

# Use GPU 2 for visualization
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --gpu 2 \
    --num_samples 10
```

#### Use CPU

```bash
# Evaluate on CPU
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --device cpu

# Visualize on CPU
python visualization/visualize_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --device cpu \
    --num_samples 10
```

## Understanding the Metrics

### MPJPE (Mean Per Joint Position Error)

- **Definition**: Average Euclidean distance between predicted and ground truth 3D joints
- **Unit**: Millimeters (mm)
- **Computation**: Computed after centering at wrist (root joint)
- **Lower is better**: Typical values range from 5-20mm for good models

### PA-MPJPE (Procrustes Aligned MPJPE)

- **Definition**: MPJPE after Procrustes alignment (also called Reconstruction Error)
- **Unit**: Millimeters (mm)
- **Computation**: 
  1. Center predictions and ground truth at wrist
  2. Apply Procrustes alignment (similarity transform)
  3. Compute MPJPE on aligned predictions
- **Lower is better**: Typically lower than MPJPE, values range from 3-15mm for good models

### Interpretation

- **MPJPE**: Measures absolute 3D position accuracy
- **PA-MPJPE**: Measures shape accuracy (ignores global scale, rotation, translation)

## Output Files

### Evaluation Results

Saved as JSON:
```json
{
  "metrics": {
    "mpjpe": 12.345,
    "pa_mpjpe": 8.901
  },
  "num_samples": 1000
}
```

### Visualization Results

- `visualization/results/keypoints_2d/`: 2D keypoint visualizations
  - Shows predictions (red) and ground truth (green)
  - Format: PNG images
  
- `visualization/results/mesh/`: 3D mesh visualizations
  - Shows rendered hand meshes
  - Includes front view, side view, and keypoint overlays
  - Format: PNG images

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size: `--batch_size 4`
   - Use CPU: `--device cpu`

2. **Mesh Rendering Fails**
   - Install dependencies: `pip install pyrender trimesh`
   - Or use keypoints-only mode: `--mode keypoints`

3. **Checkpoint Not Found**
   - Check checkpoint path
   - Ensure checkpoint file exists

4. **Dataset Path Issues**
   - Update `dataset.root_dir` in `config/config.yaml`
   - Ensure dataset is properly set up

## Advanced Usage

### Custom Evaluation

You can use the metrics functions directly in your code:

```python
from metrics import compute_mpjpe, compute_pa_mpjpe, Evaluator

# Single batch evaluation
mpjpe = compute_mpjpe(pred_joints, gt_joints)
pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)

# Batch evaluation
evaluator = Evaluator(dataset_length=1000, metrics=['mpjpe', 'pa_mpjpe'])
for batch in dataloader:
    evaluator(pred_joints, gt_joints)
evaluator.log()
```

### Custom Visualization

```python
from visualization import render_hand_keypoints, MeshRenderer

# Render 2D keypoints
img_with_keypoints = render_hand_keypoints(img, keypoints_2d)

# Render 3D mesh
renderer = MeshRenderer(img_res=256, focal_length=5000.0)
rendered_img = renderer(vertices, camera_translation, image)
```

## References

- MPJPE and PA-MPJPE are standard metrics for 3D hand pose estimation
- Implementation based on WiLoR and KeypointFusion codebases
- For more details, see `metrics/README.md` and `visualization/README.md`


