# Metrics Evaluation Guide

This directory contains evaluation tools for UTNet model performance.

## Files

- `evaluate_model.py`: Main evaluation script
- `pose_metrics.py`: MPJPE and PA-MPJPE computation functions
- `evaluator.py`: Batch evaluation class

## Usage

### Basic Usage

Evaluate model on test set:

```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --output metrics/evaluation_results.json
```

### Options

- `--config`: Path to config file (default: `config/config.yaml`)
- `--checkpoint`: Path to model checkpoint (required)
- `--split`: Dataset split to evaluate on - `train`, `test`, or `val` (default: `test`)
- `--batch_size`: Batch size for evaluation (default: 8)
- `--output`: Output file for results (default: `metrics/evaluation_results.json`)
- `--save_predictions`: Save predictions for further analysis (optional)
- `--device`: Device to use - `cuda` or `cpu` (default: `cuda`)
- `--gpu`: GPU card number to use (e.g., 0, 1, 2). If not specified, uses default GPU or device argument

### Examples

1. Evaluate on test set with custom batch size:
```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --batch_size 16 \
    --output metrics/test_results.json
```

2. Evaluate and save predictions:
```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --save_predictions \
    --output metrics/test_results_with_preds.json
```

3. Evaluate on validation set:
```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split val \
    --output metrics/val_results.json
```

4. Evaluate on specific GPU card:
```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --gpu 1 \
    --output metrics/test_results.json
```

5. Evaluate on CPU:
```bash
python metrics/evaluate_model.py \
    --checkpoint checkpoints/checkpoint_epoch_0.pth \
    --split test \
    --device cpu \
    --output metrics/test_results.json
```

## Metrics

The evaluation computes the following metrics:

- **MPJPE** (Mean Per Joint Position Error): Average Euclidean distance between predicted and ground truth 3D joints in millimeters
- **PA-MPJPE** (Procrustes Aligned MPJPE): MPJPE after Procrustes alignment (also called Reconstruction Error)

Both metrics are computed after centering predictions and ground truth at the wrist (root joint).

## Output Format

The results are saved as JSON:

```json
{
  "metrics": {
    "mpjpe": 12.345,
    "pa_mpjpe": 8.901
  },
  "num_samples": 1000
}
```

## Using Metrics in Code

You can also use the metrics functions directly in your code:

```python
from metrics import compute_mpjpe, compute_pa_mpjpe

# Compute metrics for a batch
mpjpe = compute_mpjpe(pred_joints, gt_joints)  # Returns mm
pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)  # Returns mm
```

## Requirements

- PyTorch
- NumPy


