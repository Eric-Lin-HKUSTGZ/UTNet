"""
Evaluation script for trained UTNet model
Loads a trained model and evaluates on test data, computing MPJPE and PA-MPJPE metrics
"""
import os
import sys
import yaml
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataloader.dex_ycb_dataset import DexYCBDataset
from src.models.utnet import UTNet
from torch.utils.data import DataLoader
from metrics.evaluator import Evaluator
from metrics.pose_metrics import compute_mpjpe, compute_pa_mpjpe


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_model(config: dict, checkpoint_path: str, device: torch.device) -> UTNet:
    """Load trained model from checkpoint"""
    # Create model
    model = UTNet(
        img_size=config['model']['img_size'],
        patch_size=config['model']['patch_size'],
        embed_dim=config['model']['embed_dim'],
        vit_depth=config['model']['vit_depth'],
        vit_num_heads=config['model']['vit_num_heads'],
        num_hand_joints=config['model']['num_hand_joints'],
        num_vertices=config['model']['num_vertices'],
        mano_path=config['mano']['model_path'],
        mean_params_path=config['mano']['mean_params_path'],
        num_scales=config['model']['num_scales'],
        focal_length=config['model']['focal_length']
    )
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def create_dataloader(config: dict, split: str = 'test', batch_size: int = 8):
    """Create data loader for evaluation"""
    dataset_img_size = config['dataset']['img_size']
    if isinstance(dataset_img_size, (list, tuple)):
        dataset_img_size = dataset_img_size[0]
    
    dataset = DexYCBDataset(
        setup=config['dataset']['setup'],
        split=split,
        root_dir=config['dataset']['root_dir'],
        img_size=dataset_img_size,
        aug_para=[
            config['augmentation']['sigma_com'],
            config['augmentation']['sigma_sc'],
            config['augmentation']['rot_range']
        ],
        input_modal=config['dataset']['input_modal'],
        p_drop=config['dataset']['p_drop'],
        train=False  # No augmentation for evaluation
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader, len(dataset)


def evaluate_model(model: UTNet, dataloader: DataLoader, device: torch.device,
                  dataset_length: int, save_predictions: bool = False):
    """
    Evaluate model on dataset and compute metrics
    """
    # Initialize evaluator
    evaluator = Evaluator(
        dataset_length=dataset_length,
        keypoint_list=None,  # Use all 21 keypoints
        root_joint_idx=0,  # Wrist is root joint (index 0)
        metrics=['mpjpe', 'pa_mpjpe'],
        save_predictions=save_predictions
    )
    
    model.eval()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            # Move to device
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device) if batch.get('depth') is not None else None
            n_i = batch['n_i'].to(device)
            
            # Forward pass
            predictions = model(rgb, depth, n_i)
            
            # Get predictions (21 keypoints)
            pred_keypoints_3d = predictions['keypoints_3d']  # (B, 21, 3)
            
            # Get ground truth (21 keypoints)
            # Note: dataset provides joints_3d_gt which should have 21 keypoints
            gt_keypoints_3d = batch['joints_3d_gt'].to(device)  # (B, 21, 3)
            
            # Get vertices if available
            pred_vertices = predictions.get('vertices', None)
            gt_vertices = None  # Not available in dataset
            
            # Evaluate batch
            evaluator(
                pred_keypoints_3d=pred_keypoints_3d,
                gt_keypoints_3d=gt_keypoints_3d,
                pred_vertices=pred_vertices,
                gt_vertices=gt_vertices
            )
    
    return evaluator


def print_detailed_metrics(evaluator: Evaluator):
    """Print detailed metrics including per-joint errors"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # Overall metrics
    evaluator.log()
    
    # Get metrics dictionary
    metrics_dict = evaluator.get_metrics_dict()
    
    print("\n" + "-"*60)
    print("Summary:")
    print("-"*60)
    for metric, value in metrics_dict.items():
        print(f"  {metric.upper()}: {value:.3f} mm")
    
    return metrics_dict


def save_results(metrics_dict: dict, predictions_dict: dict, output_path: str):
    """Save evaluation results to file"""
    results = {
        'metrics': metrics_dict,
        'num_samples': len(predictions_dict.get('pred_keypoints_3d', []))
    }
    
    # Save metrics as JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate UTNet model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test', 'val'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for evaluation')
    parser.add_argument('--output', type=str, default='metrics/evaluation_results.json',
                       help='Output file for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions for further analysis')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU card number to use (e.g., 0, 1, 2). If not specified, uses default GPU or device argument')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set device
    if args.device == 'cpu':
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        if args.gpu is not None:
            # Use specified GPU card
            if args.gpu >= torch.cuda.device_count():
                print(f"Warning: GPU {args.gpu} not available. Available GPUs: {torch.cuda.device_count()}")
                print(f"Using GPU 0 instead.")
                device = torch.device(f'cuda:0')
            else:
                device = torch.device(f'cuda:{args.gpu}')
        else:
            # Use default GPU or device argument
            if args.device.startswith('cuda:'):
                device = torch.device(args.device)
            else:
                device = torch.device('cuda:0')
    else:
        print("Warning: CUDA not available, using CPU")
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(config, args.checkpoint, device)
    
    # Create dataloader
    dataloader, dataset_length = create_dataloader(config, split=args.split, batch_size=args.batch_size)
    print(f"Dataset length: {dataset_length}")
    
    # Evaluate
    print("\n=== Starting Evaluation ===")
    evaluator = evaluate_model(model, dataloader, device, dataset_length, args.save_predictions)
    
    # Print results
    metrics_dict = print_detailed_metrics(evaluator)
    
    # Get predictions if saved
    predictions_dict = evaluator.get_predictions_dict() if args.save_predictions else {}
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    save_results(metrics_dict, predictions_dict, args.output)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()


