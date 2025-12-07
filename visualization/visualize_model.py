"""
Visualization script for trained UTNet model
Loads a trained model and visualizes predictions on test data
"""
import os
import sys
import yaml
import argparse
import torch
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataloader.dex_ycb_dataset import DexYCBDataset
from src.models.utnet import UTNet
from torch.utils.data import DataLoader
from visualization.keypoint_renderer import render_hand_keypoints, render_keypoints_2d
from visualization.mesh_renderer import MeshRenderer


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


def create_dataloader(config: dict, split: str = 'test', batch_size: int = 1):
    """Create data loader for visualization"""
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
        train=False  # No augmentation for visualization
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return dataloader


def visualize_keypoints_2d(model: UTNet, dataloader: DataLoader, device: torch.device,
                          output_dir: str, num_samples: int = 10):
    """
    Visualize 2D keypoint predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Visualizing 2D keypoints")):
            if sample_count >= num_samples:
                break
            
            # Move to device
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device) if batch.get('depth') is not None else None
            n_i = batch['n_i'].to(device)
            
            # Forward pass
            predictions = model(rgb, depth, n_i)
            
            # Get predictions
            pred_keypoints_2d = predictions['keypoints_2d'].cpu().numpy()  # (B, N, 2)
            pred_keypoints_3d = predictions['keypoints_3d'].cpu().numpy()  # (B, N, 3)
            
            # Get ground truth
            gt_keypoints_2d = batch['joint_img'][:, :, :2].numpy()  # (B, N, 2)
            gt_keypoints_3d = batch['joints_3d_gt'].numpy()  # (B, N, 3)
            
            # Process each sample in batch
            for i in range(rgb.shape[0]):
                if sample_count >= num_samples:
                    break
                
                # Convert image from tensor to numpy
                img = rgb[i].cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
                # Denormalize if needed (assuming ImageNet normalization)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                img = (img * 255).astype(np.uint8)
                
                # Scale keypoints to image size
                H, W = img.shape[:2]
                pred_kp_2d_scaled = pred_keypoints_2d[i] * np.array([W, H])
                gt_kp_2d_scaled = gt_keypoints_2d[i] * np.array([W, H])
                
                # Render keypoints
                img_with_keypoints = render_keypoints_2d(
                    img.copy(),
                    pred_kp_2d_scaled,
                    gt_kp_2d_scaled,
                    threshold=0.1
                )
                
                # Save visualization
                output_path = os.path.join(output_dir, f'keypoints_2d_{sample_count:04d}.png')
                cv2.imwrite(output_path, cv2.cvtColor(img_with_keypoints, cv2.COLOR_RGB2BGR))
                
                print(f"Saved visualization to {output_path}")
                sample_count += 1


def visualize_mesh(model: UTNet, dataloader: DataLoader, device: torch.device,
                  output_dir: str, num_samples: int = 5, mano_faces_path: str = None):
    """
    Visualize 3D mesh predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize mesh renderer
    try:
        # Try to load MANO faces
        faces = None
        if mano_faces_path is not None:
            from src.utils.mano_utils import load_mano_faces
            faces = load_mano_faces(mano_faces_path).cpu().numpy()
        
        renderer = MeshRenderer(
            img_res=256,  # Adjust based on your image size
            focal_length=5000.0,
            faces=faces
        )
    except Exception as e:
        print(f"Warning: Could not initialize mesh renderer: {e}")
        print("Skipping mesh visualization")
        return
    
    model.eval()
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Visualizing 3D mesh")):
            if sample_count >= num_samples:
                break
            
            # Move to device
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device) if batch.get('depth') is not None else None
            n_i = batch['n_i'].to(device)
            
            # Forward pass
            predictions = model(rgb, depth, n_i)
            
            # Get predictions
            pred_vertices = predictions['vertices'].cpu().numpy()  # (B, V, 3)
            pred_keypoints_2d = predictions['keypoints_2d'].cpu().numpy()  # (B, N, 2)
            pred_keypoints_3d = predictions['keypoints_3d'].cpu().numpy()  # (B, N, 3)
            
            # Get camera translation (simplified - you may need to extract from model)
            # For now, we'll use a default camera translation
            camera_translation = np.array([0.0, 0.0, 0.5])  # Adjust based on your camera setup
            
            # Get ground truth
            gt_keypoints_2d = batch['joint_img'][:, :, :2].numpy()  # (B, N, 2)
            
            # Process each sample in batch
            for i in range(rgb.shape[0]):
                if sample_count >= num_samples:
                    break
                
                # Convert image from tensor to numpy
                img = rgb[i].cpu().permute(1, 2, 0).numpy()  # (H, W, 3)
                # Denormalize if needed
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                # Render mesh
                try:
                    # Scale keypoints to image coordinates (normalized to [-0.5, 0.5])
                    H, W = img.shape[:2]
                    pred_kp_2d_normalized = (pred_keypoints_2d[i] - 0.5)  # Normalize to [-0.5, 0.5]
                    gt_kp_2d_normalized = (gt_keypoints_2d[i] - 0.5)
                    
                    # Create visualization grid
                    vertices_tensor = torch.from_numpy(pred_vertices[i:i+1]).float()
                    camera_translation_tensor = torch.from_numpy(camera_translation).unsqueeze(0).float()
                    images_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
                    pred_kp_2d_tensor = torch.from_numpy(pred_kp_2d_normalized).unsqueeze(0).float()
                    gt_kp_2d_tensor = torch.from_numpy(gt_kp_2d_normalized).unsqueeze(0).float()
                    
                    vis_grid = renderer.visualize_tensorboard(
                        vertices_tensor,
                        camera_translation_tensor,
                        images_tensor,
                        pred_kp_2d_tensor,
                        gt_kp_2d_tensor,
                        focal_length=5000.0
                    )
                    
                    # Save visualization
                    vis_img = vis_grid.permute(1, 2, 0).cpu().numpy()
                    vis_img = np.clip(vis_img, 0, 1)
                    vis_img = (vis_img * 255).astype(np.uint8)
                    
                    output_path = os.path.join(output_dir, f'mesh_{sample_count:04d}.png')
                    cv2.imwrite(output_path, cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
                    
                    print(f"Saved mesh visualization to {output_path}")
                    sample_count += 1
                except Exception as e:
                    print(f"Error rendering mesh for sample {sample_count}: {e}")
                    continue


def main():
    parser = argparse.ArgumentParser(description='Visualize UTNet predictions')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='visualization/results',
                       help='Output directory for visualizations')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'test', 'val'],
                       help='Dataset split to use')
    parser.add_argument('--mode', type=str, default='both',
                       choices=['keypoints', 'mesh', 'both'],
                       help='Visualization mode')
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
    dataloader = create_dataloader(config, split=args.split, batch_size=1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize
    if args.mode in ['keypoints', 'both']:
        print("\n=== Visualizing 2D Keypoints ===")
        keypoints_dir = os.path.join(args.output_dir, 'keypoints_2d')
        visualize_keypoints_2d(model, dataloader, device, keypoints_dir, args.num_samples)
    
    if args.mode in ['mesh', 'both']:
        print("\n=== Visualizing 3D Mesh ===")
        mesh_dir = os.path.join(args.output_dir, 'mesh')
        mano_faces_path = config['mano']['model_path']
        visualize_mesh(model, dataloader, device, mesh_dir, 
                      min(args.num_samples, 5), mano_faces_path)
    
    print(f"\nVisualization complete! Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()


