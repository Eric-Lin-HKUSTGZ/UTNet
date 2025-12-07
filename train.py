"""
Training Script for UTNet
单阶段端到端训练
支持多GPU分布式训练
"""
# ========== COMPATIBILITY PATCHES ==========
# Must be applied BEFORE any other imports to fix NumPy/chumpy compatibility
import numpy as np
import inspect

# Patch NumPy for old pickle files and chumpy compatibility (NumPy 1.20+, 2.0+)
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'int'):
    np.int = np.int_
if not hasattr(np, 'float'):
    np.float = np.float64
if not hasattr(np, 'complex'):
    np.complex = np.complex128
if not hasattr(np, 'object'):
    np.object = np.object_
if not hasattr(np, 'unicode'):
    np.unicode = np.str_
if not hasattr(np, 'str'):
    np.str = np.str_

# Patch inspect for Python 3.8+ compatibility with old chumpy
if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec

# ========== END PATCHES ==========

import os
import yaml
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloader.dex_ycb_dataset import DexYCBDataset
from dataloader.ho3d_dataset import HO3DDataset
from metrics.evaluator import Evaluator
from src.models.utnet import UTNet
from src.losses.loss import UTNetLoss
from src.utils.detection import HandDetector


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_dataloader(config: dict, split: str = 'train', 
                     distributed: bool = False, rank: int = 0, world_size: int = 1):
    """
    Create data loader with optional distributed sampling and train/val split
    
    Args:
        config: Configuration dictionary
        split: 'train', 'val', or 'test'
        distributed: Whether to use distributed training
        rank: Process rank for distributed training
        world_size: Number of processes for distributed training
    
    Note:
        For HO3D, 'val' split uses a subset of 'train' split since evaluation split
        has no complete GT annotations. The split ratio is controlled by 
        config['training']['val_split_ratio'].
    """
    dataset_name = config['dataset'].get('name', 'DexYCB').lower()
    val_split_ratio = config.get('training', {}).get('val_split_ratio', 0.1)
    
    # Handle rectangular img_size for dataset
    # Dataset will output square images, model will crop to rectangular if needed
    dataset_img_size = config['dataset']['img_size']
    if isinstance(dataset_img_size, (list, tuple)):
        # For rectangular input, use height (first dimension) for dataset
        # Dataset will output square images, we'll crop to rectangular in model if needed
        dataset_img_size = dataset_img_size[0]  # Use height
    
    # For HO3D, we split train into train/val since evaluation has no full GT
    if dataset_name == 'ho3d' and split in ['train', 'val']:
        actual_split = 'train'  # Always load from train split
    else:
        actual_split = split
    
    if dataset_name == 'ho3d':
        # HO3D dataset
        dataset = HO3DDataset(
            data_split=actual_split,
            root_dir=config['dataset']['root_dir'],
            dataset_version=config['dataset'].get('version', 'v3'),
            img_size=dataset_img_size,
            aug_para=[
                config['augmentation']['sigma_com'],
                config['augmentation']['sigma_sc'],
                config['augmentation']['rot_range']
            ],
            cube_size=config['dataset'].get('cube_size', [280, 280, 280]),
            input_modal=config['dataset']['input_modal'],
            color_factor=config['dataset'].get('color_factor', 0.2),
            p_drop=config['dataset']['p_drop'],
            train=(split == 'train')  # Use original split for train flag
        )
    else:
        # Dex-YCB dataset (default)
        dataset = DexYCBDataset(
            setup=config['dataset'].get('setup', 's0'),
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
        train=(split == 'train')
    )
    
    # Split train/val for HO3D (since evaluation split has no complete GT)
    if dataset_name == 'ho3d' and split in ['train', 'val']:
        total_size = len(dataset)
        indices = list(range(total_size))
        
        # Use fixed random seed for reproducibility
        import numpy as np
        np.random.seed(42)
        np.random.shuffle(indices)
        
        split_idx = int(total_size * (1 - val_split_ratio))
        
        if split == 'train':
            indices = indices[:split_idx]
        else:  # split == 'val'
            indices = indices[split_idx:]
        
        # Use Subset to create train/val split
        from torch.utils.data import Subset
        dataset = Subset(dataset, indices)
        
        if rank == 0:
            print(f"HO3D {split} split: {len(indices)} samples ({len(indices)/total_size*100:.1f}% of train split)")
    
    # Use DistributedSampler for distributed training
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=(split == 'train'),
            drop_last=(split == 'train')  # Drop last incomplete batch for training
        )
        shuffle = False  # Sampler handles shuffling
    else:
        sampler = None
        shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=(split == 'train' and distributed)  # Drop last batch in distributed training
    )
    
    return dataloader


def create_model(config: dict, device: torch.device) -> UTNet:
    """Create UTNet model"""
    # DeConv upsampler is used instead of NAF
    
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
        focal_length=config['model']['focal_length'],
        joint_rep_type=config['model'].get('joint_rep_type', 'aa')
    )
    
    # Load pretrained ViT backbone weights (similar to WiLoR)
    pretrained_weights_path = config['model'].get('pretrained_weights', None)
    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        print(f'Loading pretrained ViT backbone weights from {pretrained_weights_path}')
        try:
            checkpoint = torch.load(pretrained_weights_path, map_location='cpu', weights_only=False)
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                pretrained_state_dict = checkpoint['state_dict']
            else:
                pretrained_state_dict = checkpoint
            
            # 1. Load ViT backbone weights (blocks and last_norm)
            model_state_dict = model.vit_backbone.state_dict()
            backbone_filtered = {}
            for key, value in pretrained_state_dict.items():
                if key.startswith('blocks.') or key == 'last_norm.weight' or key == 'last_norm.bias':
                    if key in model_state_dict:
                        if model_state_dict[key].shape == value.shape:
                            backbone_filtered[key] = value
                        else:
                            print(f'Warning: Shape mismatch for {key}: model {model_state_dict[key].shape} vs pretrained {value.shape}')
            
            if backbone_filtered:
                model.vit_backbone.load_state_dict(backbone_filtered, strict=False)
                print(f'  Loaded {len(backbone_filtered)} ViT backbone weights (blocks and last_norm)')
            
            # 2. Load RGB patch_embed weights
            if 'patch_embed.proj.weight' in pretrained_state_dict:
                rgb_proj_weight = pretrained_state_dict['patch_embed.proj.weight']
                rgb_proj_bias = pretrained_state_dict.get('patch_embed.proj.bias', None)
                
                model_rgb_proj = model.modality_fusion.rgb_embed.patch_embed.proj
                if model_rgb_proj.weight.shape == rgb_proj_weight.shape:
                    model_rgb_proj.weight.data.copy_(rgb_proj_weight)
                    if rgb_proj_bias is not None and model_rgb_proj.bias is not None:
                        if model_rgb_proj.bias.shape == rgb_proj_bias.shape:
                            model_rgb_proj.bias.data.copy_(rgb_proj_bias)
                    print('  Loaded RGB patch_embed.proj weights')
                else:
                    print(f'  Warning: RGB patch_embed shape mismatch: model {model_rgb_proj.weight.shape} vs pretrained {rgb_proj_weight.shape}')
            
            # 3. Load positional encoding weights
            if 'pos_embed' in pretrained_state_dict:
                pretrained_pos_embed = pretrained_state_dict['pos_embed']  # (1, 193, 1280)
                # Pre-trained (WiLoR): img_size=(256, 192), patch_size=16
                #   patches: H=256/16=16, W=192/16=12, total=16*12=192
                #   pos_embed: (1, 193, 1280) = [cls_token, 192 patches]
                # UTNet: img_size=(256, 192), patch_size=16
                #   patches: H=256/16=16, W=192/16=12, total=16*12=192
                #   pos_embed: (1, 192, 1280) = [192 patches]
                
                # Extract patch positional embeddings (skip cls_token)
                pretrained_patches = pretrained_pos_embed[:, 1:, :]  # (1, 192, 1280)
                
                model_pos_embed = model.modality_fusion.rgb_embed.pos_embed.pos_embed  # (1, 192, 1280)
                
                # Now both have 192 patches, so we can directly copy (no interpolation needed)
                if model_pos_embed.shape == pretrained_patches.shape:
                    model_pos_embed.data.copy_(pretrained_patches)
                    print(f'  Loaded positional encoding: {pretrained_patches.shape[1]} patches (no interpolation needed)')
                else:
                    print(f'  Warning: Positional encoding shape mismatch: model {model_pos_embed.shape} vs pretrained {pretrained_patches.shape}')
            
            print('Successfully loaded pretrained weights (backbone + embedding)')
        except Exception as e:
            print(f'Warning: Failed to load pretrained weights: {e}')
            import traceback
            traceback.print_exc()
            print('Continuing with random initialization')
    elif pretrained_weights_path is not None:
        print(f'Warning: Pretrained weights path specified but file not found: {pretrained_weights_path}')
        print('Continuing with random initialization')
    
    model = model.to(device)
    return model


def create_optimizer(model: nn.Module, config: dict):
    """Create optimizer"""
    optimizer_config = config['optimizer']
    # Get model parameters (handle DDP)
    if isinstance(model, DDP):
        model_params = model.module.parameters()
    else:
        model_params = model.parameters()
    
    if optimizer_config['type'] == 'AdamW':
        optimizer = optim.AdamW(
            model_params,
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay'],
            betas=optimizer_config['betas']
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
    
    return optimizer


def create_scheduler(optimizer, config: dict):
    """Create learning rate scheduler"""
    scheduler_config = config['scheduler']
    if scheduler_config['type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model: nn.Module, dataloader: DataLoader, 
                optimizer: optim.Optimizer, criterion: nn.Module,
                device: torch.device, epoch: int, writer: SummaryWriter,
                log_freq: int = 100, rank: int = 0, distributed: bool = False):
    """Train for one epoch"""
    model.train()
    
    # Set epoch for DistributedSampler
    if distributed and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    
    total_loss = 0.0
    num_batches = 0
    
    # Only show progress bar on rank 0
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}') if rank == 0 else dataloader
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        rgb = batch['rgb'].to(device)
        depth = batch['depth'].to(device) if batch.get('depth') is not None else None
        n_i = batch['n_i'].to(device)
        
        # Forward pass
        predictions = model(rgb, depth, n_i)
        
        # Prepare targets (similar to WiLoR: directly use batch data, no None values)
        # Prepare 2D keypoints with confidence (WiLoR format)
        # Extract x, y coordinates from joint_img and add confidence = 1.0
        joint_img = batch['joint_img']  # (B, J, 3)
        keypoints_2d_xy = joint_img[:, :, :2].to(device)  # (B, J, 2)
        batch_size, num_joints = keypoints_2d_xy.shape[:2]
        confidence_2d = torch.ones(batch_size, num_joints, 1, device=device, dtype=keypoints_2d_xy.dtype)
        keypoints_2d_with_conf = torch.cat([keypoints_2d_xy, confidence_2d], dim=-1)  # (B, J, 3)
        
        # Prepare 3D keypoints with confidence (WiLoR format)
        keypoints_3d = batch['joints_3d_gt'].to(device)  # (B, J, 3)
        confidence_3d = torch.ones(batch_size, num_joints, 1, device=device, dtype=keypoints_3d.dtype)
        keypoints_3d_with_conf = torch.cat([keypoints_3d, confidence_3d], dim=-1)  # (B, J, 4)
        
        # Move all targets to device
        # Note: vertices are not provided in dataset, so we skip vertex loss
        # This is consistent with WiLoR which also doesn't use vertex loss directly
        targets = {
            'keypoints_2d': keypoints_2d_with_conf,  # (B, J, 3) with confidence
            'keypoints_3d': keypoints_3d_with_conf,  # (B, J, 4) with confidence
            'vertices': None,  # Not available in dataset, vertex loss will be skipped
            'mano_pose': batch['mano_pose'].to(device),  # (B, 48)
            'mano_shape': batch['mano_shape'].to(device)  # (B, 10)
        }
        
        # Debug: Print data statistics for first few batches of first epoch
        if rank == 0 and epoch == 0 and batch_idx < 3:
            print(f'\n[Debug Batch {batch_idx}]')
            print(f'  RGB range: [{rgb.min():.3f}, {rgb.max():.3f}]')
            if depth is not None:
                print(f'  Depth range: [{depth.min():.3f}, {depth.max():.3f}]')
            print(f'  GT 3D joints (with conf) range: [{keypoints_3d_with_conf.min():.3f}, {keypoints_3d_with_conf.max():.3f}]')
            print(f'  GT 3D joints (no conf) mean: {keypoints_3d.mean():.3f}, std: {keypoints_3d.std():.3f}')
            print(f'  Pred 3D joints mean: {predictions["keypoints_3d"].mean():.3f}, std: {predictions["keypoints_3d"].std():.3f}')
            print(f'  Pred 3D joints range: [{predictions["keypoints_3d"].min():.3f}, {predictions["keypoints_3d"].max():.3f}]')
            print(f'  GT root joint (batch 0): {keypoints_3d[0, 0, :]}')
            print(f'  Pred root joint (batch 0): {predictions["keypoints_3d"][0, 0, :]}')
            
            # Check if GT is already centered
            gt_centered_by_root = keypoints_3d - keypoints_3d[:, [0], :]
            print(f'  GT centered by root mean: {gt_centered_by_root.mean():.3f}, std: {gt_centered_by_root.std():.3f}')
        
        # Compute loss
        losses = criterion(predictions, targets)
        loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (prevent gradient explosion)
        # Get model parameters (handle DDP)
        if isinstance(model, DDP):
            model_params = model.module.parameters()
        else:
            model_params = model.parameters()
        torch.nn.utils.clip_grad_norm_(model_params, max_norm=1.0)
        
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Logging (only on rank 0)
        if rank == 0 and batch_idx % log_freq == 0:
            # Print detailed loss breakdown for diagnosis
            loss_2d = losses.get('loss_2d', torch.tensor(0.0))
            loss_3d_joint = losses.get('loss_3d_joint', torch.tensor(0.0))
            loss_3d_vert = losses.get('loss_3d_vert', torch.tensor(0.0))
            loss_aux = losses.get('loss_aux', torch.tensor(0.0))
            # Prior loss may have sub-components (shape_prior, pose_prior)
            loss_prior_shape = losses.get('shape_prior', torch.tensor(0.0))
            loss_prior_pose = losses.get('pose_prior', torch.tensor(0.0))
            total_loss_val = loss.item()
            
            # Convert to float if tensor
            if isinstance(loss_2d, torch.Tensor):
                loss_2d = loss_2d.item()
            if isinstance(loss_3d_joint, torch.Tensor):
                loss_3d_joint = loss_3d_joint.item()
            if isinstance(loss_3d_vert, torch.Tensor):
                loss_3d_vert = loss_3d_vert.item()
            if isinstance(loss_aux, torch.Tensor):
                loss_aux = loss_aux.item()
            if isinstance(loss_prior_shape, torch.Tensor):
                loss_prior_shape = loss_prior_shape.item()
            if isinstance(loss_prior_pose, torch.Tensor):
                loss_prior_pose = loss_prior_pose.item()
            
            print(f'\n[Epoch {epoch}, Iter {batch_idx}] Loss breakdown:')
            print(f'  2D keypoint loss: {loss_2d:.4f}')
            print(f'  3D joint loss: {loss_3d_joint:.4f}')
            if loss_3d_vert > 0:
                print(f'  3D vertex loss: {loss_3d_vert:.4f}')
            print(f'  Prior loss (shape): {loss_prior_shape:.4f}')
            if loss_prior_pose > 0:
                print(f'  Prior loss (pose): {loss_prior_pose:.4f}')
            print(f'  Aux loss: {loss_aux:.4f}')
            print(f'  Total loss: {total_loss_val:.4f}')
            
            # Log to tensorboard
            for key, val in losses.items():
                if isinstance(val, torch.Tensor):
                    writer.add_scalar(f'Train/{key}', val.item(), epoch * len(dataloader) + batch_idx)
                else:
                    writer.add_scalar(f'Train/{key}', val, epoch * len(dataloader) + batch_idx)
        
        # Update progress bar (only on rank 0)
        if rank == 0:
            pbar.set_postfix({'loss': loss.item()})
    
    # Average loss across all processes
    if distributed:
        # Gather loss from all processes
        loss_tensor = torch.tensor(total_loss / num_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()
    else:
        avg_loss = total_loss / num_batches
    
    return avg_loss


def test(model: nn.Module, dataloader: DataLoader,
         criterion: nn.Module, device: torch.device, epoch: int,
         writer: SummaryWriter, rank: int = 0, distributed: bool = False,
         evaluator: Optional[Evaluator] = None):
    """
    Test model and compute evaluation metrics
    
    Args:
        model: Model to test
        dataloader: Test dataloader
        criterion: Loss function
        device: Device to run on
        epoch: Current epoch
        writer: TensorBoard writer
        rank: Process rank (for distributed training)
        distributed: Whether using distributed training
        evaluator: Optional Evaluator instance for computing metrics
    Returns:
        dict: Dictionary containing test_loss, mpjpe, pa_mpjpe, and avg_metric
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Set epoch for DistributedSampler
    if distributed and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    
    # Initialize evaluator if not provided
    if evaluator is None:
        dataset_length = len(dataloader.dataset)
        evaluator = Evaluator(
            dataset_length=dataset_length,
            keypoint_list=None,  # Use all keypoints
            root_joint_idx=0,  # Wrist joint
            metrics=['mpjpe', 'pa_mpjpe'],
            save_predictions=False
        )
    
    # Collect all predictions and GT for metric computation
    all_pred_keypoints = []
    all_gt_keypoints = []
    
    with torch.no_grad():
        # Only show progress bar on rank 0
        pbar = tqdm(dataloader, desc=f'Val {epoch}') if rank == 0 else dataloader
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device) if batch.get('depth') is not None else None
            n_i = batch['n_i'].to(device)
            
            predictions = model(rgb, depth, n_i)
            
            # Prepare 2D keypoints with confidence (WiLoR format)
            # Extract x, y coordinates from joint_img and add confidence = 1.0
            joint_img = batch['joint_img']  # (B, J, 3)
            keypoints_2d_xy = joint_img[:, :, :2].to(device)  # (B, J, 2)
            batch_size, num_joints = keypoints_2d_xy.shape[:2]
            confidence_2d = torch.ones(batch_size, num_joints, 1, device=device, dtype=keypoints_2d_xy.dtype)
            keypoints_2d_with_conf = torch.cat([keypoints_2d_xy, confidence_2d], dim=-1)  # (B, J, 3)
            
            # Prepare 3D keypoints with confidence (WiLoR format)
            keypoints_3d = batch['joints_3d_gt'].to(device)  # (B, J, 3)
            confidence_3d = torch.ones(batch_size, num_joints, 1, device=device, dtype=keypoints_3d.dtype)
            keypoints_3d_with_conf = torch.cat([keypoints_3d, confidence_3d], dim=-1)  # (B, J, 4)
            
            # Move all targets to device
            # Note: vertices are not provided in dataset, so we skip vertex loss
            targets = {
                'keypoints_2d': keypoints_2d_with_conf,  # (B, J, 3) with confidence
                'keypoints_3d': keypoints_3d_with_conf,  # (B, J, 4) with confidence
                'vertices': None,  # Not available in dataset, vertex loss will be skipped
                'mano_pose': batch['mano_pose'].to(device),  # (B, 48)
                'mano_shape': batch['mano_shape'].to(device)  # (B, 10)
            }
            
            losses = criterion(predictions, targets)
            loss = losses['total_loss']
            
            total_loss += loss.item()
            num_batches += 1
    
            # Collect predictions and GT (remove confidence dimension)
            pred_keypoints_3d = predictions['keypoints_3d']  # (B, J, 3) in mm
            gt_keypoints_3d = targets['keypoints_3d'][:, :, :3]  # (B, J, 3) in mm - remove confidence
            
            # Store on CPU to save GPU memory
            all_pred_keypoints.append(pred_keypoints_3d.cpu())
            all_gt_keypoints.append(gt_keypoints_3d.cpu())
    
    # Average loss across all processes
    if distributed:
        # Gather loss from all processes
        loss_tensor = torch.tensor(total_loss / num_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()
    else:
        avg_loss = total_loss / num_batches
    
    # Compute metrics with all predictions from all GPUs
    metrics_dict = {}
    
    if distributed:
        # Concatenate all batches from this process
        local_pred = torch.cat(all_pred_keypoints, dim=0)  # (N_local, J, 3)
        local_gt = torch.cat(all_gt_keypoints, dim=0)  # (N_local, J, 3)
        
        # Use all_gather to collect data from all processes
        world_size = dist.get_world_size()
        
        # Gather sizes from all processes first
        local_size = torch.tensor([local_pred.shape[0]], dtype=torch.long, device=device)
        size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
        dist.all_gather(size_list, local_size)
        
        # Only rank 0 computes metrics
        if rank == 0:
            # Gather predictions and GTs from all processes
            all_preds_list = []
            all_gts_list = []
            
            for i, size in enumerate(size_list):
                num_samples = size.item()
                if i == rank:
                    # Use local data for rank 0
                    all_preds_list.append(local_pred.cpu())
                    all_gts_list.append(local_gt.cpu())
                else:
                    # Receive from other ranks
                    recv_pred = torch.zeros(num_samples, local_pred.shape[1], local_pred.shape[2], 
                                          dtype=local_pred.dtype, device=device)
                    recv_gt = torch.zeros(num_samples, local_gt.shape[1], local_gt.shape[2],
                                        dtype=local_gt.dtype, device=device)
                    dist.recv(recv_pred, src=i)
                    dist.recv(recv_gt, src=i)
                    all_preds_list.append(recv_pred.cpu())
                    all_gts_list.append(recv_gt.cpu())
            
            # Concatenate all data
            all_pred = torch.cat(all_preds_list, dim=0)  # (N_total, J, 3)
            all_gt = torch.cat(all_gts_list, dim=0)  # (N_total, J, 3)
            
            # Compute metrics directly without using Evaluator's array storage
            # This avoids array size mismatch issues
            from metrics.pose_metrics import compute_mpjpe, compute_pa_mpjpe
            
            # Center at root joint (wrist = index 0)
            all_pred_centered = all_pred - all_pred[:, [0], :]
            all_gt_centered = all_gt - all_gt[:, [0], :]
            
            # Filter invalid samples
            gt_var = all_gt_centered.var(dim=(1, 2))
            finite_mask = torch.isfinite(all_gt_centered).all(dim=(1, 2))
            valid_mask = (gt_var > 1e-8) & finite_mask
            
            if valid_mask.any():
                all_pred_valid = all_pred_centered[valid_mask].to(device)
                all_gt_valid = all_gt_centered[valid_mask].to(device)
                
                # Compute metrics on all valid samples at once
                mpjpe_array = compute_mpjpe(all_pred_valid, all_gt_valid)  # (N_valid,)
                pa_mpjpe_array = compute_pa_mpjpe(all_pred_valid, all_gt_valid)  # (N_valid,)
                
                # Average over all valid samples
                mpjpe = float(mpjpe_array.mean())
                pa_mpjpe = float(pa_mpjpe_array.mean())
                avg_metric = (mpjpe + pa_mpjpe) / 2.0
            else:
                mpjpe = float('inf')
                pa_mpjpe = float('inf')
                avg_metric = float('inf')
            
            metrics_dict = {
                'test_loss': avg_loss,
                'mpjpe': mpjpe,
                'pa_mpjpe': pa_mpjpe,
                'avg_metric': avg_metric
            }
            
            # Log to tensorboard
            writer.add_scalar('Val/total_loss', avg_loss, epoch)
            writer.add_scalar('Val/mpjpe', mpjpe, epoch)
            writer.add_scalar('Val/pa_mpjpe', pa_mpjpe, epoch)
            writer.add_scalar('Val/avg_metric', avg_metric, epoch)
            
            print(f'  Evaluated on {all_pred.shape[0]} samples from all {world_size} GPUs')
        else:
            # Non-rank-0 processes: send data to rank 0
            dist.send(local_pred.to(device), dst=0)
            dist.send(local_gt.to(device), dst=0)
            
            # Return dummy values
            metrics_dict = {
                'test_loss': avg_loss,
                'mpjpe': float('inf'),
                'pa_mpjpe': float('inf'),
                'avg_metric': float('inf')
            }
    else:
        # Single GPU mode - concatenate and evaluate all data
        if rank == 0:
            all_pred = torch.cat(all_pred_keypoints, dim=0)  # (N, J, 3)
            all_gt = torch.cat(all_gt_keypoints, dim=0)  # (N, J, 3)
            
            # Compute metrics directly without using Evaluator's array storage
            from metrics.pose_metrics import compute_mpjpe, compute_pa_mpjpe
            
            # Center at root joint (wrist = index 0)
            all_pred_centered = all_pred - all_pred[:, [0], :]
            all_gt_centered = all_gt - all_gt[:, [0], :]
            
            # Filter invalid samples
            gt_var = all_gt_centered.var(dim=(1, 2))
            finite_mask = torch.isfinite(all_gt_centered).all(dim=(1, 2))
            valid_mask = (gt_var > 1e-8) & finite_mask
            
            if valid_mask.any():
                all_pred_valid = all_pred_centered[valid_mask].to(device)
                all_gt_valid = all_gt_centered[valid_mask].to(device)
                
                # Compute metrics on all valid samples at once
                mpjpe_array = compute_mpjpe(all_pred_valid, all_gt_valid)  # (N_valid,)
                pa_mpjpe_array = compute_pa_mpjpe(all_pred_valid, all_gt_valid)  # (N_valid,)
                
                # Average over all valid samples
                mpjpe = float(mpjpe_array.mean())
                pa_mpjpe = float(pa_mpjpe_array.mean())
                avg_metric = (mpjpe + pa_mpjpe) / 2.0
            else:
                mpjpe = float('inf')
                pa_mpjpe = float('inf')
                avg_metric = float('inf')
            
            metrics_dict = {
                'test_loss': avg_loss,
                'mpjpe': mpjpe,
                'pa_mpjpe': pa_mpjpe,
                'avg_metric': avg_metric
            }
            
            # Log to tensorboard
            writer.add_scalar('Val/total_loss', avg_loss, epoch)
            writer.add_scalar('Val/mpjpe', mpjpe, epoch)
            writer.add_scalar('Val/pa_mpjpe', pa_mpjpe, epoch)
            writer.add_scalar('Val/avg_metric', avg_metric, epoch)
            
            print(f'  Evaluated on {all_pred.shape[0]} samples')
        else:
            metrics_dict = {
                'test_loss': avg_loss,
                'mpjpe': float('inf'),
                'pa_mpjpe': float('inf'),
                'avg_metric': float('inf')
            }
    
    return metrics_dict


def setup_distributed(backend='nccl'):
    """Initialize distributed training"""
    # Get environment variables set by torch.distributed.launch or torchrun
    rank = int(os.environ.get('RANK', -1))
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    world_size = int(os.environ.get('WORLD_SIZE', -1))
    
    if rank == -1:
        # Not in distributed mode
        return False, 0, 1, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize process group
    dist.init_process_group(backend=backend)
    
    # Set device for this process
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    # Set random seed for reproducibility
    torch.manual_seed(42 + rank)
    np.random.seed(42 + rank)
    
    return True, rank, world_size, device


def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description='Train UTNet')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU device ID to use (single GPU mode). For multi-GPU, use torch.distributed.launch or torchrun')
    args = parser.parse_args()
    
    # Setup distributed training
    distributed, rank, world_size, device = setup_distributed()
    
    # Load config
    config = load_config(args.config)
    
    # Override device if specified (single GPU mode)
    if args.gpu is not None and not distributed:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
            if rank == 0:
                print(f'Using specified GPU: {device}')
        else:
            if rank == 0:
                print(f'Warning: GPU {args.gpu} specified but CUDA not available. Using CPU.')
            device = torch.device('cpu')
    elif not distributed:
        # Use config file setting or default (single GPU mode)
        if torch.cuda.is_available():
            device_str = config.get('device', 'cuda')
            if device_str.startswith('cuda'):
                device = torch.device(device_str)
            else:
                device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')
        if rank == 0:
            print(f'Using device: {device}')
    
    if distributed and rank == 0:
        print(f'Distributed training on {world_size} GPUs')
    elif not distributed:
        print(f'Single GPU training on {device}')
    
    # Create directories (only on rank 0)
    save_dir = Path(config['training']['save_dir'])
    log_dir = Path(config['training']['log_dir'])
    if rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # TensorBoard writer (only on rank 0)
    writer = SummaryWriter(log_dir=str(log_dir)) if rank == 0 else None
    
    # Create data loaders
    if rank == 0:
        print('Creating data loaders...')
    train_loader = create_dataloader(config, split='train', 
                                     distributed=distributed, rank=rank, world_size=world_size)
    val_loader = create_dataloader(config, split='val',
                                   distributed=distributed, rank=rank, world_size=world_size)
    
    # Create model
    if rank == 0:
        print('Creating model...')
    model = create_model(config, device)
    
    # Wrap model with DDP for distributed training
    if distributed:
        # Note on find_unused_parameters:
        # - False: Better performance, but requires all parameters to be used in every forward pass
        # - True: Handles unused parameters (e.g., from conditional branches), but has ~5-10% performance overhead
        # 
        # We set to False by default because:
        # 1. Most parameters are used in every forward pass
        # 2. Even when depth is dropped (n_i=0), T_depth_aux is still computed and participates in gradient
        #    (though with zero weight, it still creates a gradient path)
        # 
        # If you encounter "RuntimeError: Expected to have finished reduction..." during training,
        # change this to True
        model = DDP(model, device_ids=[device.index], output_device=device.index,
                   find_unused_parameters=True)  # Set to True if you get unused parameter errors
        model_module = model.module  # Access underlying model
    else:
        model_module = model
    
    # Create loss function
    loss_config = config['loss']
    criterion = UTNetLoss(
        w_2d=loss_config['w_2d'],
        w_3d_joint=loss_config['w_3d_joint'],
        w_3d_vert=loss_config['w_3d_vert'],
        w_prior=loss_config['w_prior'],
        w_aux=loss_config['w_aux'],
        use_vertex_loss=loss_config['use_vertex_loss'],
        use_aux_loss=loss_config['use_aux_loss']
    )
    
    # Create optimizer
    optimizer = create_optimizer(model, config)
    
    # Create scheduler
    scheduler = create_scheduler(optimizer, config)
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        if rank == 0:
            print(f'Resuming from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        # Load model state (handle both DDP and non-DDP models)
        if distributed:
            model_module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    save_freq = config['training']['save_freq']
    test_freq = config['training']['test_freq']
    log_freq = config['logging']['log_freq']
    
    # Initialize evaluator for validation set (only on rank 0)
    if rank == 0:
        val_dataset_length = len(val_loader.dataset)
        val_evaluator = Evaluator(
            dataset_length=val_dataset_length,
            keypoint_list=None,  # Use all keypoints
            root_joint_idx=0,  # Wrist joint
            metrics=['mpjpe', 'pa_mpjpe'],
            save_predictions=False
        )
    else:
        val_evaluator = None
    
    # Track best average metric (mpjpe + pa_mpjpe) / 2
    best_avg_metric = float('inf')
    best_epoch = -1
    
    if rank == 0:
        print('Starting training...')
    
    try:
        for epoch in range(start_epoch, num_epochs):
            # Train
            train_loss = train_epoch(
                model, train_loader, optimizer, criterion,
                device, epoch, writer, log_freq, rank, distributed
            )
            
            # Update learning rate
            if scheduler is not None:
                scheduler.step()
            
            # Validation every epoch
            # Note: test() function will reset evaluator.counter internally
            val_metrics = test(model, val_loader, criterion, device, epoch, writer, 
                              rank, distributed, evaluator=val_evaluator)
            
            if rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                val_loss = val_metrics.get('test_loss', float('inf'))
                mpjpe = val_metrics.get('mpjpe', float('inf'))
                pa_mpjpe = val_metrics.get('pa_mpjpe', float('inf'))
                avg_metric = val_metrics.get('avg_metric', float('inf'))
                
                print(f'\nEpoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, LR = {current_lr:.2e}')
                print(f'  MPJPE: {mpjpe:.3f} mm, PA-MPJPE: {pa_mpjpe:.3f} mm, Avg Metric: {avg_metric:.3f} mm')
                
                # Save best model based on average metric (mpjpe + pa_mpjpe) / 2
                if avg_metric < best_avg_metric:
                    best_avg_metric = avg_metric
                    best_epoch = epoch
                    
                    # Get model state dict (handle DDP)
                    if distributed:
                        best_model_state_dict = model_module.state_dict()
                    else:
                        best_model_state_dict = model.state_dict()
                    
                    best_checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': best_model_state_dict,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'mpjpe': mpjpe,
                        'pa_mpjpe': pa_mpjpe,
                        'avg_metric': avg_metric,
                    }
                    if scheduler is not None:
                        best_checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                    
                    best_checkpoint_path = save_dir / 'best_model.pth'
                    torch.save(best_checkpoint, best_checkpoint_path)
                    print(f'  ✓ Saved best model (avg_metric={avg_metric:.3f} mm) to {best_checkpoint_path}')
            
            # Save regular checkpoint (only on rank 0, based on save_freq)
            if rank == 0 and (epoch + 1) % save_freq == 0:
                # Get model state dict (handle DDP)
                if distributed:
                    model_state_dict = model_module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                }
                if scheduler is not None:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                checkpoint_path = save_dir / f'checkpoint_epoch_{epoch+1}.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')
        
        # Save final model (only on rank 0)
        if rank == 0:
            if distributed:
                model_state_dict = model_module.state_dict()
            else:
                model_state_dict = model.state_dict()
            final_path = save_dir / 'final_model.pth'
            torch.save(model_state_dict, final_path)
            print(f'Saved final model to {final_path}')
            print(f'Best model: Epoch {best_epoch}, Avg Metric = {best_avg_metric:.3f} mm')
    finally:
        # Cleanup distributed training
        cleanup_distributed()
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()

