"""
Training Script for UTNet
单阶段端到端训练
支持多GPU分布式训练
"""
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

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dataloader.dex_ycb_dataset import DexYCBDataset
from dataloader.ho3d_dataset import HO3DDataset
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
    """Create data loader with optional distributed sampling"""
    dataset_name = config['dataset'].get('name', 'DexYCB').lower()
    
    # Handle rectangular img_size for dataset
    # Dataset will output square images, model will crop to rectangular if needed
    dataset_img_size = config['dataset']['img_size']
    if isinstance(dataset_img_size, (list, tuple)):
        # For rectangular input, use height (first dimension) for dataset
        # Dataset will output square images, we'll crop to rectangular in model if needed
        dataset_img_size = dataset_img_size[0]  # Use height
    
    if dataset_name == 'ho3d':
        # HO3D dataset
        dataset = HO3DDataset(
            data_split=split,
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
            train=(split == 'train')
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
        # Extract only x, y coordinates from joint_img (which has shape (B, J, 3))
        # Data loader always provides these, so we don't need None checks
        joint_img = batch['joint_img']  # (B, J, 3)
        keypoints_2d = joint_img[:, :, :2].to(device)  # (B, J, 2) - extract x, y coordinates
        
        # Move all targets to device
        # Note: vertices are not provided in dataset, so we skip vertex loss
        # This is consistent with WiLoR which also doesn't use vertex loss directly
        targets = {
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': batch['joints_3d_gt'].to(device),  # (B, J, 3)
            'vertices': None,  # Not available in dataset, vertex loss will be skipped
            'mano_pose': batch['mano_pose'].to(device),  # (B, 48)
            'mano_shape': batch['mano_shape'].to(device)  # (B, 10)
        }
        
        # Compute loss
        losses = criterion(predictions, targets)
        loss = losses['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        num_batches += 1
        
        # Logging (only on rank 0)
        if rank == 0 and batch_idx % log_freq == 0:
            for key, val in losses.items():
                writer.add_scalar(f'Train/{key}', val.item(), epoch * len(dataloader) + batch_idx)
        
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
         writer: SummaryWriter, rank: int = 0, distributed: bool = False):
    """Test model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Set epoch for DistributedSampler
    if distributed and hasattr(dataloader.sampler, 'set_epoch'):
        dataloader.sampler.set_epoch(epoch)
    
    with torch.no_grad():
        # Only show progress bar on rank 0
        pbar = tqdm(dataloader, desc=f'Test {epoch}') if rank == 0 else dataloader
        for batch in pbar:
            rgb = batch['rgb'].to(device)
            depth = batch['depth'].to(device) if batch.get('depth') is not None else None
            n_i = batch['n_i'].to(device)
            
            predictions = model(rgb, depth, n_i)
            
            # Prepare targets (similar to WiLoR: directly use batch data, no None values)
            # Extract only x, y coordinates from joint_img (which has shape (B, J, 3))
            joint_img = batch['joint_img']  # (B, J, 3)
            keypoints_2d = joint_img[:, :, :2].to(device)  # (B, J, 2) - extract x, y coordinates
            
            # Move all targets to device
            # Note: vertices are not provided in dataset, so we skip vertex loss
            targets = {
                'keypoints_2d': keypoints_2d,
                'keypoints_3d': batch['joints_3d_gt'].to(device),  # (B, J, 3)
                'vertices': None,  # Not available in dataset, vertex loss will be skipped
                'mano_pose': batch['mano_pose'].to(device),  # (B, 48)
                'mano_shape': batch['mano_shape'].to(device)  # (B, 10)
            }
            
            losses = criterion(predictions, targets)
            loss = losses['total_loss']
            
            total_loss += loss.item()
            num_batches += 1
    
    # Average loss across all processes
    if distributed:
        # Gather loss from all processes
        loss_tensor = torch.tensor(total_loss / num_batches, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()
    else:
        avg_loss = total_loss / num_batches
    
    # Log test loss (only on rank 0)
    if rank == 0:
        writer.add_scalar('Test/total_loss', avg_loss, epoch)
    
    return avg_loss


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
    test_loader = create_dataloader(config, split='test',
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
            
            # Test
            if (epoch + 1) % test_freq == 0:
                test_loss = test(model, test_loader, criterion, device, epoch, writer, rank, distributed)
                if rank == 0:
                    print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Test Loss = {test_loss:.4f}')
            else:
                if rank == 0:
                    print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}')
            
            # Save checkpoint (only on rank 0)
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
    finally:
        # Cleanup distributed training
        cleanup_distributed()
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()

