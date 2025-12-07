"""
NAF Multi-scale Upsampling Module
从 F_0 生成多尺度特征 {F_0, F_1, F_2, ...}
实现内容自适应的多尺度上采样
参考 NAF/src/model/naf.py

支持两种模式：
1. 官方预训练NAF (use_official_naf=True): 使用 torch.hub 加载官方预训练模型
2. 自定义NAF (use_official_naf=False): 使用自定义实现的NAF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


class MultiScaleUpsampler(nn.Module):
    """
    Multi-scale feature upsampler using cross-attention
    Simplified version using standard PyTorch operations
    """
    def __init__(self, dim=1280, num_heads=16, num_scales=3, 
                 upscale_factors=[1, 2, 4]):
        """
        Args:
            dim: feature dimension
            num_heads: number of attention heads
            num_scales: number of output scales
            upscale_factors: upscale factors for each scale [1, 2, 4, ...]
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.upscale_factors = upscale_factors
        
        # Cross-attention upsamplers for each scale
        self.upsamplers = nn.ModuleList()
        for i in range(num_scales):
            if upscale_factors[i] == 1:
                # No upsampling needed
                self.upsamplers.append(nn.Identity())
            else:
                # Use transposed convolution for upsampling
                self.upsamplers.append(
                    nn.Sequential(
                        nn.ConvTranspose2d(dim, dim, kernel_size=upscale_factors[i], 
                                         stride=upscale_factors[i], padding=0),
                        nn.GroupNorm(1, dim),  # Normalize channels (equivalent to LayerNorm for channels)
                        nn.GELU()
                    )
                )

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-scale features from input feature map
        
        Args:
            features: (B, D, H, W) input feature map F_0
        Returns:
            multi_scale_features: list of (B, D, H_i, W_i) feature maps
                [F_0, F_1, F_2, ...] with increasing resolution
        """
        B, D, H, W = features.shape
        multi_scale_features = []
        
        current_features = features
        for i, upsampler in enumerate(self.upsamplers):
            if i == 0:
                # First scale: no upsampling
                multi_scale_features.append(current_features)
            else:
                # Upsample to next scale
                upsampled = upsampler(current_features)
                multi_scale_features.append(upsampled)
                current_features = upsampled
        
        return multi_scale_features


class SimpleNAFUpsampler(nn.Module):
    """
    Simplified NAF upsampler using bilinear interpolation + conv
    For content-adaptive upsampling
    """
    def __init__(self, dim=1280, num_scales=3):
        """
        Args:
            dim: feature dimension
            num_scales: number of output scales
        """
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        
        # Refinement convolutions for each scale
        self.refine_convs = nn.ModuleList()
        for i in range(num_scales):
            self.refine_convs.append(
                nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.GroupNorm(1, dim),  # Normalize channels (equivalent to LayerNorm for channels)
                    nn.GELU(),
                    nn.Conv2d(dim, dim, kernel_size=3, padding=1),
                    nn.GroupNorm(1, dim)  # Normalize channels
                )
            )

    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-scale features
        
        Args:
            features: (B, D, H, W) input feature map F_0
        Returns:
            multi_scale_features: list of feature maps at different scales
        """
        B, D, H, W = features.shape
        multi_scale_features = []
        
        current_features = features
        for i in range(self.num_scales):
            if i == 0:
                # First scale: original resolution
                refined = self.refine_convs[i](current_features)
                multi_scale_features.append(refined)
            else:
                # Upsample by factor of 2
                upsampled = F.interpolate(
                    current_features, 
                    scale_factor=2, 
                    mode='bilinear', 
                    align_corners=False
                )
                # Refine
                refined = self.refine_convs[i](upsampled)
                multi_scale_features.append(refined)
                current_features = refined
        
        return multi_scale_features


class OfficialNAFWrapper(nn.Module):
    """
    Wrapper for official NAF model from torch.hub
    Adapts the official NAF interface to our multi-scale feature generation
    """
    def __init__(self, dim=1280, num_scales=3, device='cuda', pretrained=True):
        """
        Args:
            dim: feature dimension
            num_scales: number of output scales
            device: device to load NAF model
            pretrained: whether to use pretrained weights
        """
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales
        self.device = device
        
        # Load official NAF model from local path
        import sys
        import os
        
        # Local paths for NAF code and pretrained weights
        naf_hub_path = "/data0/users/Robert/linweiquan/UTNet/pretraining_model/hub/valeoai_NAF_main"
        naf_checkpoint_path = "/data0/users/Robert/linweiquan/UTNet/pretraining_model/hub/checkpoints/naf_release.pth"
        
        try:
            # Check if local NAF code exists
            if not os.path.exists(naf_hub_path):
                raise FileNotFoundError(f"NAF code not found at {naf_hub_path}")
            if not os.path.exists(os.path.join(naf_hub_path, "src", "model", "naf.py")):
                raise FileNotFoundError(f"NAF model file not found in {naf_hub_path}")
            
            # Add NAF root path to sys.path so that 'from src.model.naf import NAF' works
            # The path must be the root directory containing 'src' folder
            if naf_hub_path not in sys.path:
                sys.path.insert(0, naf_hub_path)
            
            # Directly import NAF class instead of using torch.hub.load
            # This avoids the module import issues with torch.hub
            from pretraining_model.hub.valeoai_NAF_main.src.model.naf import NAF
            
            # Create NAF model instance
            self.naf_model = NAF().to(device)
            
            # Load pretrained weights from local checkpoint if requested
            if pretrained:
                if os.path.exists(naf_checkpoint_path):
                    checkpoint = torch.load(naf_checkpoint_path, map_location=device, weights_only=False)
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        state_dict = checkpoint['state_dict']
                    else:
                        state_dict = checkpoint
                    self.naf_model.load_state_dict(state_dict, strict=False)
                    print(f"Successfully loaded NAF pretrained weights from {naf_checkpoint_path}")
                else:
                    print(f"Warning: Pretrained checkpoint not found at {naf_checkpoint_path}, using random initialization")
            
            self.naf_model.eval()
            # Freeze NAF parameters
            for param in self.naf_model.parameters():
                param.requires_grad = False
            self.use_official = True
            print(f"Successfully loaded official NAF model from local path: {naf_hub_path}")
        except Exception as e:
            print(f"Warning: Failed to load official NAF from local path: {e}. Falling back to custom NAF.")
            import traceback
            traceback.print_exc()
            self.use_official = False
            # Fallback to custom implementation
            self.custom_naf = SimpleNAFUpsampler(dim=dim, num_scales=num_scales)
    
    def forward(self, features: torch.Tensor, image: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Generate multi-scale features using official NAF
        
        Args:
            features: (B, D, H, W) input feature map F_0
            image: (B, 3, H_img, W_img) optional RGB image for NAF guidance
        Returns:
            multi_scale_features: list of (B, D, H_i, W_i) feature maps
                [F_0, F_1, F_2, ...] with increasing resolution
        """
        if not self.use_official:
            # Fallback to custom NAF
            return self.custom_naf(features)
        
        B, D, H, W = features.shape
        multi_scale_features = []
        
        # First scale: original resolution (no upsampling)
        multi_scale_features.append(features)
        
        current_features = features
        current_h, current_w = H, W
        
        # Generate multi-scale features
        for i in range(1, self.num_scales):
            # Target size: upscale by factor of 2 each time
            target_h, target_w = current_h * 2, current_w * 2
            target_size = (target_h, target_w)
            
            # If image is provided, use it for guidance; otherwise use upsampled features as guidance
            if image is not None:
                # Resize image to match target size for NAF
                guidance_image = F.interpolate(image, size=target_size, mode='bilinear', align_corners=False)
            else:
                # Use bilinear upsampled features as guidance (3-channel)
                guidance_image = F.interpolate(
                    current_features[:, :3, :, :] if current_features.shape[1] >= 3 
                    else current_features.repeat(1, 3, 1, 1)[:, :3, :, :],
                    size=target_size, mode='bilinear', align_corners=False
                )
            
            # Use official NAF for upsampling
            # Note: NAF parameters are frozen, but we allow gradients to flow through
            upsampled = self.naf_model(guidance_image, current_features, target_size)
            
            multi_scale_features.append(upsampled)
            current_features = upsampled
            current_h, current_w = target_h, target_w
        
        return multi_scale_features


class NAFUpsampler(nn.Module):
    """
    Unified NAF upsampler that supports both official and custom implementations
    """
    def __init__(self, dim=1280, num_scales=3, use_official_naf=True, device='cuda', pretrained=True):
        """
        Args:
            dim: feature dimension
            num_scales: number of output scales
            use_official_naf: if True, use official pretrained NAF; else use custom NAF
            device: device to load NAF model (only used if use_official_naf=True)
            pretrained: whether to use pretrained weights (only used if use_official_naf=True)
        """
        super().__init__()
        self.use_official_naf = use_official_naf
        
        if use_official_naf:
            self.upsampler = OfficialNAFWrapper(dim=dim, num_scales=num_scales, device=device, pretrained=pretrained)
        else:
            self.upsampler = SimpleNAFUpsampler(dim=dim, num_scales=num_scales)
    
    def forward(self, features: torch.Tensor, image: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        """
        Generate multi-scale features
        
        Args:
            features: (B, D, H, W) input feature map F_0
            image: (B, 3, H_img, W_img) optional RGB image for NAF guidance (only used if use_official_naf=True)
        Returns:
            multi_scale_features: list of (B, D, H_i, W_i) feature maps
        """
        if self.use_official_naf:
            return self.upsampler(features, image)
        else:
            return self.upsampler(features)

