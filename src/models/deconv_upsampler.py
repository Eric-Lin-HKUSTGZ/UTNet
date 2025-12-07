"""
DeConv Multi-scale Upsampling Module
基于WiLoR的DeConv实现多尺度上采样
从 F_0 生成多尺度特征 {F_0, F_1, F_2, ...}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


def make_conv_layers(feat_dims, kernel=3, stride=1, padding=1, bnrelu_final=True):
    """
    Create convolutional layers (from WiLoR)
    """
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.Conv2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=kernel,
                stride=stride,
                padding=padding
            ))
        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def make_deconv_layers(feat_dims, bnrelu_final=True):
    """
    Create deconvolutional layers (from WiLoR)
    """
    layers = []
    for i in range(len(feat_dims)-1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=feat_dims[i],
                out_channels=feat_dims[i+1],
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))

        # Do not use BN and ReLU for final estimation
        if i < len(feat_dims)-2 or (i == len(feat_dims)-2 and bnrelu_final):
            layers.append(nn.BatchNorm2d(feat_dims[i+1]))
            layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class DeConvUpsampler(nn.Module):
    """
    Multi-scale upsampler using DeConv (based on WiLoR's DeConvNet)
    Generates multi-scale features from input feature map
    """
    def __init__(self, feat_dim=1280, num_scales=3, upscale=4):
        """
        Args:
            feat_dim: feature dimension (e.g., 1280 for ViT-L)
            num_scales: number of output scales
            upscale: maximum upscale factor (2^upscale)
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_scales = num_scales
        self.upscale = upscale
        
        # First conv to reduce dimension
        self.first_conv = make_conv_layers(
            [feat_dim, feat_dim//2], 
            kernel=1, 
            stride=1, 
            padding=0, 
            bnrelu_final=False
        )
        
        # Deconv layers for different scales
        self.deconv = nn.ModuleList([])
        for i in range(int(math.log2(upscale))+1):
            if i == 0:
                # First scale: feat_dim//2 -> feat_dim//4
                self.deconv.append(make_deconv_layers([feat_dim//2, feat_dim//4]))
            elif i == 1:
                # Second scale: feat_dim//2 -> feat_dim//4 -> feat_dim//8
                self.deconv.append(make_deconv_layers([feat_dim//2, feat_dim//4, feat_dim//8]))
            elif i == 2:
                # Third scale: feat_dim//2 -> feat_dim//4 -> feat_dim//8 -> feat_dim//8
                self.deconv.append(make_deconv_layers([feat_dim//2, feat_dim//4, feat_dim//8, feat_dim//8]))
            else:
                # For more scales, use the same pattern as i==2
                self.deconv.append(make_deconv_layers([feat_dim//2, feat_dim//4, feat_dim//8, feat_dim//8]))
    
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-scale features from input feature map
        
        Args:
            features: (B, D, H, W) input feature map F_0
        Returns:
            multi_scale_features: list of (B, D_i, H_i, W_i) feature maps
                [F_0, F_1, F_2, ...] with increasing resolution
                Note: Output is reversed (high resolution -> low resolution) as in WiLoR
        """
        B, D, H, W = features.shape
        
        # Apply first conv
        img_feat = self.first_conv(features)  # (B, D//2, H, W)
        
        # Collect features at different scales
        face_img_feats = []
        face_img_feats.append(img_feat)  # Original scale (after first_conv)
        
        # Generate upsampled features at different scales
        for i, deconv in enumerate(self.deconv):
            img_feat_i = deconv(img_feat)  # Upsample by 2^i
            face_img_feats.append(img_feat_i)
        
        # Reverse to get high resolution -> low resolution (as in WiLoR)
        # But we want low resolution -> high resolution for our use case
        # So we'll reverse it back
        face_img_feats = face_img_feats[::-1]  # Now: low resolution -> high resolution
        
        # Adjust to match num_scales requirement
        # If we have more scales than needed, take the first num_scales
        # If we have fewer, we'll pad with the last one
        if len(face_img_feats) >= self.num_scales:
            multi_scale_features = face_img_feats[:self.num_scales]
        else:
            # Pad with the last feature map
            multi_scale_features = face_img_feats
            while len(multi_scale_features) < self.num_scales:
                multi_scale_features.append(face_img_feats[-1])
        
        # Ensure all features have the same channel dimension for concatenation
        # We'll use interpolation to match the original dimension if needed
        # But for now, we'll return them as is and let the GCN handle dimension matching
        return multi_scale_features


class DeConvUpsamplerV2(nn.Module):
    """
    Simplified DeConv upsampler that maintains feature dimension
    Uses deconvolution to upsample while keeping the same channel dimension
    """
    def __init__(self, feat_dim=1280, num_scales=3):
        """
        Args:
            feat_dim: feature dimension
            num_scales: number of output scales
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.num_scales = num_scales
        
        # First conv to reduce dimension
        self.first_conv = make_conv_layers(
            [feat_dim, feat_dim//2], 
            kernel=1, 
            stride=1, 
            padding=0, 
            bnrelu_final=False
        )
        
        # Deconv layers for upsampling (maintain dimension)
        self.deconv_layers = nn.ModuleList()
        for i in range(num_scales - 1):  # num_scales - 1 because first scale is original
            # Each deconv upsamples by 2x
            self.deconv_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        in_channels=feat_dim//2,
                        out_channels=feat_dim//2,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        output_padding=0,
                        bias=False
                    ),
                    nn.BatchNorm2d(feat_dim//2),
                    nn.ReLU(inplace=True)
                )
            )
    
    def forward(self, features: torch.Tensor) -> List[torch.Tensor]:
        """
        Generate multi-scale features
        
        Args:
            features: (B, D, H, W) input feature map F_0
        Returns:
            multi_scale_features: list of (B, D//2, H_i, W_i) feature maps
                [F_0, F_1, F_2, ...] with increasing resolution
        """
        B, D, H, W = features.shape
        
        # Apply first conv
        img_feat = self.first_conv(features)  # (B, D//2, H, W)
        
        # Collect features at different scales
        multi_scale_features = []
        multi_scale_features.append(img_feat)  # First scale: original resolution
        
        # Generate upsampled features
        current_feat = img_feat
        for deconv in self.deconv_layers:
            current_feat = deconv(current_feat)  # Upsample by 2x
            multi_scale_features.append(current_feat)
        
        return multi_scale_features



