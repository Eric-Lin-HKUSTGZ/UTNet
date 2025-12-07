"""
RGB Token Embedding Module
Patch切片 + 线性embedding + 位置编码
"""
import torch
import torch.nn as nn
import math
from typing import Tuple, Optional


class RGBPatchEmbed(nn.Module):
    """
    RGB Image to Patch Embedding
    (B, 3, H, W) -> (B, N, D)
    Supports rectangular input images
    """
    def __init__(self, img_size, patch_size: int = 16, 
                 embed_dim: int = 1280, norm_layer: Optional[nn.Module] = None):
        """
        Args:
            img_size: input image size, can be int (square) or tuple/list (H, W)
            patch_size: patch size
            embed_dim: embedding dimension
            norm_layer: normalization layer
        """
        super().__init__()
        # Handle both int and tuple/list for img_size
        if isinstance(img_size, (int, float)):
            self.img_h = self.img_w = int(img_size)
        elif isinstance(img_size, (tuple, list)):
            self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
        else:
            raise ValueError(f"img_size must be int or tuple/list, got {type(img_size)}")
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding: Conv2d with kernel=stride=patch_size
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.LayerNorm(embed_dim)
        
        # Calculate number of patches
        self.patches_h = self.img_h // patch_size
        self.patches_w = self.img_w // patch_size
        self.num_patches = self.patches_h * self.patches_w
        self.patches_resolution = (self.patches_h, self.patches_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image
        Returns:
            tokens: (B, N, D) where N = num_patches
        """
        B, C, H, W = x.shape
        assert H == self.img_h and W == self.img_w, f"Input size {H}x{W} != {self.img_h}x{self.img_w}"
        
        # Patch embedding
        x = self.proj(x)  # (B, D, H_p, W_p)
        H_p, W_p = x.shape[2], x.shape[3]
        
        # Flatten and transpose: (B, D, H_p, W_p) -> (B, H_p*W_p, D)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Normalize
        x = self.norm(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for patches
    """
    def __init__(self, num_patches: int, embed_dim: int):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self._init_pos_embed()

    def _init_pos_embed(self):
        """Initialize positional embedding"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) token embeddings
        Returns:
            x with positional encoding added
        """
        return x + self.pos_embed


class RGBEmbedding(nn.Module):
    """
    Complete RGB embedding: patch embedding + positional encoding
    Supports rectangular input images
    """
    def __init__(self, img_size, patch_size: int = 16, 
                 embed_dim: int = 1280, use_pos_embed: bool = True):
        """
        Args:
            img_size: input image size, can be int (square) or tuple/list (H, W)
            patch_size: patch size
            embed_dim: embedding dimension
            use_pos_embed: whether to use positional encoding
        """
        super().__init__()
        self.patch_embed = RGBPatchEmbed(img_size, patch_size, embed_dim)
        self.use_pos_embed = use_pos_embed
        if use_pos_embed:
            self.pos_embed = PositionalEncoding(self.patch_embed.num_patches, embed_dim)
        else:
            self.pos_embed = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image
        Returns:
            tokens: (B, N, D) RGB tokens with positional encoding
        """
        x = self.patch_embed(x)  # (B, N, D)
        x = self.pos_embed(x)  # Add positional encoding
        return x


