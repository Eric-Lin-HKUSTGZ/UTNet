"""
Depth Token Embedding Module
单层卷积（kernel=stride=patch_size）得到 T_depth^{aux}
实现Depth placeholder token T_depth^{plh}
"""
import torch
import torch.nn as nn
from typing import Optional


class DepthPatchEmbed(nn.Module):
    """
    Depth Image to Patch Embedding
    (B, 1, H, W) -> (B, N, D)
    Single conv layer with kernel=stride=patch_size
    Supports rectangular input images
    """
    def __init__(self, img_size, patch_size: int = 16, 
                 embed_dim: int = 1280, norm_layer: Optional[nn.Module] = None):
        """
        Args:
            img_size: input image size, can be int (square) or tuple/list (H, W)
            patch_size: patch size (should match RGB patch size)
            embed_dim: embedding dimension (should match RGB embed_dim)
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
        
        # Single conv layer: kernel=stride=patch_size
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.LayerNorm(embed_dim)
        
        # Calculate number of patches
        self.patches_h = self.img_h // patch_size
        self.patches_w = self.img_w // patch_size
        self.num_patches = self.patches_h * self.patches_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) depth image
        Returns:
            tokens: (B, N, D) depth tokens
        """
        B, C, H, W = x.shape
        assert C == 1, f"Expected 1 channel for depth, got {C}"
        assert H == self.img_h and W == self.img_w, f"Input size {H}x{W} != {self.img_h}x{self.img_w}"
        
        # Patch embedding
        x = self.proj(x)  # (B, D, H_p, W_p)
        
        # Flatten and transpose: (B, D, H_p, W_p) -> (B, H_p*W_p, D)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # Normalize
        x = self.norm(x)
        
        return x


class DepthPlaceholder(nn.Module):
    """
    Depth Placeholder Token
    T_depth^{plh}: learnable placeholder when depth is not available
    """
    def __init__(self, num_patches: int, embed_dim: int):
        """
        Args:
            num_patches: number of patches
            embed_dim: embedding dimension
        """
        super().__init__()
        self.placeholder = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self._init_placeholder()

    def _init_placeholder(self):
        """Initialize placeholder token"""
        nn.init.normal_(self.placeholder, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Args:
            batch_size: batch size
        Returns:
            placeholder tokens: (B, N, D)
        """
        return self.placeholder.expand(batch_size, -1, -1)


class DepthEmbedding(nn.Module):
    """
    Complete Depth embedding: patch embedding + placeholder
    Supports rectangular input images
    """
    def __init__(self, img_size, patch_size: int = 16, embed_dim: int = 1280):
        """
        Args:
            img_size: input image size, can be int (square) or tuple/list (H, W)
            patch_size: patch size
            embed_dim: embedding dimension
        """
        super().__init__()
        self.patch_embed = DepthPatchEmbed(img_size, patch_size, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.placeholder = DepthPlaceholder(num_patches, embed_dim)

    def forward(self, depth: Optional[torch.Tensor], batch_size: int) -> torch.Tensor:
        """
        Args:
            depth: (B, 1, H, W) depth image, or None if not available
            batch_size: batch size (used when depth is None)
        Returns:
            tokens: (B, N, D) depth tokens or placeholder tokens
        """
        if depth is not None:
            return self.patch_embed(depth)
        else:
            return self.placeholder(batch_size)


