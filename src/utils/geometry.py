"""
Geometry Utilities
用于投影、坐标转换等
"""
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def perspective_projection(points: torch.Tensor,
                          translation: torch.Tensor,
                          focal_length: torch.Tensor,
                          camera_center: Optional[torch.Tensor] = None,
                          rotation: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Perspective projection of 3D points to 2D (WiLoR style)
    
    Args:
        points: (B, N, 3) 3D points
        translation: (B, 3) camera translation [tx, ty, tz]
        focal_length: (B, 2) focal length [fx, fy] (normalized or in pixels)
        camera_center: (B, 2) camera center [cx, cy], optional (default: zeros)
        rotation: (B, 3, 3) camera rotation, optional (default: identity)
    Returns:
        projected_points: (B, N, 2) 2D projected points
    """
    batch_size = points.shape[0]
    device = points.device
    dtype = points.dtype
    
    if camera_center is None:
        camera_center = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    if rotation is None:
        rotation = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Populate intrinsic camera matrix K (WiLoR style)
    K = torch.zeros([batch_size, 3, 3], device=device, dtype=dtype)
    K[:, 0, 0] = focal_length[:, 0]  # fx
    K[:, 1, 1] = focal_length[:, 1]  # fy
    K[:, 2, 2] = 1.0
    K[:, :-1, -1] = camera_center  # [cx, cy] in last column
    
    # Transform points: apply rotation first, then translation
    points = torch.einsum('bij,bkj->bki', rotation, points)  # (B, N, 3)
    points = points + translation.unsqueeze(1)  # (B, N, 3)
    
    # Apply perspective distortion (divide by Z)
    # Avoid division by zero
    z_coords = points[:, :, 2:3]  # (B, N, 1)
    z_coords_safe = torch.where(z_coords == 0, torch.ones_like(z_coords) * 1e-8, z_coords)
    projected_points = points / z_coords_safe  # (B, N, 3)
    
    # Apply camera intrinsics: K @ projected_points
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)  # (B, N, 3)
    
    # Return only x, y coordinates
    return projected_points[:, :, :-1]  # (B, N, 2)


def project_vertices_to_2d(vertices: torch.Tensor,
                          cam_params: torch.Tensor,
                          img_size: Tuple[int, int]) -> torch.Tensor:
    """
    Project 3D vertices to 2D image coordinates
    
    Args:
        vertices: (B, V, 3) 3D vertices
        cam_params: (B, 3) camera parameters [scale, tx, ty]
        img_size: (H, W) image size
    Returns:
        vertices_2d: (B, V, 2) 2D coordinates
    """
    B, V, _ = vertices.shape
    H, W = img_size
    
    # Extract camera parameters
    scale = cam_params[:, 0:1]  # (B, 1)
    tx = cam_params[:, 1:2]  # (B, 1)
    ty = cam_params[:, 2:3]  # (B, 1)
    
    # Project
    x = vertices[:, :, 0] * scale + W / 2 + tx
    y = vertices[:, :, 1] * scale + H / 2 + ty
    
    return torch.stack([x, y], dim=-1)

