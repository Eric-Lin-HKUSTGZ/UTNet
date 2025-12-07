"""
Modality Fusion Module
T_input = T_rgb + (n_i * T_depth^{aux} + (1-n_i) * T_depth^{plh})
使用mano_mean_params.npz初始化pose_tokens, shape_tokens, cam_tokens (WiLoR风格)
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from .rgb_embed import RGBEmbedding
from .depth_embed import DepthEmbedding


class ModalityFusion(nn.Module):
    """
    Modality Fusion Module
    Fuses RGB and Depth tokens with random modality sampling
    Uses mano_mean_params.npz to initialize pose_tokens, shape_tokens, cam_tokens (WiLoR style)
    Supports rectangular input images
    """
    def __init__(self, img_size, patch_size: int = 16, embed_dim: int = 1280,
                 mean_params_path: Optional[str] = None, num_hand_joints: int = 15,
                 joint_rep_type: str = 'aa'):
        """
        Args:
            img_size: input image size, can be int (square) or tuple/list (H, W)
            patch_size: patch size
            embed_dim: embedding dimension
            mean_params_path: path to mano_mean_params.npz
            num_hand_joints: number of hand joints (15 for MANO)
            joint_rep_type: 'aa' (axis-angle) or '6d' (6D rotation)
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_hand_joints = num_hand_joints
        self.joint_rep_type = joint_rep_type
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[joint_rep_type]
        
        # RGB and Depth embeddings
        self.rgb_embed = RGBEmbedding(img_size, patch_size, embed_dim)
        self.depth_embed = DepthEmbedding(img_size, patch_size, embed_dim)
        
        # Load mean parameters from mano_mean_params.npz
        if mean_params_path is not None:
            mean_params = np.load(mean_params_path)
            init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)  # (1, 3)
            pose_raw = mean_params['pose'].astype(np.float32)
            init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)  # (1, 10)
            
            # Check if pose format matches joint_rep_type
            expected_size = self.joint_rep_dim * (num_hand_joints + 1)  # 48 for aa, 96 for 6d
            actual_size = pose_raw.size
            
            if actual_size == expected_size:
                # Format matches, use directly
                init_hand_pose = torch.from_numpy(pose_raw).unsqueeze(0)  # (1, npose)
            elif actual_size == 96 and self.joint_rep_type == 'aa':
                # Pose is 6D (96), but we need axis-angle (48)
                # Convert 6D to axis-angle via rotation matrices
                from ..backbone.vit import rot6d_to_rotmat
                pose_6d = torch.from_numpy(pose_raw).reshape(1, num_hand_joints + 1, 6)  # (1, 16, 6)
                # Convert each joint from 6D to rotation matrix, then to axis-angle
                pose_6d_flat = pose_6d.reshape(-1, 6)  # (16, 6)
                rotmats = rot6d_to_rotmat(pose_6d_flat)  # (16, 3, 3)
                # Convert rotation matrices to axis-angle using Rodrigues formula
                # R = I + sin(θ) * K + (1-cos(θ)) * K^2, where K is skew-symmetric
                # trace(R) = 1 + 2*cos(θ), so θ = arccos((trace(R) - 1) / 2)
                trace = rotmats[:, 0, 0] + rotmats[:, 1, 1] + rotmats[:, 2, 2]
                angle = torch.acos(torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0))
                # Axis: (R - R^T) / (2*sin(θ))
                sin_angle = torch.sin(angle)
                # Avoid division by zero
                sin_angle = torch.where(sin_angle.abs() < 1e-6, torch.ones_like(sin_angle) * 1e-6, sin_angle)
                axis_x = (rotmats[:, 2, 1] - rotmats[:, 1, 2]) / (2.0 * sin_angle)
                axis_y = (rotmats[:, 0, 2] - rotmats[:, 2, 0]) / (2.0 * sin_angle)
                axis_z = (rotmats[:, 1, 0] - rotmats[:, 0, 1]) / (2.0 * sin_angle)
                axis = torch.stack([axis_x, axis_y, axis_z], dim=1)  # (16, 3)
                pose_aa_flat = axis * angle.unsqueeze(1)  # (16, 3)
                init_hand_pose = pose_aa_flat.reshape(1, -1)  # (1, 48)
            elif actual_size == 48 and self.joint_rep_type == '6d':
                # Pose is axis-angle (48), but we need 6D (96)
                # Convert axis-angle to 6D via rotation matrices
                from ..backbone.vit import aa_to_rotmat
                pose_aa = torch.from_numpy(pose_raw).reshape(1, num_hand_joints + 1, 3)  # (1, 16, 3)
                # Convert each joint from axis-angle to rotation matrix, then to 6D
                pose_aa_flat = pose_aa.reshape(-1, 3)  # (16, 3)
                rotmats = aa_to_rotmat(pose_aa_flat)  # (16, 3, 3)
                # Convert rotation matrices to 6D (first two columns)
                pose_6d_flat = rotmats[:, :, :2].reshape(-1, 6)  # (16, 6)
                init_hand_pose = pose_6d_flat.reshape(1, -1)  # (1, 96)
            else:
                raise ValueError(
                    f"Pose size mismatch: expected {expected_size} for {joint_rep_type}, "
                    f"but got {actual_size} from {mean_params_path}"
                )
        else:
            # Default initialization (zeros)
            npose = self.joint_rep_dim * (num_hand_joints + 1)  # +1 for global orientation
            init_cam = torch.zeros(1, 3)
            init_hand_pose = torch.zeros(1, npose)
            init_betas = torch.zeros(1, 10)
        
        # Register as buffers (non-trainable parameters)
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        
        # Embedding layers to convert mean params to tokens (WiLoR style)
        # pose_emb: (joint_rep_dim,) -> embed_dim (one embedding per joint)
        self.pose_emb = nn.Linear(self.joint_rep_dim, embed_dim)
        # shape_emb: (10,) -> embed_dim
        self.shape_emb = nn.Linear(10, embed_dim)
        # cam_emb: (3,) -> embed_dim
        self.cam_emb = nn.Linear(3, embed_dim)

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor], 
                n_i: torch.Tensor) -> torch.Tensor:
        """
        Fuse RGB and Depth tokens with modality sampling
        Generate pose_tokens, shape_tokens, cam_tokens from mean params (WiLoR style)
        
        Args:
            rgb: (B, 3, H, W) RGB image
            depth: (B, 1, H, W) depth image, or None
            n_i: (B,) modality flags: 0 if depth dropped, 1 if depth kept
        Returns:
            tokens: (B, N+18, D) where:
                - First 16 tokens: pose_tokens (one per joint + global orientation)
                - Next 1 token: shape_token
                - Next 1 token: cam_token
                - Remaining N tokens: image tokens
        """
        B = rgb.shape[0]
        
        # RGB tokens: T_rgb
        T_rgb = self.rgb_embed(rgb)  # (B, N, D)
        
        # Depth tokens: T_depth^{aux} or T_depth^{plh}
        if depth is not None:
            T_depth_aux = self.depth_embed.patch_embed(depth)  # (B, N, D)
            T_depth_plh = self.depth_embed.placeholder(B)  # (B, N, D)
            
            # Modality fusion: n_i * T_depth^{aux} + (1-n_i) * T_depth^{plh}
            # Note: When n_i=0, T_depth_aux is computed but not used in final output.
            # This is expected behavior (depth is dropped), but causes some parameters
            # to not receive gradients. DDP handles this with find_unused_parameters=True.
            n_i_expanded = n_i.view(B, 1, 1)  # (B, 1, 1)
            T_depth = n_i_expanded * T_depth_aux + (1 - n_i_expanded) * T_depth_plh
        else:
            # No depth available, use placeholder
            T_depth = self.depth_embed.placeholder(B)  # (B, N, D)
        
        # Fused input tokens: T_input = T_rgb + T_depth
        T_input = T_rgb + T_depth  # (B, N, D)
        
        # Generate tokens from mean params (WiLoR style)
        # Reshape init_hand_pose: (1, npose) -> (1, num_hand_joints+1, joint_rep_dim)
        # npose = joint_rep_dim * (num_hand_joints + 1)
        pose_tokens = self.pose_emb(
            self.init_hand_pose.reshape(1, self.num_hand_joints + 1, self.joint_rep_dim)
        ).repeat(B, 1, 1)  # (B, num_hand_joints+1, embed_dim)
        
        shape_tokens = self.shape_emb(self.init_betas).unsqueeze(1).repeat(B, 1, 1)  # (B, 1, embed_dim)
        cam_tokens = self.cam_emb(self.init_cam).unsqueeze(1).repeat(B, 1, 1)  # (B, 1, embed_dim)
        
        # Concatenate: [Pose tokens (16), Shape token (1), Cam token (1), Image tokens (N)]
        tokens = torch.cat([pose_tokens, shape_tokens, cam_tokens, T_input], dim=1)  # (B, 18+N, D)
        
        return tokens


