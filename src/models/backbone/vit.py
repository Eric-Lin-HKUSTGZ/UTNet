"""
ViT Backbone for UTNet
Modified to accept fused token sequence (Camera + MANO + Image tokens)
参考 WiLoR/wilor/models/backbones/vit.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from typing import Tuple, Dict, Optional


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to rotation matrix"""
    x = x.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def aa_to_rotmat(theta: torch.Tensor):
    """
    Convert axis-angle representation to rotation matrix.
    Args:
        theta (torch.Tensor): Tensor of shape (B, 3) containing axis-angle representations.
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat_to_rotmat(quat)


def quat_to_rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion representation to rotation matrix.
    Args:
        quat (torch.Tensor) of shape (B, 4); 4 <===> (w, x, y, z).
    Returns:
        torch.Tensor: Corresponding rotation matrices with shape (B, 3, 3).
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]
    
    B = quat.size(0)
    
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTBackbone(nn.Module):
    """
    ViT Backbone for UTNet
    Accepts fused token sequence: [Camera Token, MANO Token, Image Tokens]
    Outputs: coarse MANO parameters and image features
    """
    def __init__(self, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.55,
                 num_patches=192, num_hand_joints=15, joint_rep_type='6d',
                 mean_params_path=None, use_checkpoint=False):
        """
        Args:
            embed_dim: embedding dimension
            depth: number of transformer blocks
            num_heads: number of attention heads
            mlp_ratio: MLP ratio
            qkv_bias: whether to use bias in QKV projection
            drop_rate: dropout rate
            attn_drop_rate: attention dropout rate
            drop_path_rate: drop path rate
            num_patches: number of image patches
            num_hand_joints: number of hand joints (15 for MANO)
            joint_rep_type: '6d' or 'aa' for joint representation
            mean_params_path: path to MANO mean parameters
            use_checkpoint: whether to use gradient checkpointing
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.num_hand_joints = num_hand_joints
        self.joint_rep_type = joint_rep_type
        self.use_checkpoint = use_checkpoint
        
        # Joint representation
        self.joint_rep_dim = {'6d': 6, 'aa': 3}[joint_rep_type]
        npose = self.joint_rep_dim * (num_hand_joints + 1)  # +1 for global orientation
        
        # Load mean parameters
        if mean_params_path is not None:
            mean_params = np.load(mean_params_path)
            init_cam = torch.from_numpy(mean_params['cam'].astype(np.float32)).unsqueeze(0)
            pose_raw = mean_params['pose'].astype(np.float32)
            init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
            
            # Check if pose format matches joint_rep_type
            expected_size = self.joint_rep_dim * (num_hand_joints + 1)  # 48 for aa, 96 for 6d
            actual_size = pose_raw.size
            
            if actual_size == expected_size:
                # Format matches, use directly
                init_hand_pose = torch.from_numpy(pose_raw).unsqueeze(0)  # (1, npose)
            elif actual_size == 96 and self.joint_rep_type == 'aa':
                # Pose is 6D (96), but we need axis-angle (48)
                # Convert 6D to axis-angle via rotation matrices
                pose_6d = torch.from_numpy(pose_raw).reshape(1, num_hand_joints + 1, 6)  # (1, 16, 6)
                # Convert each joint from 6D to rotation matrix, then to axis-angle
                pose_6d_flat = pose_6d.reshape(-1, 6)  # (16, 6)
                rotmats = rot6d_to_rotmat(pose_6d_flat)  # (16, 3, 3)
                # Convert rotation matrices to axis-angle using Rodrigues formula
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
            # Default initialization
            init_cam = torch.zeros(1, 3)
            init_hand_pose = torch.zeros(1, npose)
            init_betas = torch.zeros(1, 10)
        
        self.register_buffer('init_cam', init_cam)
        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)
        
        # Decoders for coarse parameters (WiLoR style)
        # Pose decoder: outputs per-joint representation
        # Input: (B, 16, D) -> Output: (B, 16, joint_rep_dim)
        self.dec_pose = nn.Linear(embed_dim, self.joint_rep_dim)  # Per-joint decoder
        
        # Shape decoder: (B, D) -> (B, 10)
        self.dec_shape = nn.Linear(embed_dim, 10)
        
        # Camera decoder: (B, D) -> (B, 3)
        self.dec_cam = nn.Linear(embed_dim, 3)
        
        # Initialize decoders
        nn.init.xavier_uniform_(self.dec_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dec_shape.weight, gain=0.01)
        nn.init.xavier_uniform_(self.dec_cam.weight, gain=0.01)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=nn.LayerNorm
            )
            for i in range(depth)
        ])
        
        self.last_norm = nn.LayerNorm(embed_dim)
        
        # Calculate feature map size
        # Support both square and rectangular patches
        # Try to infer spatial dimensions from num_patches
        # For rectangular: patches_h * patches_w = num_patches
        # We'll calculate this from the actual patch resolution
        # For now, we'll use a flexible approach that works with rectangular inputs
        # The actual H_p and W_p will be determined from the input tokens
        # Store num_patches for reshaping
        self.num_patches = num_patches

    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            tokens: (B, N+18, D) fused tokens (WiLoR style)
                   tokens[:, 0:16] = Pose Tokens (16 tokens, one per joint + global orientation)
                   tokens[:, 16] = Shape Token (1 token)
                   tokens[:, 17] = Cam Token (1 token)
                   tokens[:, 18:] = Image Tokens (N patches)
        Returns:
            dict containing:
                - coarse_mano_params: dict with 'global_orient', 'hand_pose', 'betas'
                - coarse_cam: (B, 3) camera parameters
                - coarse_mano_feats: dict with raw features
                - img_feat: (B, D, H_p, W_p) image features reshaped to feature map
        """
        B = tokens.shape[0]
        
        # Process through transformer blocks
        x = tokens
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        
        x = self.last_norm(x)
        
        # Extract tokens (WiLoR style)
        num_pose_tokens = self.num_hand_joints + 1  # 16 tokens (15 joints + 1 global orientation)
        pose_feat = x[:, :num_pose_tokens]  # (B, 16, D) Pose Tokens
        shape_feat = x[:, num_pose_tokens:num_pose_tokens+1]  # (B, 1, D) Shape Token
        cam_feat = x[:, num_pose_tokens+1:num_pose_tokens+2]  # (B, 1, D) Cam Token
        img_tokens = x[:, num_pose_tokens+2:]  # (B, N, D) Image Tokens
        
        # Decode coarse parameters (WiLoR style)
        # Pose: decpose outputs per-joint representation, then reshape
        # pose_feat: (B, 16, D) -> decpose -> (B, 16, joint_rep_dim) -> reshape -> (B, npose)
        coarse_pose = self.dec_pose(pose_feat).reshape(B, -1) + self.init_hand_pose  # (B, npose)
        
        # Shape: shape_feat: (B, 1, D) -> squeeze -> (B, D) -> decshape -> (B, 10)
        coarse_shape = self.dec_shape(shape_feat.squeeze(1)) + self.init_betas  # (B, 10)
        
        # Camera: cam_feat: (B, 1, D) -> squeeze -> (B, D) -> deccam -> (B, 3)
        coarse_cam = self.dec_cam(cam_feat.squeeze(1)) + self.init_cam  # (B, 3)
        
        # Convert pose to rotation matrices
        # Reshape coarse_pose based on joint representation type
        if self.joint_rep_type == 'aa':
            # Axis-angle: (B, 48) -> (B, 16, 3)
            coarse_pose_aa = coarse_pose.view(B, 16, 3)
            global_orient_aa = coarse_pose_aa[:, 0:1].reshape(B, 3)  # (B, 3)
            hand_pose_aa = coarse_pose_aa[:, 1:].reshape(B * 15, 3)  # (B*15, 3)
            
            # Convert to rotation matrices
            global_orient_rot = aa_to_rotmat(global_orient_aa).reshape(B, 1, 3, 3)  # (B, 1, 3, 3)
            hand_pose_rot = aa_to_rotmat(hand_pose_aa).reshape(B, 15, 3, 3)  # (B, 15, 3, 3)
        else:
            # 6D rotation: (B, 96) -> (B, 16, 6)
            coarse_pose_6d = coarse_pose.view(B, 16, 6)
            global_orient_6d = coarse_pose_6d[:, 0:1].reshape(B, 6)  # (B, 6)
            hand_pose_6d = coarse_pose_6d[:, 1:].reshape(B * 15, 6)  # (B*15, 6)
            
            # Convert to rotation matrices
            global_orient_rot = rot6d_to_rotmat(global_orient_6d).reshape(B, 1, 3, 3)  # (B, 1, 3, 3)
            hand_pose_rot = rot6d_to_rotmat(hand_pose_6d).reshape(B, 15, 3, 3)  # (B, 15, 3, 3)
        
        coarse_mano_params = {
            'global_orient': global_orient_rot,
            'hand_pose': hand_pose_rot,
            'betas': coarse_shape
        }
        
        coarse_mano_feats = {
            'hand_pose': coarse_pose,
            'betas': coarse_shape,
            'cam': coarse_cam
        }
        
        # Reshape image tokens to feature map
        # (B, N, D) -> (B, H_p, W_p, D) -> (B, D, H_p, W_p)
        # For rectangular input, we need to infer H_p and W_p from num_patches
        # Common cases: 16x12=192 (rectangular) or 16x16=256 (square)
        # We'll try to infer from num_patches
        import math
        H_p = int(math.sqrt(self.num_patches))
        W_p = self.num_patches // H_p
        # If not perfect square, try to find better factorization
        if H_p * W_p != self.num_patches:
            # Find factors that are closer together
            best_diff = float('inf')
            best_h, best_w = H_p, W_p
            for h in range(1, int(math.sqrt(self.num_patches)) + 1):
                if self.num_patches % h == 0:
                    w = self.num_patches // h
                    diff = abs(h - w)
                    if diff < best_diff:
                        best_diff = diff
                        best_h, best_w = h, w
            H_p, W_p = best_h, best_w
        
        img_feat = img_tokens.reshape(B, H_p, W_p, self.embed_dim).permute(0, 3, 1, 2)  # (B, D, H_p, W_p)
        
        return {
            'coarse_mano_params': coarse_mano_params,
            'coarse_cam': coarse_cam,
            'coarse_mano_feats': coarse_mano_feats,
            'img_feat': img_feat
        }

