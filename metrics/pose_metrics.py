"""
Pose evaluation metrics for hand pose estimation
Based on WiLoR's pose_utils.py
"""
import torch
import numpy as np
from typing import Tuple, Optional


def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrustes problem.
    
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """
    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)  # (B, 3, N)
    S2 = S2.permute(0, 2, 1)  # (B, 3, N)
    
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)  # (B, 3, 1)
    mu2 = S2.mean(dim=2, keepdim=True)  # (B, 3, 1)
    X1 = S1 - mu1  # (B, 3, N)
    X2 = S2 - mu2  # (B, 3, N)

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1, 2))  # (B,)

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))  # (B, 3, 3)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    try:
        U, s, V = torch.svd(K)
    except:
        # Fallback for older PyTorch versions
        U, s, Vh = torch.linalg.svd(K)
        V = Vh.permute(0, 2, 1)
    
    Vh = V.permute(0, 2, 1)  # (B, 3, 3)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device, dtype=U.dtype).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, 3)
    # Compute determinant
    try:
        det = torch.linalg.det(torch.matmul(U, Vh))
    except:
        # Fallback for older PyTorch versions
        UVh = torch.matmul(U, Vh)
        det = torch.det(UVh)
    Z[:, -1, -1] *= torch.sign(det)

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))  # (B, 3, 3)

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)  # (B,)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)  # (B, 1, 1)

    # 6. Recover translation.
    t = mu2 - scale * torch.matmul(R, mu1)  # (B, 3, 1)

    # 7. Apply transformation.
    S1_hat = scale * torch.matmul(R, S1) + t  # (B, 3, N)

    return S1_hat.permute(0, 2, 1)  # (B, N, 3)


def compute_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> np.ndarray:
    """
    Compute Mean Per Joint Position Error (MPJPE) in mm.
    
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3) or (N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3) or (N, 3).
    Returns:
        (np.ndarray): MPJPE in mm, shape (B,) or scalar.
    """
    # Handle single sample case
    if pred_joints.dim() == 2:
        pred_joints = pred_joints.unsqueeze(0)
        gt_joints = gt_joints.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Compute Euclidean distance for each joint
    joint_errors = torch.sqrt(((pred_joints - gt_joints) ** 2).sum(dim=-1))  # (B, N)
    
    # Mean over joints
    mpjpe = joint_errors.mean(dim=-1)  # (B,)
    
    # Convert to mm and numpy
    mpjpe_mm = (1000 * mpjpe).cpu().numpy()
    
    if squeeze_output:
        return mpjpe_mm[0]
    return mpjpe_mm


def compute_pa_mpjpe(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> np.ndarray:
    """
    Compute Procrustes Aligned Mean Per Joint Position Error (PA-MPJPE) in mm.
    This is also called Reconstruction Error (RE).
    
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3) or (N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3) or (N, 3).
    Returns:
        (np.ndarray): PA-MPJPE in mm, shape (B,) or scalar.
    """
    # Handle single sample case
    if pred_joints.dim() == 2:
        pred_joints = pred_joints.unsqueeze(0)
        gt_joints = gt_joints.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Apply Procrustes alignment
    pred_joints_aligned = compute_similarity_transform(pred_joints, gt_joints)  # (B, N, 3)
    
    # Compute Euclidean distance after alignment
    joint_errors = torch.sqrt(((pred_joints_aligned - gt_joints) ** 2).sum(dim=-1))  # (B, N)
    
    # Mean over joints
    pa_mpjpe = joint_errors.mean(dim=-1)  # (B,)
    
    # Convert to mm and numpy
    pa_mpjpe_mm = (1000 * pa_mpjpe).cpu().numpy()
    
    if squeeze_output:
        return pa_mpjpe_mm[0]
    return pa_mpjpe_mm


def compute_joint_errors(pred_joints: torch.Tensor, gt_joints: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute both MPJPE and PA-MPJPE.
    
    Args:
        pred_joints (torch.Tensor): Predicted 3D joints of shape (B, N, 3) or (N, 3).
        gt_joints (torch.Tensor): Ground truth 3D joints of shape (B, N, 3) or (N, 3).
    Returns:
        Tuple[np.ndarray, np.ndarray]: (MPJPE in mm, PA-MPJPE in mm)
    """
    mpjpe = compute_mpjpe(pred_joints, gt_joints)
    pa_mpjpe = compute_pa_mpjpe(pred_joints, gt_joints)
    return mpjpe, pa_mpjpe

