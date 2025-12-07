"""
Loss Functions for UTNet
包括2D/3D关键点损失、顶点损失、MANO参数正则、辅助损失
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class Keypoint2DLoss(nn.Module):
    """
    2D Keypoint Reprojection Loss (WiLoR style)
    L_2D = ||J_2D_pred - J_2D_gt||_1 with confidence weighting
    """
    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type: 'l1' or 'l2' loss
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, N, 2] or [B, S, N, 2] containing projected 2D keypoints
                                               (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, N, 3] or [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        # Extract confidence from last dimension
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        # Compute loss with confidence weighting
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1,2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):
    """
    3D Keypoint Loss (WiLoR style: relative to root joint)
    L_3D_joint = ||(J_3D_pred - J_root_pred) - (J_3D_gt - J_root_gt)||_1 with confidence weighting
    This removes global translation and focuses on relative joint positions.
    """
    def __init__(self, loss_type: str = 'l1'):
        """
        Args:
            loss_type: 'l1' or 'l2' loss
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor, pelvis_id: int = 0):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, N, 3] or [B, S, N, 3] containing the predicted 3D keypoints
                                               (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, N, 4] or [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
            pelvis_id (int): Index of root joint (wrist for hand, pelvis for body). Default is 0.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        batch_size = pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        
        # Debug: print statistics (controlled by environment variable)
        import os
        if os.environ.get('UTNET_DEBUG_LOSS', '0') == '1':
            print(f'[Debug 3D Loss]')
            print(f'  Before centering - Pred mean: {pred_keypoints_3d.mean():.3f}, std: {pred_keypoints_3d.std():.3f}')
            print(f'  Before centering - GT mean: {gt_keypoints_3d[:, :, :-1].mean():.3f}, std: {gt_keypoints_3d[:, :, :-1].std():.3f}')
        
        # Normalize by subtracting root joint (removes global translation)
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(dim=1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(dim=1)
        
        # Extract confidence
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :-1]
        
        # Debug: print after centering
        if os.environ.get('UTNET_DEBUG_LOSS', '0') == '1':
            print(f'  After centering - Pred mean: {pred_keypoints_3d.mean():.3f}, std: {pred_keypoints_3d.std():.3f}')
            print(f'  After centering - GT mean: {gt_keypoints_3d.mean():.3f}, std: {gt_keypoints_3d.std():.3f}')
            per_element_loss = self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)
            print(f'  Per-element loss mean: {per_element_loss.mean():.3f}, max: {per_element_loss.max():.3f}')
        
        # Compute loss with confidence weighting
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(dim=(1,2))
        
        if os.environ.get('UTNET_DEBUG_LOSS', '0') == '1':
            print(f'  Loss value: {loss.sum().item():.3f}')
        
        return loss.sum()


class Vertex3DLoss(nn.Module):
    """
    3D Vertex Loss
    L_3D_vert = ||M_pred - M_gt||_2^2
    """
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, pred_vertices: torch.Tensor,
                gt_vertices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_vertices: (B, V, 3) predicted mesh vertices
            gt_vertices: (B, V, 3) ground truth mesh vertices
        Returns:
            loss: scalar loss value
        """
        return self.loss_fn(pred_vertices, gt_vertices)


class MANOParameterPrior(nn.Module):
    """
    MANO Parameter Prior Loss
    Shape prior: ||β||_2^2 (encourage shape close to zero)
    Pose prior: optional penalty for unreasonable joint angles
    """
    def __init__(self, shape_weight: float = 1.0, pose_weight: float = 0.0):
        """
        Args:
            shape_weight: weight for shape prior
            pose_weight: weight for pose prior (0 to disable)
        """
        super().__init__()
        self.shape_weight = shape_weight
        self.pose_weight = pose_weight

    def forward(self, betas: torch.Tensor, 
                hand_pose: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            betas: (B, 10) shape parameters
            hand_pose: (B, 15, 3, 3) hand pose rotation matrices (optional)
        Returns:
            dict with 'shape_prior' and optionally 'pose_prior'
        """
        losses = {}
        
        # Shape prior: L2 norm
        shape_prior = (betas ** 2).mean()
        losses['shape_prior'] = self.shape_weight * shape_prior
        
        # Pose prior (optional): penalize large rotations
        if self.pose_weight > 0 and hand_pose is not None:
            # Convert rotation matrices to axis-angle and penalize large angles
            # Simplified: use Frobenius norm deviation from identity
            identity = torch.eye(3, device=hand_pose.device).unsqueeze(0).unsqueeze(0)
            pose_deviation = ((hand_pose - identity) ** 2).sum(dim=(-2, -1)).mean()
            losses['pose_prior'] = self.pose_weight * pose_deviation
        
        return losses


class AuxiliaryLoss(nn.Module):
    """
    Auxiliary Loss for Coarse Predictions
    L_aux = λ_aux(||θ^c - θ_gt||_2^2 + ||β^c - β_gt||_2^2)
    """
    def __init__(self, weight: float = 0.1):
        """
        Args:
            weight: weight for auxiliary loss
        """
        super().__init__()
        self.weight = weight
        self.pose_loss_fn = nn.MSELoss(reduction='mean')
        self.shape_loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, coarse_pose: torch.Tensor, coarse_shape: torch.Tensor,
                gt_pose: torch.Tensor, gt_shape: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coarse_pose: (B, 48) coarse pose prediction
            coarse_shape: (B, 10) coarse shape prediction
            gt_pose: (B, 48) ground truth pose
            gt_shape: (B, 10) ground truth shape
        Returns:
            loss: scalar auxiliary loss
        """
        pose_loss = self.pose_loss_fn(coarse_pose, gt_pose)
        shape_loss = self.shape_loss_fn(coarse_shape, gt_shape)
        return self.weight * (pose_loss + shape_loss)


class UTNetLoss(nn.Module):
    """
    Combined Loss for UTNet
    L = w_2D*L_2D + w_3D_j*L_3D_joint + w_3D_v*L_3D_vert + w_prior*L_prior + w_aux*L_aux
    """
    def __init__(self, 
                 w_2d: float = 1.0,
                 w_3d_joint: float = 1.0,
                 w_3d_vert: float = 0.5,
                 w_prior: float = 0.01,
                 w_aux: float = 0.1,
                 use_vertex_loss: bool = True,
                 use_aux_loss: bool = True):
        """
        Args:
            w_2d: weight for 2D keypoint loss
            w_3d_joint: weight for 3D joint loss
            w_3d_vert: weight for 3D vertex loss
            w_prior: weight for MANO parameter prior
            w_aux: weight for auxiliary loss
            use_vertex_loss: whether to use vertex loss
            use_aux_loss: whether to use auxiliary loss
        """
        super().__init__()
        self.w_2d = w_2d
        self.w_3d_joint = w_3d_joint
        self.w_3d_vert = w_3d_vert
        self.w_prior = w_prior
        self.w_aux = w_aux
        self.use_vertex_loss = use_vertex_loss
        self.use_aux_loss = use_aux_loss
        
        # Initialize loss modules (WiLoR style)
        self.loss_2d = Keypoint2DLoss(loss_type='l1')  # WiLoR uses L1 for 2D
        self.loss_3d_joint = Keypoint3DLoss(loss_type='l1')  # WiLoR uses L1 for 3D
        if use_vertex_loss:
            self.loss_3d_vert = Vertex3DLoss()
        self.loss_prior = MANOParameterPrior()
        if use_aux_loss:
            self.loss_aux = AuxiliaryLoss(weight=w_aux)

    def forward(self, predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute total loss
        
        Args:
            predictions: dict containing:
                - keypoints_2d: (B, N, 2) predicted 2D keypoints
                - keypoints_3d: (B, N, 3) predicted 3D keypoints
                - vertices: (B, V, 3) predicted vertices (optional)
                - mano_params: dict with 'betas', 'hand_pose'
                - coarse_pose: (B, 48) coarse pose (optional)
                - coarse_shape: (B, 10) coarse shape (optional)
            targets: dict containing:
                - keypoints_2d: (B, N, 2) ground truth 2D keypoints
                - keypoints_3d: (B, N, 3) ground truth 3D keypoints
                - vertices: (B, V, 3) ground truth vertices (optional)
                - mano_pose: (B, 48) ground truth pose (optional)
                - mano_shape: (B, 10) ground truth shape (optional)
        Returns:
            dict with individual losses and total loss
        """
        losses = {}
        
        # 2D keypoint loss
        if 'keypoints_2d' in predictions and 'keypoints_2d' in targets:
            if targets['keypoints_2d'] is not None:
                loss_2d = self.loss_2d(predictions['keypoints_2d'], targets['keypoints_2d'])
                losses['loss_2d'] = loss_2d
                total_loss = self.w_2d * loss_2d
            else:
                total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        else:
            total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        
        # 3D joint loss
        if 'keypoints_3d' in predictions and 'keypoints_3d' in targets:
            if targets['keypoints_3d'] is not None:
                loss_3d_joint = self.loss_3d_joint(predictions['keypoints_3d'], targets['keypoints_3d'])
                losses['loss_3d_joint'] = loss_3d_joint
                total_loss = total_loss + self.w_3d_joint * loss_3d_joint
        
        # 3D vertex loss
        if self.use_vertex_loss and 'vertices' in predictions and 'vertices' in targets:
            if targets['vertices'] is not None:
                loss_3d_vert = self.loss_3d_vert(predictions['vertices'], targets['vertices'])
                losses['loss_3d_vert'] = loss_3d_vert
                total_loss = total_loss + self.w_3d_vert * loss_3d_vert
        
        # MANO parameter prior
        if 'mano_params' in predictions:
            mano_params = predictions['mano_params']
            prior_losses = self.loss_prior(
                mano_params.get('betas'),
                mano_params.get('hand_pose')
            )
            for key, val in prior_losses.items():
                losses[key] = val
                total_loss = total_loss + self.w_prior * val
        
        # Auxiliary loss
        if self.use_aux_loss and 'coarse_pose' in predictions and 'coarse_shape' in predictions:
            if 'mano_pose' in targets and 'mano_shape' in targets:
                if targets['mano_pose'] is not None and targets['mano_shape'] is not None:
                    loss_aux = self.loss_aux(
                        predictions['coarse_pose'],
                        predictions['coarse_shape'],
                        targets['mano_pose'],
                        targets['mano_shape']
                    )
                    losses['loss_aux'] = loss_aux
                    total_loss = total_loss + loss_aux
        
        losses['total_loss'] = total_loss
        return losses

