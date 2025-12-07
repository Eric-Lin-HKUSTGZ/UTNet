"""
UTNet Main Model
整合所有模块：模态融合、ViT主干、NAF上采样、GCN精细化
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List
import numpy as np

from .tokenization import ModalityFusion
from .backbone.vit import ViTBackbone
from .deconv_upsampler import DeConvUpsamplerV2
from .gcn_refinement import GCNRefinement, sample_vertex_features, create_default_adjacency
from ..utils.geometry import perspective_projection
from ..utils.mano_utils import MANOWrapper


class UTNet(nn.Module):
    """
    UTNet: Unified Transformer Network for Hand Pose Estimation
    Single-stage end-to-end training
    """
    def __init__(self, 
                 img_size,
                 patch_size: int = 16,
                 embed_dim: int = 1280,
                 vit_depth: int = 32,
                 vit_num_heads: int = 16,
                 num_hand_joints: int = 15,
                 num_vertices: int = 778,
                 mano_path: Optional[str] = None,
                 mean_params_path: Optional[str] = None,
                 num_scales: int = 3,
                 focal_length: float = 5000.0,
                 joint_rep_type: str = 'aa'):
        """
        Args:
            img_size: input image size, can be int (square) or tuple/list (H, W)
            patch_size: patch size for tokenization
            embed_dim: embedding dimension
            vit_depth: ViT depth
            vit_num_heads: number of attention heads
            num_hand_joints: number of hand joints
            num_vertices: number of MANO vertices
            mano_path: path to MANO model
            mean_params_path: path to MANO mean parameters
            num_scales: number of scales for multi-scale features
            focal_length: focal length for projection
        """
        super().__init__()
        # Handle both int and tuple/list for img_size
        if isinstance(img_size, (int, float)):
            self.img_h = self.img_w = int(img_size)
        elif isinstance(img_size, (tuple, list)):
            self.img_h, self.img_w = int(img_size[0]), int(img_size[1])
        else:
            raise ValueError(f"img_size must be int or tuple/list, got {type(img_size)}")
        
        self.img_size = (self.img_h, self.img_w)  # Store as tuple for compatibility
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_hand_joints = num_hand_joints
        self.num_vertices = num_vertices
        self.focal_length = focal_length
        self.joint_rep_type = joint_rep_type
        
        # Calculate number of patches for rectangular input
        self.patches_h = self.img_h // patch_size
        self.patches_w = self.img_w // patch_size
        self.num_patches = self.patches_h * self.patches_w
        
        # Modality fusion (WiLoR style: uses mano_mean_params.npz for token initialization)
        self.modality_fusion = ModalityFusion(
            img_size, 
            patch_size, 
            embed_dim,
            mean_params_path=mean_params_path,
            num_hand_joints=num_hand_joints,
            joint_rep_type=joint_rep_type
        )
        
        # ViT backbone
        self.vit_backbone = ViTBackbone(
            embed_dim=embed_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            num_patches=self.num_patches,
            num_hand_joints=num_hand_joints,
            joint_rep_type=joint_rep_type,
            mean_params_path=mean_params_path
        )
        
        # DeConv upsampler (based on WiLoR)
        self.deconv_upsampler = DeConvUpsamplerV2(
            feat_dim=embed_dim,
            num_scales=num_scales
        )
        
        # GCN refinement
        # Feature dimension = (embed_dim//2) * num_scales (concatenated from all scales)
        # Note: DeConv reduces dimension to embed_dim//2
        feature_dim = (embed_dim // 2) * num_scales
        self.gcn_refinement = GCNRefinement(
            feature_dim=feature_dim,
            num_vertices=num_vertices
        )
        
        # MANO model - try to load using smplx first (like WiLoR)
        adjacency_set = False
        if mano_path is not None:
            # First, try to use smplx (preferred method, handles all compatibility issues)
            try:
                import smplx
                # Determine if mano_path is a file or directory
                import os.path as osp
                if osp.isdir(mano_path):
                    model_path = mano_path
                else:
                    model_path = osp.dirname(mano_path) if osp.dirname(mano_path) else '.'
                
                # Load MANO model using smplx (similar to WiLoR)
                mano_model = smplx.create(
                    model_path=model_path,
                    model_type='mano',
                    gender='neutral',
                    num_hand_joints=num_hand_joints,
                    use_pca=False,  # Don't use PCA - we provide full 45-dim axis-angle
                    flat_hand_mean=True
                )
                from ..utils.mano_utils import MANOWrapper
                self.mano = MANOWrapper(mano_model)
                # Get faces from smplx model (it handles all compatibility internally)
                faces = torch.from_numpy(mano_model.faces.astype(np.int64))
                self.gcn_refinement.set_adjacency(faces)
                adjacency_set = True
                print(f"Successfully loaded MANO model and faces using smplx.")
            except ImportError:
                # smplx not available, try to load faces only (for GCN adjacency)
                print(f"Note: smplx not available. Loading MANO faces only for GCN adjacency matrix.")
                self.mano = None
            except Exception as e:
                print(f"Warning: Failed to load MANO model with smplx: {e}. Trying to load faces only.")
                self.mano = None
            
            # If smplx failed or not available, try to load faces only (for GCN adjacency)
            if not adjacency_set:
                from ..utils.mano_utils import load_mano_faces
                # Try to load faces directly from pkl file for GCN
                # This is sufficient for GCN refinement (we only need the adjacency matrix)
                try:
                    faces = load_mano_faces(mano_path)
                    self.gcn_refinement.set_adjacency(faces)
                    print(f"Successfully loaded MANO faces for GCN adjacency matrix.")
                    adjacency_set = True
                except Exception as face_error:
                    print(f"Warning: Failed to load MANO faces: {face_error}")
                    print(f"Note: GCN will use default adjacency matrix. MANO model will not be available for forward pass.")
                    print(f"To fix this, you can:")
                    print(f"  1. Install scipy: pip install scipy")
                    print(f"  2. Install smplx (recommended): pip install smplx")
        
        # If adjacency matrix was not set, use a default one
        if not adjacency_set:
            if self.mano is None:
                print(f"Warning: MANO model not available. Using default adjacency matrix for GCN.")
            # Use CPU device for default adjacency (will be moved to correct device during forward)
            default_adj = create_default_adjacency(num_vertices, device=torch.device('cpu'))
            self.gcn_refinement.set_adjacency_from_matrix(default_adj)
            print(f"Successfully set default adjacency matrix for GCN.")

    def forward(self, rgb: torch.Tensor, depth: Optional[torch.Tensor],
                n_i: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            rgb: (B, 3, H, W) RGB image
            depth: (B, 1, H, W) depth image, or None
            n_i: (B,) modality flags
        Returns:
            dict with predictions
        """
        B = rgb.shape[0]
        device = rgb.device
        
        # Crop to rectangular input if needed (similar to WiLoR)
        # If input is square but model expects rectangular, crop from center
        _, C, H, W = rgb.shape
        if H == W and (self.img_h != self.img_w):
            # Input is square but model expects rectangular
            # Crop from center: for (256, 192), crop 32 pixels from left and right
            crop_w = (W - self.img_w) // 2
            rgb = rgb[:, :, :, crop_w:W-crop_w]  # (B, 3, H, W_rect)
            if depth is not None:
                depth = depth[:, :, :, crop_w:W-crop_w]  # (B, 1, H, W_rect)
        
        # Step 1: Tokenization and modality fusion
        tokens = self.modality_fusion(rgb, depth, n_i)  # (B, N+2, D)
        
        # Step 2: ViT backbone + coarse parameter prediction
        vit_output = self.vit_backbone(tokens)
        coarse_mano_params = vit_output['coarse_mano_params']
        coarse_cam = vit_output['coarse_cam']
        coarse_mano_feats = vit_output['coarse_mano_feats']
        img_feat = vit_output['img_feat']  # (B, D, H_p, W_p)
        
        # Step 3: DeConv upsampling - generate multi-scale features
        multi_scale_features = self.deconv_upsampler(img_feat)  # List of (B, D//2, H_i, W_i)
        
        # Step 4: Get coarse mesh from MANO
        if self.mano is not None:
            mano_output = self.mano(
                global_orient=coarse_mano_params['global_orient'],
                hand_pose=coarse_mano_params['hand_pose'],
                betas=coarse_mano_params['betas']
            )
            coarse_vertices = mano_output['vertices']  # (B, V, 3) in meters
            coarse_joints = mano_output['joints']  # (B, J, 3) in meters
            
            # Convert from meters to millimeters to match dataset format
            coarse_vertices = coarse_vertices * 1000.0  # (B, V, 3) in mm
            coarse_joints = coarse_joints * 1000.0  # (B, J, 3) in mm
        else:
            # Fallback: use dummy values if MANO not available
            # Use 21 keypoints to match dataset format
            coarse_vertices = torch.zeros(B, self.num_vertices, 3, device=device)
            coarse_joints = torch.zeros(B, 21, 3, device=device)  # 21 keypoints for Dex-YCB
        
        # Step 5: Project vertices to 2D for feature sampling
        # Project coarse vertices to 2D coordinates at each scale
        vertices_2d_list = []
        for i, feat_map in enumerate(multi_scale_features):
            _, _, H_i, W_i = feat_map.shape
            # Project vertices to this scale
            # Use coarse camera parameters for projection
            vertices_2d = self._project_vertices_to_scale(
                coarse_vertices, coarse_cam, (H_i, W_i)
            )
            vertices_2d_list.append(vertices_2d)
        
        # Step 6: Sample vertex features from multi-scale feature maps
        vertex_features = sample_vertex_features(multi_scale_features, vertices_2d_list)  # (B, V, C_total)
        
        # Step 7: GCN refinement
        # GCN outputs delta_pose in axis-angle format (B, 48) for now
        # TODO: Update GCN to support different joint_rep_type
        delta_pose, delta_shape = self.gcn_refinement(vertex_features)  # (B, 48), (B, 10)
        
        # Step 8: Compute final parameters
        # Add delta to coarse pose representation
        # Note: GCN currently outputs axis-angle (48 dim), so we need to handle conversion
        coarse_pose = coarse_mano_feats['hand_pose']  # (B, npose) where npose = 48 for aa, 96 for 6d
        
        if self.joint_rep_type == 'aa':
            # Both are axis-angle, can add directly
            final_pose = coarse_pose + delta_pose  # (B, 48)
            final_pose_aa = final_pose.reshape(B, 16, 3)
            
            # Import axis-angle to rotation matrix conversion
            from .backbone.vit import aa_to_rotmat
            
            # Convert to rotation matrices
            global_orient_aa = final_pose_aa[:, 0:1].reshape(B, 3)  # (B, 3)
            hand_pose_aa = final_pose_aa[:, 1:].reshape(B * 15, 3)  # (B*15, 3)
            
            final_global_orient = aa_to_rotmat(global_orient_aa).reshape(B, 1, 3, 3)  # (B, 1, 3, 3)
            final_hand_pose = aa_to_rotmat(hand_pose_aa).reshape(B, 15, 3, 3)  # (B, 15, 3, 3)
        else:
            # Coarse pose is 6D, but delta_pose is axis-angle (from GCN)
            # Convert delta_pose to 6D first, then add
            from .backbone.vit import aa_to_rotmat, rot6d_to_rotmat
            import torch.nn.functional as F
            
            # Convert delta_pose (axis-angle) to rotation matrices, then to 6D
            delta_pose_aa = delta_pose.reshape(B, 16, 3)
            delta_rotmats = aa_to_rotmat(delta_pose_aa.reshape(B * 16, 3)).reshape(B, 16, 3, 3)
            
            # Convert coarse_pose (6D) to rotation matrices
            coarse_pose_6d = coarse_pose.reshape(B, 16, 6)
            coarse_rotmats = rot6d_to_rotmat(coarse_pose_6d.reshape(B * 16, 6)).reshape(B, 16, 3, 3)
            
            # Combine rotations: final = delta * coarse
            final_rotmats = torch.matmul(delta_rotmats, coarse_rotmats)  # (B, 16, 3, 3)
            
            # Convert back to 6D
            # Extract first two columns as 6D representation
            final_pose_6d = final_rotmats[:, :, :, :2].reshape(B, 16, 6)  # (B, 16, 6)
            final_pose = final_pose_6d.reshape(B, 96)  # (B, 96)
            
            # For MANO, we still need rotation matrices
            final_global_orient = final_rotmats[:, 0:1]  # (B, 1, 3, 3)
            final_hand_pose = final_rotmats[:, 1:]  # (B, 15, 3, 3)
        
        final_shape = coarse_mano_feats['betas'] + delta_shape  # (B, 10)
        
        final_mano_params = {
            'global_orient': final_global_orient,
            'hand_pose': final_hand_pose,
            'betas': final_shape
        }
        
        # Step 9: MANO decoding for final output
        if self.mano is not None:
            final_mano_output = self.mano(
                global_orient=final_mano_params['global_orient'],
                hand_pose=final_mano_params['hand_pose'],
                betas=final_mano_params['betas']
            )
            final_vertices = final_mano_output['vertices']  # (B, V, 3) in meters
            final_joints = final_mano_output['joints']  # (B, J, 3) in meters
            
            # Convert from meters to millimeters to match dataset format
            final_vertices = final_vertices * 1000.0  # (B, V, 3) in mm
            final_joints = final_joints * 1000.0  # (B, J, 3) in mm
        else:
            final_vertices = coarse_vertices
            final_joints = coarse_joints  # Already 21 keypoints from coarse_joints, already in mm
        
        # Step 10: Project to 2D keypoints
        keypoints_2d = self._project_joints_to_2d(final_joints, coarse_cam)  # (B, J, 2)
        
        return {
            'keypoints_2d': keypoints_2d,
            'keypoints_3d': final_joints,
            'vertices': final_vertices,
            'mano_params': final_mano_params,
            'coarse_mano_params': coarse_mano_params,
            'coarse_cam': coarse_cam,
            'coarse_pose': coarse_mano_feats['hand_pose'],
            'coarse_shape': coarse_mano_feats['betas'],
            'delta_pose': delta_pose,
            'delta_shape': delta_shape
        }

    def _project_vertices_to_scale(self, vertices: torch.Tensor,
                                   cam_params: torch.Tensor,
                                   scale_size: Tuple[int, int]) -> torch.Tensor:
        """
        Project 3D vertices to 2D coordinates at a specific scale (for feature sampling)
        Uses simplified orthogonal projection for feature sampling (not for final 2D prediction)
        
        Args:
            vertices: (B, V, 3) 3D vertices in millimeters
            cam_params: (B, 3) camera parameters [scale, tx, ty]
            scale_size: (H, W) target scale size
        Returns:
            vertices_2d: (B, V, 2) 2D coordinates
        """
        H, W = scale_size
        B = vertices.shape[0]
        
        # Extract camera parameters
        scale = cam_params[:, 0:1]  # (B, 1)
        tx = cam_params[:, 1:2]  # (B, 1)
        ty = cam_params[:, 2:3]  # (B, 1)
        
        # Simplified orthogonal projection for feature sampling
        # Note: This is only used for sampling features, not for final 2D keypoint prediction
        x = vertices[:, :, 0] * scale + W / 2 + tx
        y = vertices[:, :, 1] * scale + H / 2 + ty
        
        return torch.stack([x, y], dim=-1)

    def _project_joints_to_2d(self, joints: torch.Tensor,
                              cam_params: torch.Tensor) -> torch.Tensor:
        """
        Project 3D joints to 2D keypoints using WiLoR-style perspective projection
        Normalized to [-0.5, 0.5] to match dataset format
        
        Args:
            joints: (B, J, 3) 3D joints in millimeters
            cam_params: (B, 3) camera parameters [scale, tx, ty] from model
        Returns:
            keypoints_2d: (B, J, 2) 2D keypoints normalized to [-0.5, 0.5]
        """
        B, J, _ = joints.shape
        device = joints.device
        
        # Step 1: Convert cam_params to camera translation (WiLoR style)
        # cam_params = [scale, tx, ty]
        scale = cam_params[:, 0:1]  # (B, 1)
        tx = cam_params[:, 1:2]  # (B, 1)
        ty = cam_params[:, 2:3]  # (B, 1)
        
        # Convert to camera translation (WiLoR formula)
        # tz = 2 * focal_length / (IMAGE_SIZE * scale)
        focal_length = torch.tensor([self.focal_length], device=device, dtype=joints.dtype).expand(B, 2)  # (B, 2)
        tz = 2 * focal_length[:, 0:1] / (self.img_h * scale + 1e-9)  # (B, 1)
        cam_translation = torch.cat([tx, ty, tz], dim=1)  # (B, 3)
        
        # Step 2: Normalize focal length (WiLoR style)
        # WiLoR divides focal_length by IMAGE_SIZE for normalization
        focal_length_norm = focal_length / self.img_h  # (B, 2)
        
        # Step 3: Camera center (WiLoR default: center of image)
        camera_center = torch.tensor([self.img_w/2.0, self.img_h/2.0], 
                                     device=device, dtype=joints.dtype).expand(B, 2)  # (B, 2)
        
        # Step 4: Convert joints from millimeters to meters (WiLoR uses meters)
        joints_meters = joints / 1000.0  # (B, J, 3) in meters
        
        # Step 5: Apply perspective projection (WiLoR style)
        keypoints_2d_pixel = perspective_projection(
            joints_meters,
            translation=cam_translation,
            focal_length=focal_length_norm,
            camera_center=camera_center
        )  # (B, J, 2) in pixel coordinates
        
        # Step 6: Normalize to [-0.5, 0.5] to match WiLoR dataset format
        # WiLoR uses: keypoints_2d = keypoints_2d / patch_width - 0.5
        keypoints_2d_pixel[:, :, 0] = keypoints_2d_pixel[:, :, 0] / self.img_w - 0.5
        keypoints_2d_pixel[:, :, 1] = keypoints_2d_pixel[:, :, 1] / self.img_h - 0.5
        
        return keypoints_2d_pixel

