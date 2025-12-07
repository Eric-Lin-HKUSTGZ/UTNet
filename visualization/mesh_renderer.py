"""
3D mesh renderer for hand visualization
Based on WiLoR's mesh_renderer.py
"""
import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
try:
    import pyrender
    import trimesh
    PYRENDER_AVAILABLE = True
except ImportError:
    PYRENDER_AVAILABLE = False
    print("Warning: pyrender not available. Mesh rendering will be disabled.")

import cv2
from typing import Optional, Tuple

from .keypoint_renderer import render_hand_keypoints

# Try to import MANO face loading utility
try:
    import sys
    import os
    # Add UTNet src to path
    utnet_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(utnet_root, 'src'))
    from utils.mano_utils import load_mano_faces
    MANO_UTILS_AVAILABLE = True
except ImportError:
    MANO_UTILS_AVAILABLE = False


def create_raymond_lights():
    """
    Create lighting for mesh rendering.
    """
    if not PYRENDER_AVAILABLE:
        return []
    
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


class MeshRenderer:
    """
    Renderer for MANO hand meshes.
    """
    def __init__(self, 
                 img_res: int = 256,
                 focal_length: float = 5000.0,
                 faces: Optional[np.ndarray] = None):
        """
        Args:
            img_res (int): Image resolution.
            focal_length (float): Camera focal length.
            faces (np.ndarray): Mesh faces of shape (F, 3). If None, will try to load from MANO.
        """
        if not PYRENDER_AVAILABLE:
            raise ImportError("pyrender and trimesh are required for mesh rendering. "
                            "Install with: pip install pyrender trimesh")
        
        self.img_res = img_res
        self.focal_length = focal_length
        self.camera_center = [img_res // 2, img_res // 2]
        self.faces = faces
        
        if self.faces is None:
            # Try to load MANO faces if available
            if MANO_UTILS_AVAILABLE:
                try:
                    # Try to load from default MANO path
                    import os
                    default_mano_path = os.path.join(os.path.dirname(__file__), '../../mano_data/MANO_RIGHT.pkl')
                    if os.path.exists(default_mano_path):
                        self.faces = load_mano_faces(default_mano_path).cpu().numpy()
                        print(f"Loaded MANO faces from {default_mano_path}")
                    else:
                        print("Warning: No faces provided and default MANO path not found. Mesh rendering may not work correctly.")
                        self.faces = np.array([])
                except Exception as e:
                    print(f"Warning: Failed to load MANO faces: {e}. Mesh rendering may not work correctly.")
                    self.faces = np.array([])
            else:
                print("Warning: No faces provided. Mesh rendering may not work correctly.")
                self.faces = np.array([])

    def __call__(self,
                 vertices: np.ndarray,
                 camera_translation: np.ndarray,
                 image: np.ndarray,
                 focal_length: Optional[float] = None,
                 side_view: bool = False,
                 rot_angle: float = 90,
                 baseColorFactor: Tuple[float, float, float, float] = (1.0, 1.0, 0.9, 1.0)) -> np.ndarray:
        """
        Render mesh on image.
        
        Args:
            vertices (np.ndarray): Mesh vertices of shape (V, 3).
            camera_translation (np.ndarray): Camera translation of shape (3,).
            image (np.ndarray): Background image of shape (H, W, 3) with values in [0, 1] or [0, 255].
            focal_length (float, optional): Focal length. If None, uses self.focal_length.
            side_view (bool): Whether to render side view.
            rot_angle (float): Rotation angle for side view.
            baseColorFactor (Tuple): Mesh base color (R, G, B, A).
        Returns:
            (np.ndarray): Rendered image of shape (H, W, 3) with values in [0, 1].
        """
        if not PYRENDER_AVAILABLE:
            return image
        
        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=baseColorFactor)

        camera_translation = camera_translation.copy()
        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        fl = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=fl, fy=fl,
                                           cx=camera_center[0], cy=camera_center[1])
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def visualize_tensorboard(self,
                              vertices: torch.Tensor,
                              camera_translation: torch.Tensor,
                              images: torch.Tensor,
                              pred_keypoints_2d: torch.Tensor,
                              gt_keypoints_2d: Optional[torch.Tensor] = None,
                              focal_length: Optional[float] = None,
                              nrow: int = 5,
                              padding: int = 2) -> torch.Tensor:
        """
        Create visualization grid for TensorBoard.
        
        Args:
            vertices (torch.Tensor): Mesh vertices of shape (B, V, 3).
            camera_translation (torch.Tensor): Camera translation of shape (B, 3).
            images (torch.Tensor): Input images of shape (B, 3, H, W) with values in [0, 1].
            pred_keypoints_2d (torch.Tensor): Predicted 2D keypoints of shape (B, N, 2).
            gt_keypoints_2d (torch.Tensor, optional): Ground truth 2D keypoints of shape (B, N, 2) or (B, N, 3).
            focal_length (float, optional): Focal length.
            nrow (int): Number of images per row in grid.
            padding (int): Padding between images.
        Returns:
            (torch.Tensor): Grid image of shape (3, H_grid, W_grid).
        """
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
        rend_imgs = []
        
        # Prepare keypoints
        pred_keypoints_2d_np = pred_keypoints_2d.cpu().numpy()
        if pred_keypoints_2d_np.shape[2] == 2:
            pred_keypoints_2d_np = np.concatenate([pred_keypoints_2d_np, np.ones((*pred_keypoints_2d_np.shape[:2], 1))], axis=2)
        
        if gt_keypoints_2d is not None:
            gt_keypoints_2d_np = gt_keypoints_2d.cpu().numpy()
            if gt_keypoints_2d_np.shape[2] == 2:
                gt_keypoints_2d_np = np.concatenate([gt_keypoints_2d_np, np.ones((*gt_keypoints_2d_np.shape[:2], 1))], axis=2)
        else:
            gt_keypoints_2d_np = None
        
        # Scale keypoints to image resolution
        pred_keypoints_2d_scaled = self.img_res * (pred_keypoints_2d_np[:, :, :2] + 0.5)
        if gt_keypoints_2d_np is not None:
            gt_keypoints_2d_scaled = self.img_res * (gt_keypoints_2d_np[:, :, :2] + 0.5)
        else:
            gt_keypoints_2d_scaled = None
        
        for i in range(vertices.shape[0]):
            fl = focal_length if focal_length is not None else self.focal_length
            
            # Render mesh (front view)
            rend_img = self.__call__(
                vertices[i].cpu().numpy(),
                camera_translation[i].cpu().numpy(),
                images_np[i],
                focal_length=fl,
                side_view=False
            )
            rend_img = torch.from_numpy(rend_img.transpose(2, 0, 1)).float()  # (3, H, W)
            
            # Render mesh (side view)
            rend_img_side = self.__call__(
                vertices[i].cpu().numpy(),
                camera_translation[i].cpu().numpy(),
                images_np[i],
                focal_length=fl,
                side_view=True
            )
            rend_img_side = torch.from_numpy(rend_img_side.transpose(2, 0, 1)).float()  # (3, H, W)
            
            # Render keypoints
            pred_keypoints_img = render_hand_keypoints(
                255 * images_np[i].copy(),
                pred_keypoints_2d_scaled[i]
            ) / 255.0
            pred_keypoints_img = torch.from_numpy(pred_keypoints_img.transpose(2, 0, 1)).float()  # (3, H, W)
            
            if gt_keypoints_2d_scaled is not None:
                gt_keypoints_img = render_hand_keypoints(
                    255 * images_np[i].copy(),
                    gt_keypoints_2d_scaled[i]
                ) / 255.0
                gt_keypoints_img = torch.from_numpy(gt_keypoints_img.transpose(2, 0, 1)).float()  # (3, H, W)
            else:
                gt_keypoints_img = torch.from_numpy(images_np[i].transpose(2, 0, 1)).float()
            
            # Collect images: [input, mesh_front, mesh_side, pred_keypoints, gt_keypoints]
            rend_imgs.append(torch.from_numpy(images[i].cpu().numpy()))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            rend_imgs.append(pred_keypoints_img)
            rend_imgs.append(gt_keypoints_img)
        
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

