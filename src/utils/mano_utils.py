"""
MANO Utilities
MANO模型包装和工具函数
"""
import torch
import torch.nn as nn
from typing import Dict, Optional
import os
import pickle
import numpy as np


def load_mano_faces(mano_path: str) -> torch.Tensor:
    """
    Load MANO faces directly from pkl file without loading the full model
    This is useful when smplx is not available
    
    Args:
        mano_path: path to MANO model file (.pkl) or directory containing MANO_RIGHT.pkl
    Returns:
        faces: (F, 3) tensor of face indices
    """
    import os.path as osp
    
    # If mano_path is a directory, look for MANO_RIGHT.pkl
    if osp.isdir(mano_path):
        mano_file = osp.join(mano_path, 'MANO_RIGHT.pkl')
        if not osp.exists(mano_file):
            # Try MANO_LEFT.pkl
            mano_file = osp.join(mano_path, 'MANO_LEFT.pkl')
    else:
        mano_file = mano_path
    
    if not osp.exists(mano_file):
        raise FileNotFoundError(f"MANO file not found: {mano_file}")
    
    # Fix numpy compatibility before loading pickle
    # This is needed because pickle files may reference numpy.bool, numpy.int, etc.
    # which were removed in numpy 1.20+ and 2.0+
    import sys
    if 'numpy' in sys.modules:
        # Temporarily add deprecated numpy types to numpy module
        np_module = sys.modules['numpy']
        # Use compatible types for NumPy 2.0+
        try:
            np_bool_type = np.bool_ if hasattr(np, 'bool_') else bool
            np_int_type = np.int_ if hasattr(np, 'int_') else int
            np_float_type = np.float64  # Use float64 instead of float_ for NumPy 2.0+
            np_complex_type = np.complex128  # Use complex128 instead of complex_ for NumPy 2.0+
        except:
            np_bool_type = bool
            np_int_type = int
            np_float_type = float
            np_complex_type = complex
        
        if not hasattr(np_module, 'bool'):
            np_module.bool = np_bool_type
        if not hasattr(np_module, 'int'):
            np_module.int = np_int_type
        if not hasattr(np_module, 'float'):
            np_module.float = np_float_type
        if not hasattr(np_module, 'complex'):
            np_module.complex = np_complex_type
        if not hasattr(np_module, 'object'):
            np_module.object = object
        if not hasattr(np_module, 'unicode'):
            np_module.unicode = str
        if not hasattr(np_module, 'str'):
            np_module.str = str
    
    try:
        # Custom unpickler to handle chumpy objects gracefully
        # Note: In Python 3, we need to handle encoding for string objects
        class ChumpyUnpickler(pickle.Unpickler):
            """Custom unpickler that handles chumpy objects without requiring chumpy module"""
            def __init__(self, file, **kwargs):
                # Python 3.8+ doesn't support encoding in Unpickler constructor
                # We need to handle it differently by overriding load_string
                super().__init__(file, **kwargs)
            
            def load_string(self):
                """Override to handle latin1 encoded strings"""
                # Read the string length
                length = self.load()
                # Read the string data
                data = self.file.read(length)
                # Decode with latin1 encoding
                return data.decode('latin1')
            
            def find_class(self, module, name):
                # Handle numpy compatibility issues (numpy 1.20+ and 2.0+ removed direct bool/int/float exports)
                if module == 'numpy' and name in ['bool', 'int', 'float', 'complex', 'object', 'unicode', 'str']:
                    # Map old numpy types to new ones (compatible with NumPy 2.0+)
                    try:
                        type_map = {
                            'bool': np.bool_ if hasattr(np, 'bool_') else bool,
                            'int': np.int_ if hasattr(np, 'int_') else int,
                            'float': np.float64,  # Use float64 instead of float_ for NumPy 2.0+
                            'complex': np.complex128,  # Use complex128 instead of complex_ for NumPy 2.0+
                            'object': object,
                            'unicode': str,
                            'str': str
                        }
                        result = type_map.get(name)
                        if result is not None:
                            return result
                    except Exception:
                        pass
                    # Fallback: try to get from numpy module
                    try:
                        return getattr(np, name)
                    except AttributeError:
                        # Return a safe default type
                        fallback_map = {
                            'bool': bool,
                            'int': int,
                            'float': float,
                            'complex': complex,
                            'object': object,
                            'unicode': str,
                            'str': str
                        }
                        return fallback_map.get(name, object)
                
                # If trying to load chumpy, return a dummy class
                if module.startswith('chumpy') or module == 'chumpy':
                    # Return a dummy class that can be instantiated and accessed
                    class DummyChumpy:
                        def __init__(self, *args, **kwargs):
                            # Store the underlying data if provided
                            if args:
                                self._data = args[0]
                            else:
                                self._data = None
                        def __getattr__(self, name):
                            if name == 'r':  # chumpy arrays have .r attribute for raw data
                                return self._data
                            return None
                        def __array__(self):
                            # Allow conversion to numpy array
                            if self._data is not None:
                                return np.array(self._data)
                            return np.array([])
                    return DummyChumpy
                # For other modules, use default behavior
                try:
                    return super().find_class(module, name)
                except (AttributeError, ModuleNotFoundError) as e:
                    # If module not found, return a dummy class
                    class DummyClass:
                        def __init__(self, *args, **kwargs):
                            pass
                    return DummyClass
        
        # Try to load MANO pickle file
        # MANO files are typically Python 2 pickles that need latin1 encoding
        # Python 3.8+ removed encoding parameter from pickle.load(), so we need a workaround
        with open(mano_file, 'rb') as f:
            # Method 1: Try with custom unpickler that handles encoding and missing modules
            try:
                unpickler = ChumpyUnpickler(f)
                mano_data = unpickler.load()
            except (UnicodeDecodeError, TypeError, AttributeError, ModuleNotFoundError) as e1:
                # Method 2: Try with a wrapper that handles encoding
                f.seek(0)
                try:
                    # Create a wrapper class that handles latin1 encoding
                    class Latin1FileWrapper:
                        def __init__(self, file):
                            self.file = file
                        def read(self, size=-1):
                            return self.file.read(size)
                        def readline(self):
                            return self.file.readline()
                    
                    # Use the wrapper with custom unpickler
                    wrapped_file = Latin1FileWrapper(f)
                    unpickler = ChumpyUnpickler(wrapped_file)
                    mano_data = unpickler.load()
                except Exception as e2:
                    # Method 3: Try standard pickle with protocol handling
                    f.seek(0)
                    try:
                        # Try to load with lower protocol (more compatible)
                        # Read the file and try different protocols
                        import io
                        raw_data = f.read()
                        f_obj = io.BytesIO(raw_data)
                        # Try protocol 2 (Python 2 compatible)
                        try:
                            mano_data = pickle.load(f_obj, encoding='latin1')
                        except TypeError:
                            # Python 3.8+ doesn't support encoding parameter
                            # Try without encoding (may fail but worth trying)
                            f_obj.seek(0)
                            mano_data = pickle.load(f_obj)
                    except Exception as e3:
                        # Final fallback: try to extract faces directly using a minimal loader
                        # Provide helpful error message
                        error_msg = (
                            f"Failed to load MANO file '{mano_file}'. "
                            f"This typically requires either:\n"
                            f"1. Install scipy: pip install scipy\n"
                            f"2. Use smplx library (recommended): pip install smplx\n"
                            f"Original errors: {e1}, {e2}, {e3}"
                        )
                        raise RuntimeError(error_msg)
        
        # Extract faces from MANO data
        # Handle both numpy arrays and chumpy arrays
        faces = None
        if 'f' in mano_data:
            faces_raw = mano_data['f']
        elif 'faces' in mano_data:
            faces_raw = mano_data['faces']
        else:
            raise KeyError("Could not find 'f' or 'faces' in MANO data")
        
        # Convert to numpy array (handles chumpy arrays, numpy arrays, etc.)
        if hasattr(faces_raw, 'r'):  # chumpy array has .r attribute
            faces = np.array(faces_raw.r, dtype=np.int64)
        elif hasattr(faces_raw, 'data'):  # some array-like objects
            faces = np.array(faces_raw.data, dtype=np.int64)
        elif isinstance(faces_raw, np.ndarray):
            faces = np.array(faces_raw, dtype=np.int64)
        else:
            # Regular list or other iterable
            faces = np.array(faces_raw, dtype=np.int64)
        
        # Convert to torch tensor
        faces = torch.from_numpy(faces)
        return faces
        
    except Exception as e:
        raise RuntimeError(f"Failed to load MANO faces from {mano_file}: {e}")


def load_mano_model(mano_path: str, gender: str = 'neutral', 
                   num_hand_joints: int = 15, use_pca: bool = True,
                   flat_hand_mean: bool = True) -> nn.Module:
    """
    Load MANO model using smplx
    
    Args:
        mano_path: path to MANO model directory
        gender: 'male', 'female', or 'neutral'
        num_hand_joints: number of hand joints
        use_pca: whether to use PCA for hand pose
        flat_hand_mean: whether to use flat hand mean
    Returns:
        MANO model
    """
    try:
        import smplx
    except ImportError:
        raise ImportError("smplx is required. Install with: pip install smplx")
    
    try:
        mano_model = smplx.create(
            model_path=mano_path,
            model_type='mano',
            gender=gender,
            num_hand_joints=num_hand_joints,
            use_pca=use_pca,
            flat_hand_mean=flat_hand_mean
        )
        return mano_model
    except Exception as e:
        raise RuntimeError(f"Failed to load MANO model: {e}")


class MANOWrapper(nn.Module):
    """
    Wrapper for MANO model that extends 16 joints to 21 keypoints
    Similar to WiLoR's MANO wrapper
    """
    def __init__(self, mano_model):
        super().__init__()
        self.mano = mano_model
        # Freeze MANO parameters (they are not trainable)
        for param in self.mano.parameters():
            param.requires_grad = False
        
        # Fingertip vertex indices for right hand (MANO mesh)
        # These are used to extract fingertip positions from vertices
        # Index finger: 317, Middle: 444, Ring: 556, Pinky: 673, Thumb: 745
        self.fingertip_vertex_idx = [745, 317, 444, 556, 673]  # Thumb, Index, Middle, Ring, Pinky
        self.register_buffer('fingertip_indices', torch.tensor(self.fingertip_vertex_idx, dtype=torch.long))
        
        # Mapping from MANO 16 joints + 5 fingertips to 21 keypoints (Dex-YCB format)
        # MANO joint order: [wrist, thumb1-3, index1-3, middle1-3, ring1-3, pinky1-3]
        # We need to add fingertips and reorder to match Dex-YCB format
        # Dex-YCB order: [wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4]
        # MANO 16 joints: [0, 1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15]
        # We add fingertips at positions: thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip
        # Then map to Dex-YCB order: [0, 1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20]
        # The mapping: take MANO joints [0,1,2,3, 4,5,6, 7,8,9, 10,11,12, 13,14,15]
        # and add fingertips at appropriate positions
        
    def _extend_joints_to_21(self, joints: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
        """
        Extend MANO 16 joints to 21 keypoints by adding fingertips
        
        Args:
            joints: (B, 16, 3) MANO joints
            vertices: (B, 778, 3) MANO vertices
        Returns:
            keypoints_21: (B, 21, 3) 21 keypoints in Dex-YCB format
        """
        B = joints.shape[0]
        device = joints.device
        
        # Extract fingertip positions from vertices
        fingertips = torch.index_select(vertices, 1, self.fingertip_indices)  # (B, 5, 3)
        
        # MANO joint order: [wrist(0), thumb1(1), thumb2(2), thumb3(3), 
        #                     index1(4), index2(5), index3(6),
        #                     middle1(7), middle2(8), middle3(9),
        #                     ring1(10), ring2(11), ring3(12),
        #                     pinky1(13), pinky2(14), pinky3(15)]
        # Dex-YCB order: [wrist(0), thumb1(1), thumb2(2), thumb3(3), thumb_tip(4),
        #                 index1(5), index2(6), index3(7), index_tip(8),
        #                 middle1(9), middle2(10), middle3(11), middle_tip(12),
        #                 ring1(13), ring2(14), ring3(15), ring_tip(16),
        #                 pinky1(17), pinky2(18), pinky3(19), pinky_tip(20)]
        
        # Build 21 keypoints
        keypoints_21 = torch.zeros(B, 21, 3, device=device)
        
        # Wrist
        keypoints_21[:, 0] = joints[:, 0]
        
        # Thumb: joints 1,2,3 + thumb tip
        keypoints_21[:, 1:4] = joints[:, 1:4]
        keypoints_21[:, 4] = fingertips[:, 0]  # thumb tip
        
        # Index: joints 4,5,6 + index tip
        keypoints_21[:, 5:8] = joints[:, 4:7]
        keypoints_21[:, 8] = fingertips[:, 1]  # index tip
        
        # Middle: joints 7,8,9 + middle tip
        keypoints_21[:, 9:12] = joints[:, 7:10]
        keypoints_21[:, 12] = fingertips[:, 2]  # middle tip
        
        # Ring: joints 10,11,12 + ring tip
        keypoints_21[:, 13:16] = joints[:, 10:13]
        keypoints_21[:, 16] = fingertips[:, 3]  # ring tip
        
        # Pinky: joints 13,14,15 + pinky tip
        keypoints_21[:, 17:20] = joints[:, 13:16]
        keypoints_21[:, 20] = fingertips[:, 4]  # pinky tip
        
        return keypoints_21
    
    def _rotation_matrix_to_axis_angle(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrix to axis-angle representation
        
        Args:
            rotation_matrix: (B, 3, 3) rotation matrices
        Returns:
            axis_angle: (B, 3) axis-angle vectors
        """
        # Use PyTorch3D-style conversion if available, otherwise use simple method
        batch_size = rotation_matrix.shape[0]
        device = rotation_matrix.device
        
        # Compute the angle
        trace = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + 1e-7, 1 - 1e-7))
        
        # Compute the axis
        # For small angles, use simplified formula to avoid numerical issues
        small_angle_mask = angle.abs() < 1e-3
        
        axis = torch.zeros(batch_size, 3, device=device)
        
        # For non-small angles
        if (~small_angle_mask).any():
            r = rotation_matrix[~small_angle_mask]
            axis[~small_angle_mask] = torch.stack([
                r[:, 2, 1] - r[:, 1, 2],
                r[:, 0, 2] - r[:, 2, 0],
                r[:, 1, 0] - r[:, 0, 1]
            ], dim=1) / (2 * torch.sin(angle[~small_angle_mask]).unsqueeze(1))
        
        # Axis-angle = angle * axis
        axis_angle = angle.unsqueeze(1) * axis
        
        return axis_angle

    def forward(self, global_orient: torch.Tensor,
                hand_pose: torch.Tensor,
                betas: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through MANO
        
        Args:
            global_orient: (B, 1, 3, 3) global orientation rotation matrices
            hand_pose: (B, 15, 3, 3) hand pose rotation matrices
            betas: (B, 10) shape parameters
        Returns:
            dict with 'vertices', 'joints' (21 keypoints), etc.
        """
        # Convert rotation matrices to axis-angle
        # smplx MANO with use_pca=True expects axis-angle input
        batch_size = global_orient.shape[0]
        
        # Convert global_orient: (B, 1, 3, 3) -> (B, 3)
        global_orient_aa = self._rotation_matrix_to_axis_angle(
            global_orient.reshape(batch_size, 3, 3)
        )  # (B, 3)
        
        # Convert hand_pose: (B, 15, 3, 3) -> (B, 45)
        hand_pose_aa = self._rotation_matrix_to_axis_angle(
            hand_pose.reshape(batch_size * 15, 3, 3)
        ).reshape(batch_size, 45)  # (B, 45)
        
        # Call MANO with axis-angle inputs
        mano_output = self.mano(
            global_orient=global_orient_aa,  # (B, 3)
            hand_pose=hand_pose_aa,  # (B, 45)
            betas=betas,  # (B, 10)
            pose2rot=True  # Convert axis-angle to rotation matrices internally
        )
        
        # Extend 16 joints to 21 keypoints
        joints_21 = self._extend_joints_to_21(
            mano_output.joints,  # (B, 16, 3)
            mano_output.vertices  # (B, 778, 3)
        )
        
        return {
            'vertices': mano_output.vertices,
            'joints': joints_21,  # (B, 21, 3)
            'faces': self.mano.faces
        }

