"""
GCN Refinement Module (GPPR)
从MANO mesh faces计算邻接矩阵，使用GCN精细化顶点特征，回归姿态和形状残差
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


def compute_adjacency_from_faces(faces: torch.Tensor, num_vertices: int) -> torch.Tensor:
    """
    Compute adjacency matrix from MANO mesh faces
    
    Args:
        faces: (F, 3) triangle faces, each row contains 3 vertex indices
        num_vertices: number of vertices
    Returns:
        adj: (V, V) adjacency matrix (symmetric, binary)
    """
    device = faces.device
    adj = torch.zeros(num_vertices, num_vertices, device=device)
    
    # Build adjacency from faces
    for face in faces:
        i, j, k = face[0].item(), face[1].item(), face[2].item()
        # Each edge in the triangle connects two vertices
        adj[i, j] = adj[j, i] = 1
        adj[i, k] = adj[k, i] = 1
        adj[j, k] = adj[k, j] = 1
    
    return adj


def create_default_adjacency(num_vertices: int, device: torch.device = None) -> torch.Tensor:
    """
    Create a default adjacency matrix when MANO faces cannot be loaded.
    Uses a simple k-nearest neighbor graph structure.
    
    Args:
        num_vertices: number of vertices
        device: device for the tensor
    Returns:
        adj: (V, V) adjacency matrix (symmetric, binary)
    """
    if device is None:
        device = torch.device('cpu')
    
    adj = torch.zeros(num_vertices, num_vertices, device=device)
    
    # Create a simple connectivity pattern
    # Connect each vertex to its neighbors in a ring structure
    # This is a fallback when real mesh topology is unavailable
    for i in range(num_vertices):
        # Connect to next vertex (ring structure)
        next_idx = (i + 1) % num_vertices
        adj[i, next_idx] = 1
        adj[next_idx, i] = 1
        
        # Connect to previous vertex
        prev_idx = (i - 1) % num_vertices
        adj[i, prev_idx] = 1
        adj[prev_idx, i] = 1
        
        # Connect to vertex at distance 2 (for better connectivity)
        if num_vertices > 2:
            next2_idx = (i + 2) % num_vertices
            adj[i, next2_idx] = 1
            adj[next2_idx, i] = 1
    
    return adj


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network Layer
    H^{(l+1)} = σ(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})
    where D is the degree matrix
    """
    def __init__(self, in_dim: int, out_dim: int, activation: Optional[nn.Module] = None):
        """
        Args:
            in_dim: input feature dimension
            out_dim: output feature dimension
            activation: activation function (default: ReLU)
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = activation if activation is not None else nn.ReLU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Graph convolution
        
        Args:
            x: (B, V, C_in) vertex features
            adj: (V, V) adjacency matrix
        Returns:
            out: (B, V, C_out) output vertex features
        """
        B, V, C = x.shape
        
        # Compute degree matrix D
        degree = adj.sum(dim=1)  # (V,)
        # Avoid division by zero
        degree = torch.clamp(degree, min=1.0)
        D_inv_sqrt = torch.diag(degree.pow(-0.5))  # (V, V)
        
        # Normalize adjacency: D^{-1/2} A D^{-1/2}
        adj_norm = D_inv_sqrt @ adj @ D_inv_sqrt  # (V, V)
        
        # Graph convolution: adj_norm @ x @ W
        # x: (B, V, C_in), adj_norm: (V, V)
        # adj_norm @ x: (B, V, C_in)
        support = torch.bmm(adj_norm.unsqueeze(0).expand(B, -1, -1), x)  # (B, V, C_in)
        
        # Linear transformation
        out = self.linear(support)  # (B, V, C_out)
        
        # Activation
        out = self.activation(out)
        
        return out


class GCN(nn.Module):
    """
    Multi-layer GCN
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Args:
            input_dim: input feature dimension
            hidden_dims: list of hidden layer dimensions
            output_dim: output feature dimension
        """
        super().__init__()
        self.layers = nn.ModuleList()
        
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            activation = nn.ReLU() if i < len(dims) - 2 else nn.Identity()
            self.layers.append(GCNLayer(dims[i], dims[i+1], activation))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward through GCN layers
        
        Args:
            x: (B, V, C_in) input vertex features
            adj: (V, V) adjacency matrix
        Returns:
            out: (B, V, C_out) output vertex features
        """
        for layer in self.layers:
            x = layer(x, adj)
        return x


def sample_vertex_features(multi_scale_features: List[torch.Tensor],
                          vertices_2d: List[torch.Tensor]) -> torch.Tensor:
    """
    Sample vertex features from multi-scale feature maps
    
    Args:
        multi_scale_features: list of (B, D, H_i, W_i) feature maps at different scales
        vertices_2d: list of (B, V, 2) 2D vertex coordinates for each scale
    Returns:
        vertex_features: (B, V, C) sampled vertex features
        where C = sum of feature dimensions across scales
    """
    B, V = vertices_2d[0].shape[:2]
    sampled_features = []
    
    for feat_map, verts_2d in zip(multi_scale_features, vertices_2d):
        B_f, D, H, W = feat_map.shape
        assert B == B_f, f"Batch size mismatch: {B} vs {B_f}"
        
        # Normalize coordinates to [-1, 1] for grid_sample
        # verts_2d: (B, V, 2) in pixel coordinates
        x = verts_2d[:, :, 0] / (W - 1) * 2 - 1  # (B, V)
        y = verts_2d[:, :, 1] / (H - 1) * 2 - 1  # (B, V)
        
        # Create grid: (B, V, 1, 2)
        grid = torch.stack([x, y], dim=-1).unsqueeze(2)  # (B, V, 1, 2)
        
        # Sample features: (B, D, V, 1)
        sampled = F.grid_sample(
            feat_map, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )  # (B, D, V, 1)
        
        # Reshape: (B, V, D)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)
        sampled_features.append(sampled)
    
    # Concatenate features from all scales
    vertex_features = torch.cat(sampled_features, dim=-1)  # (B, V, C_total)
    
    return vertex_features


class GCNRefinement(nn.Module):
    """
    GCN-based Parameter Refinement (GPPR)
    Uses GCN to refine vertex features and predict pose/shape residuals
    """
    def __init__(self, feature_dim: int, num_vertices: int = 778,
                 gcn_hidden_dims: List[int] = [512, 256], 
                 hidden_dim: int = 256):
        """
        Args:
            feature_dim: total feature dimension from multi-scale sampling
            num_vertices: number of MANO vertices (778)
            gcn_hidden_dims: hidden dimensions for GCN layers
            hidden_dim: hidden dimension for regression head
        """
        super().__init__()
        self.num_vertices = num_vertices
        self.feature_dim = feature_dim
        
        # GCN for vertex feature refinement
        self.gcn = GCN(
            input_dim=feature_dim,
            hidden_dims=gcn_hidden_dims,
            output_dim=hidden_dim
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Regression heads for pose and shape residuals
        # Pose residual: 48 dim (16 joints * 3)
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 48)
        )
        
        # Shape residual: 10 dim
        self.shape_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )
        
        # Store adjacency matrix (will be set from MANO faces)
        self.register_buffer('adjacency', None)

    def set_adjacency(self, faces: torch.Tensor):
        """
        Set adjacency matrix from MANO faces
        
        Args:
            faces: (F, 3) triangle faces
        """
        self.adjacency = compute_adjacency_from_faces(faces, self.num_vertices)
    
    def set_adjacency_from_matrix(self, adj: torch.Tensor):
        """
        Set adjacency matrix directly from a matrix
        
        Args:
            adj: (V, V) adjacency matrix
        """
        if adj.shape != (self.num_vertices, self.num_vertices):
            raise ValueError(f"Adjacency matrix shape {adj.shape} does not match expected ({self.num_vertices}, {self.num_vertices})")
        self.adjacency = adj

    def forward(self, vertex_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine vertex features and predict residuals
        
        Args:
            vertex_features: (B, V, C) vertex features from multi-scale sampling
        Returns:
            delta_pose: (B, 48) pose residual
            delta_shape: (B, 10) shape residual
        """
        if self.adjacency is None:
            raise ValueError("Adjacency matrix not set. Call set_adjacency() first.")
        
        # GCN refinement
        refined_features = self.gcn(vertex_features, self.adjacency)  # (B, V, hidden_dim)
        
        # Global pooling: (B, V, hidden_dim) -> (B, hidden_dim)
        # Transpose for pooling: (B, hidden_dim, V)
        pooled = self.global_pool(refined_features.transpose(1, 2))  # (B, hidden_dim, 1)
        pooled = pooled.squeeze(-1)  # (B, hidden_dim)
        
        # Predict residuals
        delta_pose = self.pose_head(pooled)  # (B, 48)
        delta_shape = self.shape_head(pooled)  # (B, 10)
        
        return delta_pose, delta_shape

