"""
Multi-Modal Fusion Graph Neural Network (MMF-GNN)
==================================================

This module implements the Multi-Modal Fusion Graph Neural Network for NexusFusion.
The MMF-GNN performs tightly-coupled fusion of heterogeneous sensor data including
LiDAR point clouds, camera features, GNSS coordinates, IMU sequences, and V2X communications.

Architecture:
- Multi-modal patch embedding for unified feature representation
- Cooperative attention mechanism for cross-modal fusion
- Global pooling for final feature aggregation

Authors: NexusFusion Research Team
License: MIT
"""

from typing import Dict, Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiModalPatchEmbedding(nn.Module):
    """
    Multi-modal patch embedding based on V2X-ViT architecture.
    
    This module embeds different sensor modalities into a unified 256-dimensional
    feature space, enabling cross-modal attention and fusion.
    
    Args:
        embed_dim (int): Embedding dimension (default: 256)
        modality_types (list): List of supported modalities
        dropout (float): Dropout rate for regularization
    """
    
    def __init__(self, embed_dim=256, modality_types=['lidar', 'camera', 'gnss', 'imu', 'v2x'], dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.modality_types = modality_types
        self.dropout = dropout
        
        # LiDAR point cloud embedding (x, y, z, intensity)
        self.lidar_embedding = nn.Sequential(
            nn.Linear(4, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Camera feature embedding (2D coordinates + 128D descriptor)
        self.camera_embedding = nn.Sequential(
            nn.Linear(130, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # GNSS coordinate embedding (latitude, longitude, altitude)
        self.gnss_embedding = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout * 0.5)
        )
        
        # IMU sequence embedding (6-axis inertial data)
        self.imu_embedding = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.imu_gru = nn.GRU(embed_dim, embed_dim, batch_first=True, dropout=dropout if dropout > 0 else 0)
        
        # V2X communication embedding (vehicle state vectors)
        self.v2x_embedding = nn.Sequential(
            nn.Linear(8, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Modality type embeddings for distinguishing different sensor types
        self.modality_type_embeddings = nn.Embedding(len(modality_types), embed_dim)
        
        # Positional embeddings for sequence modeling
        self.position_embeddings = nn.Parameter(torch.zeros(1, 2048, embed_dim))
        
    def forward(self, data_dict):
        """
        Forward pass for multi-modal embedding.
        
        Args:
            data_dict (dict): Dictionary containing sensor data for each modality
            
        Returns:
            torch.Tensor: Unified multi-modal embeddings [B, N_total, D]
        """
        batch_size = list(data_dict.values())[0].size(0)
        all_embeddings = []
        
        for idx, modality in enumerate(self.modality_types):
            if modality in data_dict:
                data = data_dict[modality]
                
                if modality == 'lidar':
                    x = self.lidar_embedding(data)  # (B, N_pts, D)
                elif modality == 'camera':
                    x = self.camera_embedding(data)  # (B, N_kp, D)
                elif modality == 'gnss':
                    x = self.gnss_embedding(data).unsqueeze(1)  # (B, 1, D)
                elif modality == 'imu':
                    x = self.imu_embedding(data)
                    x, _ = self.imu_gru(x)  # (B, T, D)
                elif modality == 'v2x':
                    x = self.v2x_embedding(data)  # (B, M, D)
                
                # Add modality type embedding
                modality_emb = self.modality_type_embeddings(
                    torch.tensor(idx, device=x.device, dtype=torch.long)
                )
                x = x + modality_emb.unsqueeze(0).expand(batch_size, x.size(1), -1)
                
                all_embeddings.append(x)
        
        # Concatenate all modalities
        embeddings = torch.cat(all_embeddings, dim=1)
        
        # Add positional embeddings
        seq_len = embeddings.size(1)
        if seq_len <= self.position_embeddings.size(1):
            pos_emb = self.position_embeddings[:, :seq_len, :]
            embeddings = embeddings + pos_emb
        
        return embeddings


class CooperativeAttention(nn.Module):
    """
    Cooperative attention mechanism for multi-modal fusion.
    
    This module implements spatial, temporal, cross-modal, and cooperative
    attention mechanisms to enable effective information fusion across
    different sensor modalities and vehicle communications.
    
    Args:
        embed_dim (int): Feature embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout rate
    """
    
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Spatial attention for within-modality relationships
        self.spatial_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Temporal attention for sequence modeling
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cross-modal attention for inter-modality fusion
        self.cross_modal_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Cooperative attention for V2X communication
        self.cooperative_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization for each attention type
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.norm4 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)
        
    def forward(self, x, temporal_context=None, cooperative_features=None):
        """
        Forward pass for cooperative attention.
        
        Args:
            x (torch.Tensor): Input features [B, N, D]
            temporal_context (torch.Tensor, optional): Temporal context [B, T, N, D]
            cooperative_features (torch.Tensor, optional): V2X cooperative features [B, M, D]
            
        Returns:
            torch.Tensor: Attended features [B, N, D]
        """
        # Spatial self-attention
        attn_out, _ = self.spatial_attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Temporal attention
        if temporal_context is not None:
            B, T, N, D = temporal_context.shape
            temp_context = temporal_context.reshape(B, T * N, D)
            attn_out, _ = self.temporal_attention(x, temp_context, temp_context)
            x = self.norm2(x + attn_out)
        
        # Cross-modal attention
        attn_out, _ = self.cross_modal_attention(x, x, x)
        x = self.norm3(x + attn_out)
        
        # Cooperative attention for V2X
        if cooperative_features is not None:
            attn_out, _ = self.cooperative_attention(x, cooperative_features, cooperative_features)
            x = self.norm4(x + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm_ffn(x + ffn_out)
        
        return x


class AdvancedMMF_GNN(nn.Module):
    """
    Advanced Multi-Modal Fusion Graph Neural Network.
    
    This is the core MMF-GNN model that performs deep fusion of heterogeneous
    sensor data for autonomous driving applications. The model follows the
    V2X-ViT architecture with cooperative attention mechanisms.
    
    Key Features:
    - Multi-modal patch embedding for unified representation
    - Deep transformer architecture with cooperative attention
    - Global pooling for feature aggregation
    - Residual connections and layer normalization for stable training
    
    Args:
        embed_dim (int): Feature embedding dimension (default: 256)
        depth (int): Number of transformer layers (default: 6)
        num_heads (int): Number of attention heads (default: 8)
        dropout (float): Dropout rate for regularization (default: 0.1)
    """
    
    def __init__(self, embed_dim=256, depth=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.dropout = dropout
        
        # Multi-modal patch embedding
        self.patch_embedding = MultiModalPatchEmbedding(embed_dim, dropout=dropout)
        
        # Transformer blocks with cooperative attention
        self.transformer_blocks = nn.ModuleList([
            CooperativeAttention(embed_dim, num_heads, dropout)
            for _ in range(depth)
        ])
        
        # Layer normalization for each transformer block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(depth)
        ])
        
        # Global pooling with attention weighting
        self.global_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Output head for final feature processing
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
        # Residual connection dropout
        self.residual_dropout = nn.Dropout(dropout)
        
        # Classification token for global representation
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights using truncated normal distribution."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, lidar_points, camera_keypoints, gnss, imu, obu, v2v_states, participants):
        """
        Forward pass of the MMF-GNN model.
        
        Args:
            lidar_points (torch.Tensor): LiDAR point cloud data [B, N_pts, 4]
            camera_keypoints (torch.Tensor): Camera keypoint features [B, N_kp, 130]
            gnss (torch.Tensor): GNSS coordinates [B, 3]
            imu (torch.Tensor): IMU sequence data [B, T, 6]
            obu (torch.Tensor): On-board unit data [B, D_obu]
            v2v_states (torch.Tensor): V2V communication states [B, M, 8]
            participants (torch.Tensor): Participant information [B, N_part, D_part]
            
        Returns:
            tuple: (global_features, auxiliary_outputs)
                - global_features (torch.Tensor): Fused global features [B, D]
                - auxiliary_outputs (dict): Additional outputs including embeddings and weights
        """
        B = lidar_points.size(0)
        
        # Prepare multi-modal data dictionary
        data_dict = {
            'lidar': lidar_points,
            'camera': camera_keypoints,
            'gnss': gnss,
            'imu': imu,
            'v2x': v2v_states
        }
        
        # Multi-modal embedding
        embeddings = self.patch_embedding(data_dict)
        
        # Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)
        
        # Transformer processing with residual connections
        x = embeddings
        for i, (block, layer_norm) in enumerate(zip(self.transformer_blocks, self.layer_norms)):
            # Pre-norm residual connection
            residual = x
            x = layer_norm(x)
            x = block(x, temporal_context=None, cooperative_features=None)
            x = self.residual_dropout(x) + residual
        
        # Global pooling (exclude cls token)
        weights = self.global_pool(x[:, 1:])
        h_global = torch.sum(x[:, 1:] * weights, dim=1)
        
        # Final output processing
        h_global = self.head(h_global)
        
        return h_global, {"embeddings": x, "weights": weights}


# Model configuration
MMF_GNN_CONFIG = {
    "embed_dim": 256,
    "depth": 6,
    "num_heads": 8,
    "dropout": 0.1,
    "modality_types": ['lidar', 'camera', 'gnss', 'imu', 'v2x']
}


def create_mmf_gnn(config=None):
    """
    Factory function to create MMF-GNN model.
    
    Args:
        config (dict, optional): Model configuration. Uses default if None.
        
    Returns:
        AdvancedMMF_GNN: Initialized MMF-GNN model
    """
    if config is None:
        config = MMF_GNN_CONFIG
    
    return AdvancedMMF_GNN(
        embed_dim=config["embed_dim"],
        depth=config["depth"],
        num_heads=config["num_heads"],
        dropout=config["dropout"]
    )
