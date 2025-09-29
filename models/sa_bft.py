"""
Semantic-Aware Byzantine Fault-Tolerant (SA-BFT) Consensus Module
================================================================

This module implements the Semantic-Aware Byzantine Fault-Tolerant consensus 
mechanism for NexusFusion. The SA-BFT protocol ensures reliable consensus
among distributed vehicles in the presence of malicious or faulty nodes.

Key Features:
- Communication quality encoding for V2X networks
- Byzantine node detection using anomaly analysis
- Adaptive threshold computation for fault tolerance
- Iterative consensus protocol with voting mechanisms
- Historical state tracking for temporal consistency

Authors: NexusFusion Research Team
License: MIT
"""

from typing import Dict, List, Tuple, Optional
import math
import torch
from torch import nn
import torch.nn.functional as F


class CommunicationQualityEncoder(nn.Module):
    """
    V2X Communication Quality Encoder.
    
    This module encodes various communication quality metrics including
    signal strength, latency, packet loss, and contextual information
    such as distance and vehicle types.
    
    Args:
        embed_dim (int): Embedding dimension for feature representation
    """
    
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Communication quality feature encoder
        # Input: [rssi, snr, latency, packet_loss]
        self.comm_encoder = nn.Sequential(
            nn.Linear(4, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim)
        )
        
        # Distance embedding (discretized up to 300 meters)
        self.distance_embedding = nn.Embedding(300, embed_dim)
        
        # Temporal encoding for timestamp information
        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim)
        )
        
        # Vehicle type embedding: [ego, car, truck, bus, bike]
        self.vehicle_type_embedding = nn.Embedding(5, embed_dim)
        
    def forward(self, comm_quality, distances, timestamps, vehicle_types):
        """
        Forward pass for communication quality encoding.
        
        Args:
            comm_quality (torch.Tensor): Communication metrics [B, M, 4]
            distances (torch.Tensor): Inter-vehicle distances [B, M]
            timestamps (torch.Tensor): Message timestamps [B, M, 1]
            vehicle_types (torch.Tensor): Vehicle type indices [B, M]
            
        Returns:
            torch.Tensor: Encoded communication features [B, M, D]
        """
        # Encode communication quality metrics
        comm_features = self.comm_encoder(comm_quality)
        
        # Distance embedding (discretized and clamped)
        distance_bins = torch.clamp(distances.long(), 0, 299)
        distance_features = self.distance_embedding(distance_bins)
        
        # Temporal encoding
        temporal_features = self.temporal_encoder(timestamps)
        
        # Vehicle type encoding
        type_features = self.vehicle_type_embedding(vehicle_types)
        
        # Combine all features through addition
        combined_features = (comm_features + distance_features + 
                           temporal_features + type_features)
        
        return combined_features


class ByzantineDetectionNetwork(nn.Module):
    """
    Byzantine Node Detection Network.
    
    This network identifies potentially malicious or faulty nodes through
    multiple detection mechanisms including anomaly detection, consistency
    checking, reputation scoring, and temporal analysis.
    
    Args:
        embed_dim (int): Feature embedding dimension
        num_heads (int): Number of attention heads for anomaly detection
    """
    
    def __init__(self, embed_dim=256, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Anomaly detection attention mechanism
        self.anomaly_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=0.1, batch_first=True
        )
        
        # Consistency checking network
        self.consistency_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )
        
        # Reputation scoring network
        self.reputation_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency checker using GRU
        self.temporal_consistency = nn.GRU(embed_dim, embed_dim, batch_first=True)
        
    def forward(self, peer_features, trusted_context, historical_features=None):
        """
        Forward pass for Byzantine detection.
        
        Args:
            peer_features (torch.Tensor): Peer node features [B, M, D]
            trusted_context (torch.Tensor): Trusted reference context [B, D]
            historical_features (torch.Tensor, optional): Historical data [B, M, T, D]
            
        Returns:
            tuple: (anomaly_scores, detection_info)
                - anomaly_scores (torch.Tensor): Anomaly scores for each peer [B, M]
                - detection_info (dict): Detailed detection information
        """
        B, M, D = peer_features.shape
        
        # Anomaly detection using attention mechanism
        context_expanded = trusted_context.unsqueeze(1).expand(-1, M, -1)
        attn_out, attn_weights = self.anomaly_attention(
            peer_features, context_expanded, context_expanded
        )
        
        # Consistency scoring
        consistency_input = torch.cat([peer_features, attn_out], dim=-1)
        consistency_scores = self.consistency_net(consistency_input).squeeze(-1)
        
        # Reputation scoring
        reputation_scores = self.reputation_net(peer_features).squeeze(-1)
        
        # Temporal consistency analysis (if historical data available)
        temporal_scores = torch.ones_like(consistency_scores)
        if historical_features is not None:
            B, M, T, D = historical_features.shape
            hist_reshaped = historical_features.reshape(B * M, T, D)
            _, hidden = self.temporal_consistency(hist_reshaped)
            current_hidden = self.temporal_consistency(peer_features.reshape(B * M, 1, D))[1]
            
            # Compute temporal deviation
            temporal_diff = torch.norm(hidden.squeeze(0) - current_hidden.squeeze(0), dim=-1)
            temporal_scores = torch.sigmoid(-temporal_diff).reshape(B, M)
        
        # Combine anomaly scores with weighted contributions
        anomaly_scores = (
            consistency_scores * 0.4 +           # Consistency weight
            (1 - reputation_scores) * 0.3 +      # Reputation weight (inverted)
            (1 - temporal_scores) * 0.3           # Temporal weight (inverted)
        )
        
        return anomaly_scores, {
            'consistency': consistency_scores,
            'reputation': reputation_scores,
            'temporal': temporal_scores,
            'attention_weights': attn_weights
        }


class ConsensusProtocol(nn.Module):
    """
    Simplified Byzantine Fault-Tolerant Consensus Protocol.
    
    This module implements an iterative consensus mechanism that combines
    voting weights with validity scores to achieve agreement among
    non-faulty nodes in the network.
    
    Args:
        embed_dim (int): Feature embedding dimension
        max_iterations (int): Maximum number of consensus iterations
    """
    
    def __init__(self, embed_dim=256, max_iterations=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_iterations = max_iterations
        
        # Voting weight computation network
        self.voting_weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        # Consensus aggregation network
        self.consensus_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, valid_features, validity_scores):
        """
        Forward pass for consensus protocol.
        
        Args:
            valid_features (torch.Tensor): Valid node features [B, M, D]
            validity_scores (torch.Tensor): Node validity scores [B, M]
            
        Returns:
            tuple: (consensus_state, consensus_weights)
                - consensus_state (torch.Tensor): Final consensus state [B, D]
                - consensus_weights (torch.Tensor): Final consensus weights [B, M]
        """
        B, M, D = valid_features.shape
        
        # Compute initial voting weights
        voting_weights = self.voting_weight_net(valid_features)  # (B, M, 1)
        
        # Combine with validity scores
        combined_weights = voting_weights.squeeze(-1) * validity_scores  # (B, M)
        combined_weights = F.softmax(combined_weights, dim=1)
        
        # Iterative consensus refinement
        consensus_state = torch.zeros(B, D, device=valid_features.device)
        
        for iteration in range(self.max_iterations):
            # Weighted feature aggregation
            weighted_features = (valid_features * combined_weights.unsqueeze(-1)).sum(dim=1)
            
            # Update consensus state
            consensus_state = self.consensus_net(weighted_features + consensus_state)
            
            # Recompute weights based on similarity to current consensus
            similarities = F.cosine_similarity(
                valid_features, 
                consensus_state.unsqueeze(1).expand(-1, M, -1), 
                dim=-1
            )
            combined_weights = F.softmax(similarities * validity_scores, dim=1)
        
        return consensus_state, combined_weights


class AdvancedSABFTValidator(nn.Module):
    """
    Advanced Semantic-Aware Byzantine Fault-Tolerant Validator.
    
    This is the main SA-BFT module that orchestrates communication quality
    encoding, Byzantine detection, and consensus formation to produce
    reliable and trustworthy consensus states for cooperative driving.
    
    Key Features:
    - Multi-modal communication quality assessment
    - Sophisticated Byzantine fault detection
    - Adaptive threshold computation
    - Historical state tracking for temporal consistency
    - Confidence estimation for consensus reliability
    
    Args:
        embed_dim (int): Feature embedding dimension
        max_peers (int): Maximum number of peer vehicles
    """
    
    def __init__(self, embed_dim=256, max_peers=20):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_peers = max_peers
        
        # Communication quality encoder
        self.comm_quality_encoder = CommunicationQualityEncoder(embed_dim)
        
        # Byzantine detection network
        self.byzantine_detector = ByzantineDetectionNetwork(embed_dim)
        
        # Consensus protocol
        self.consensus_protocol = ConsensusProtocol(embed_dim)
        
        # Adaptive threshold computation network
        self.threshold_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Confidence estimation network
        self.confidence_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Historical state buffer for temporal consistency
        self.register_buffer('historical_states', torch.zeros(1, max_peers, 10, embed_dim))
        self.register_buffer('history_ptr', torch.zeros(1, dtype=torch.long))
        
    def update_history(self, peer_features):
        """
        Update historical state buffer for temporal consistency checking.
        
        Args:
            peer_features (torch.Tensor): Current peer features [B, M, D]
        """
        B, M, D = peer_features.shape
        
        # Expand buffer if needed
        if B > self.historical_states.size(0):
            new_history = torch.zeros(B, self.max_peers, 10, D, 
                                    device=peer_features.device)
            new_history[:self.historical_states.size(0)] = self.historical_states
            self.historical_states = new_history
            
            new_ptr = torch.zeros(B, dtype=torch.long, device=peer_features.device)
            new_ptr[:self.history_ptr.size(0)] = self.history_ptr
            self.history_ptr = new_ptr
        
        # Update circular buffer
        for b in range(B):
            ptr = self.history_ptr[b].item()
            self.historical_states[b, :M, ptr] = peer_features[b]
            self.history_ptr[b] = (ptr + 1) % 10
    
    def forward(self, peer_features, trusted_context, 
                comm_quality=None, distances=None, timestamps=None, vehicle_types=None):
        """
        Forward pass for SA-BFT consensus validation.
        
        Args:
            peer_features (torch.Tensor): Peer node features [B, M, D]
            trusted_context (torch.Tensor): Trusted reference context [B, D]
            comm_quality (torch.Tensor, optional): Communication quality [B, M, 4]
            distances (torch.Tensor, optional): Inter-vehicle distances [B, M]
            timestamps (torch.Tensor, optional): Message timestamps [B, M, 1]
            vehicle_types (torch.Tensor, optional): Vehicle type indices [B, M]
            
        Returns:
            dict: Comprehensive consensus results including:
                - consensus_state: Final trusted consensus state
                - consensus_weights: Node contribution weights
                - anomaly_scores: Byzantine detection scores
                - validity_scores: Node validity assessments
                - confidence: Consensus confidence level
                - adaptive_threshold: Dynamic fault detection threshold
                - detection_info: Detailed Byzantine detection information
        """
        B, M, D = peer_features.shape
        
        # Encode communication quality if provided
        comm_features = None
        if all(x is not None for x in [comm_quality, distances, timestamps, vehicle_types]):
            comm_features = self.comm_quality_encoder(
                comm_quality, distances, timestamps, vehicle_types
            )
            # Fuse communication quality with peer features
            peer_features = peer_features + comm_features
        
        # Update historical states during training
        if self.training:
            self.update_history(peer_features.detach())
        
        # Retrieve historical features for temporal consistency
        historical_features = None
        if self.historical_states.size(0) >= B:
            historical_features = self.historical_states[:B, :M]
        
        # Byzantine fault detection
        anomaly_scores, detection_info = self.byzantine_detector(
            peer_features, trusted_context, historical_features
        )
        
        # Compute adaptive threshold based on trusted context
        adaptive_threshold = self.threshold_net(trusted_context).squeeze(-1)  # (B,)
        
        # Determine node validity using adaptive threshold
        validity_scores = torch.sigmoid(-anomaly_scores + adaptive_threshold.unsqueeze(1))
        
        # Execute BFT consensus protocol
        consensus_state, consensus_weights = self.consensus_protocol(
            peer_features, validity_scores
        )
        
        # Estimate consensus confidence
        confidence = self.confidence_net(consensus_state).squeeze(-1)
        
        return {
            'consensus_state': consensus_state,
            'consensus_weights': consensus_weights,
            'anomaly_scores': anomaly_scores,
            'validity_scores': validity_scores,
            'confidence': confidence,
            'adaptive_threshold': adaptive_threshold,
            'detection_info': detection_info
        }
    
    @torch.no_grad()
    def get_trusted_peers(self, anomaly_scores, quantile=0.7):
        """
        Identify trusted peer nodes based on anomaly scores.
        
        Args:
            anomaly_scores (torch.Tensor): Anomaly scores for all peers [B, M]
            quantile (float): Quantile threshold for trust determination
            
        Returns:
            torch.Tensor: Boolean mask indicating trusted peers [B, M]
        """
        threshold = torch.quantile(anomaly_scores.reshape(-1), quantile)
        trusted_mask = anomaly_scores <= threshold
        return trusted_mask
    
    def get_consensus_quality(self, consensus_weights, validity_scores):
        """
        Evaluate the quality of achieved consensus.
        
        Args:
            consensus_weights (torch.Tensor): Final consensus weights [B, M]
            validity_scores (torch.Tensor): Node validity scores [B, M]
            
        Returns:
            torch.Tensor: Consensus quality scores [B]
        """
        # Weight distribution entropy (lower is better, indicates more focused consensus)
        entropy = -torch.sum(consensus_weights * torch.log(consensus_weights + 1e-8), dim=1)
        
        # Proportion of valid nodes
        valid_ratio = validity_scores.mean(dim=1)
        
        # Combined quality score
        quality = valid_ratio * torch.exp(-entropy)
        
        return quality


# Model configuration
SA_BFT_CONFIG = {
    "embed_dim": 256,
    "max_peers": 20,
    "max_iterations": 3,
    "num_heads": 8,
    "dropout": 0.1
}


def create_sa_bft(config=None):
    """
    Factory function to create SA-BFT validator.
    
    Args:
        config (dict, optional): Model configuration. Uses default if None.
        
    Returns:
        AdvancedSABFTValidator: Initialized SA-BFT validator
    """
    if config is None:
        config = SA_BFT_CONFIG
    
    return AdvancedSABFTValidator(
        embed_dim=config["embed_dim"],
        max_peers=config["max_peers"]
    )
