"""
NexusFusion Inference API
=========================

This module provides a high-level API for running inference with the complete
NexusFusion model pipeline. It handles model loading, data preprocessing,
and end-to-end prediction for autonomous driving applications.

Key Features:
- Complete NexusFusion pipeline integration
- Multi-modal sensor data processing
- Real-time inference optimization
- Batch processing support
- Device-agnostic deployment (CPU/GPU/MPS)

Authors: NexusFusion Research Team
License: MIT
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import (
    create_mmf_gnn, create_sa_bft, create_ktf_transformer,
    MMF_GNN_CONFIG, SA_BFT_CONFIG, KTF_CONFIG
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NexusFusionAPI:
    """
    Main API class for NexusFusion inference.
    
    This class provides a unified interface for running the complete NexusFusion
    pipeline including multi-modal fusion, Byzantine consensus, and trajectory
    prediction. It handles model loading, data validation, and optimization.
    
    Args:
        model_dir (str): Directory containing model checkpoints
        device (str): Target device ('cpu', 'cuda', 'mps', 'auto')
        precision (str): Inference precision ('fp32', 'fp16')
        batch_size (int): Maximum batch size for inference
    """
    
    def __init__(self, model_dir: str = None, device: str = 'auto', 
                 precision: str = 'fp32', batch_size: int = 32):
        self.model_dir = Path(model_dir) if model_dir else None
        self.device = self._setup_device(device)
        self.precision = precision
        self.batch_size = batch_size
        
        # Model components
        self.mmf_gnn = None
        self.sa_bft = None
        self.ktf_transformer = None
        
        # Performance tracking
        self.inference_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'avg_latency': 0.0,
            'last_update': time.time()
        }
        
        logger.info(f"NexusFusion API initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate the target device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        torch_device = torch.device(device)
        logger.info(f"Using device: {torch_device}")
        return torch_device
    
    def load_models(self, checkpoint_paths: Optional[Dict[str, str]] = None):
        """
        Load all NexusFusion model components.
        
        Args:
            checkpoint_paths (dict, optional): Paths to model checkpoints
                Format: {'mmf_gnn': path, 'sa_bft': path, 'ktf_transformer': path}
        """
        logger.info("Loading NexusFusion models...")
        
        try:
            # Create model instances
            self.mmf_gnn = create_mmf_gnn().to(self.device)
            self.sa_bft = create_sa_bft().to(self.device)
            self.ktf_transformer = create_ktf_transformer().to(self.device)
            
            # Load checkpoints if provided
            if checkpoint_paths:
                self._load_checkpoints(checkpoint_paths)
            
            # Set models to evaluation mode
            self.mmf_gnn.eval()
            self.sa_bft.eval()
            self.ktf_transformer.eval()
            
            # Enable mixed precision if requested
            if self.precision == 'fp16':
                self.mmf_gnn = self.mmf_gnn.half()
                self.sa_bft = self.sa_bft.half()
                self.ktf_transformer = self.ktf_transformer.half()
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_checkpoints(self, checkpoint_paths: Dict[str, str]):
        """Load model checkpoints from disk."""
        for model_name, path in checkpoint_paths.items():
            if not os.path.exists(path):
                logger.warning(f"Checkpoint not found: {path}")
                continue
                
            try:
                checkpoint = torch.load(path, map_location=self.device)
                
                if model_name == 'mmf_gnn':
                    self.mmf_gnn.load_state_dict(checkpoint['model_state_dict'])
                elif model_name == 'sa_bft':
                    self.sa_bft.load_state_dict(checkpoint['model_state_dict'])
                elif model_name == 'ktf_transformer':
                    self.ktf_transformer.load_state_dict(checkpoint['model_state_dict'])
                
                logger.info(f"Loaded checkpoint for {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load checkpoint for {model_name}: {e}")
    
    def preprocess_sensor_data(self, sensor_data: Dict) -> Dict[str, torch.Tensor]:
        """
        Preprocess multi-modal sensor data for model input.
        
        Args:
            sensor_data (dict): Raw sensor data containing:
                - lidar_points: LiDAR point cloud [N, 4] (x, y, z, intensity)
                - camera_keypoints: Camera features [M, 130] (2D + descriptor)
                - gnss: GPS coordinates [3] (lat, lon, alt)
                - imu: IMU sequence [T, 6] (3-axis accel + gyro)
                - v2x_states: V2X vehicle states [K, 8]
                
        Returns:
            dict: Preprocessed tensor data ready for model input
        """
        processed = {}
        
        try:
            # LiDAR preprocessing
            if 'lidar_points' in sensor_data:
                lidar = np.array(sensor_data['lidar_points'])
                if len(lidar.shape) == 2 and lidar.shape[1] >= 4:
                    processed['lidar_points'] = torch.tensor(
                        lidar[:, :4], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)  # Add batch dimension
            
            # Camera preprocessing
            if 'camera_keypoints' in sensor_data:
                camera = np.array(sensor_data['camera_keypoints'])
                if len(camera.shape) == 2 and camera.shape[1] >= 130:
                    processed['camera_keypoints'] = torch.tensor(
                        camera[:, :130], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
            
            # GNSS preprocessing
            if 'gnss' in sensor_data:
                gnss = np.array(sensor_data['gnss'])
                if len(gnss.shape) == 1 and gnss.shape[0] >= 3:
                    processed['gnss'] = torch.tensor(
                        gnss[:3], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
            
            # IMU preprocessing
            if 'imu' in sensor_data:
                imu = np.array(sensor_data['imu'])
                if len(imu.shape) == 2 and imu.shape[1] >= 6:
                    processed['imu'] = torch.tensor(
                        imu[:, :6], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
            
            # V2X preprocessing
            if 'v2x_states' in sensor_data:
                v2x = np.array(sensor_data['v2x_states'])
                if len(v2x.shape) == 2 and v2x.shape[1] >= 8:
                    processed['v2x_states'] = torch.tensor(
                        v2x[:, :8], dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise ValueError(f"Invalid sensor data format: {e}")
        
        return processed
    
    def run_mmf_gnn(self, sensor_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Run Multi-Modal Fusion GNN inference.
        
        Args:
            sensor_data (dict): Preprocessed sensor data tensors
            
        Returns:
            tuple: (fused_features, auxiliary_outputs)
        """
        if self.mmf_gnn is None:
            raise RuntimeError("MMF-GNN model not loaded")
        
        # Prepare inputs with defaults for missing modalities
        lidar_points = sensor_data.get('lidar_points', torch.zeros(1, 1000, 4, device=self.device))
        camera_keypoints = sensor_data.get('camera_keypoints', torch.zeros(1, 500, 130, device=self.device))
        gnss = sensor_data.get('gnss', torch.zeros(1, 3, device=self.device))
        imu = sensor_data.get('imu', torch.zeros(1, 10, 6, device=self.device))
        v2x_states = sensor_data.get('v2x_states', torch.zeros(1, 5, 8, device=self.device))
        
        # Dummy inputs for compatibility
        obu = torch.zeros(1, 256, device=self.device)
        participants = torch.zeros(1, 10, 256, device=self.device)
        
        with torch.no_grad():
            fused_features, aux_outputs = self.mmf_gnn(
                lidar_points, camera_keypoints, gnss, imu, obu, v2x_states, participants
            )
        
        return fused_features, aux_outputs
    
    def run_sa_bft(self, local_features: torch.Tensor, 
                   peer_features: List[torch.Tensor] = None) -> Dict:
        """
        Run SA-BFT consensus inference.
        
        Args:
            local_features (torch.Tensor): Local vehicle features [B, D]
            peer_features (list, optional): Peer vehicle features
            
        Returns:
            dict: Consensus results including trusted state and validity scores
        """
        if self.sa_bft is None:
            raise RuntimeError("SA-BFT model not loaded")
        
        # Prepare peer features
        if peer_features is None or len(peer_features) == 0:
            # Single vehicle scenario
            peer_features_tensor = local_features.unsqueeze(1)  # [B, 1, D]
        else:
            # Multi-vehicle scenario
            peer_list = [local_features] + peer_features
            peer_features_tensor = torch.stack(peer_list, dim=1)  # [B, M, D]
        
        trusted_context = local_features  # Use local features as trusted context
        
        with torch.no_grad():
            consensus_results = self.sa_bft(peer_features_tensor, trusted_context)
        
        return consensus_results
    
    def run_ktf_transformer(self, consensus_state: torch.Tensor,
                           agent_history: torch.Tensor = None,
                           map_data: torch.Tensor = None) -> Dict:
        """
        Run KTF Transformer trajectory prediction.
        
        Args:
            consensus_state (torch.Tensor): Trusted consensus state [B, D]
            agent_history (torch.Tensor, optional): Agent trajectory history [B, A, T, 4]
            map_data (torch.Tensor, optional): Map polyline data [B, L, P, 2]
            
        Returns:
            dict: Trajectory predictions with uncertainty and auxiliary outputs
        """
        if self.ktf_transformer is None:
            raise RuntimeError("KTF Transformer model not loaded")
        
        # Prepare inputs with defaults
        if agent_history is None:
            agent_history = torch.zeros(consensus_state.size(0), 5, 10, 4, device=self.device)
        if map_data is None:
            map_data = torch.zeros(consensus_state.size(0), 20, 50, 2, device=self.device)
        
        with torch.no_grad():
            trajectory_results = self.ktf_transformer(agent_history, map_data, consensus_state)
        
        return trajectory_results
    
    def predict(self, sensor_data: Dict, peer_data: List[Dict] = None) -> Dict:
        """
        Run complete NexusFusion inference pipeline.
        
        Args:
            sensor_data (dict): Multi-modal sensor data
            peer_data (list, optional): Peer vehicle sensor data for consensus
            
        Returns:
            dict: Complete prediction results including trajectories, consensus, and metrics
        """
        start_time = time.time()
        
        try:
            # Step 1: Preprocess sensor data
            processed_data = self.preprocess_sensor_data(sensor_data)
            
            # Step 2: Multi-modal fusion
            local_features, mmf_aux = self.run_mmf_gnn(processed_data)
            
            # Step 3: Process peer data if available
            peer_features = []
            if peer_data:
                for peer_sensors in peer_data:
                    peer_processed = self.preprocess_sensor_data(peer_sensors)
                    peer_feat, _ = self.run_mmf_gnn(peer_processed)
                    peer_features.append(peer_feat)
            
            # Step 4: Byzantine consensus
            consensus_results = self.run_sa_bft(local_features, peer_features)
            trusted_state = consensus_results['consensus_state']
            
            # Step 5: Trajectory prediction
            trajectory_results = self.run_ktf_transformer(trusted_state)
            
            # Update performance stats
            inference_time = time.time() - start_time
            self._update_stats(inference_time)
            
            # Compile final results
            results = {
                'trajectories': trajectory_results['trajectories'].cpu().numpy(),
                'mode_probabilities': trajectory_results['mode_probs'].cpu().numpy(),
                'collision_risk': trajectory_results['risk_scores'].cpu().numpy(),
                'intentions': trajectory_results['intentions'].cpu().numpy(),
                'consensus_confidence': consensus_results['confidence'].cpu().numpy(),
                'validity_scores': consensus_results['validity_scores'].cpu().numpy(),
                'inference_time_ms': inference_time * 1000,
                'num_peers': len(peer_features),
                'timestamp': time.time()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _update_stats(self, inference_time: float):
        """Update inference performance statistics."""
        self.inference_stats['total_calls'] += 1
        self.inference_stats['total_time'] += inference_time
        self.inference_stats['avg_latency'] = (
            self.inference_stats['total_time'] / self.inference_stats['total_calls']
        )
        self.inference_stats['last_update'] = time.time()
    
    def get_performance_stats(self) -> Dict:
        """Get inference performance statistics."""
        return self.inference_stats.copy()
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_stats = {
            'total_calls': 0,
            'total_time': 0.0,
            'avg_latency': 0.0,
            'last_update': time.time()
        }
    
    def save_config(self, config_path: str):
        """Save API configuration to file."""
        config = {
            'device': str(self.device),
            'precision': self.precision,
            'batch_size': self.batch_size,
            'model_configs': {
                'mmf_gnn': MMF_GNN_CONFIG,
                'sa_bft': SA_BFT_CONFIG,
                'ktf_transformer': KTF_CONFIG
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_config(cls, config_path: str):
        """Create API instance from configuration file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return cls(
            device=config.get('device', 'auto'),
            precision=config.get('precision', 'fp32'),
            batch_size=config.get('batch_size', 32)
        )


# Convenience functions
def create_api(model_dir: str = None, device: str = 'auto') -> NexusFusionAPI:
    """
    Create and initialize NexusFusion API.
    
    Args:
        model_dir (str, optional): Directory containing model checkpoints
        device (str): Target device
        
    Returns:
        NexusFusionAPI: Initialized API instance
    """
    api = NexusFusionAPI(model_dir=model_dir, device=device)
    api.load_models()
    return api


def quick_predict(sensor_data: Dict, model_dir: str = None) -> Dict:
    """
    Quick prediction function for single inference calls.
    
    Args:
        sensor_data (dict): Multi-modal sensor data
        model_dir (str, optional): Model checkpoint directory
        
    Returns:
        dict: Prediction results
    """
    api = create_api(model_dir=model_dir)
    return api.predict(sensor_data)
