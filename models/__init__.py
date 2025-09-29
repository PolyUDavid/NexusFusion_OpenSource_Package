"""
NexusFusion Models Package
=========================

This package contains the core neural network models for the NexusFusion
multi-modal autonomous driving fusion architecture.

Models:
- MMF-GNN: Multi-Modal Fusion Graph Neural Network
- SA-BFT: Semantic-Aware Byzantine Fault-Tolerant Consensus
- KTF: Kinodynamic Trajectory Forecaster Transformer

Authors: NexusFusion Research Team
License: MIT
"""

from .mmf_gnn import AdvancedMMF_GNN, create_mmf_gnn, MMF_GNN_CONFIG
from .sa_bft import AdvancedSABFTValidator, create_sa_bft, SA_BFT_CONFIG
from .ktf_transformer import AdvancedKTFTransformer, create_ktf_transformer, KTF_CONFIG

__all__ = [
    # MMF-GNN exports
    'AdvancedMMF_GNN',
    'create_mmf_gnn',
    'MMF_GNN_CONFIG',
    
    # SA-BFT exports
    'AdvancedSABFTValidator',
    'create_sa_bft',
    'SA_BFT_CONFIG',
    
    # KTF exports
    'AdvancedKTFTransformer',
    'create_ktf_transformer',
    'KTF_CONFIG',
]

# Model parameter counts (approximate)
MODEL_PARAMS = {
    'mmf_gnn': '8.2M',      # Multi-Modal Fusion GNN parameters
    'sa_bft': '2.1M',       # SA-BFT Consensus parameters  
    'ktf_transformer': '5.3M' # KTF Transformer parameters
}

# Total NexusFusion model size
TOTAL_PARAMS = '15.6M'

def get_model_info():
    """
    Get information about all NexusFusion models.
    
    Returns:
        dict: Model information including parameter counts and descriptions
    """
    return {
        'models': {
            'mmf_gnn': {
                'class': 'AdvancedMMF_GNN',
                'params': MODEL_PARAMS['mmf_gnn'],
                'description': 'Multi-Modal Fusion Graph Neural Network for sensor fusion',
                'input_modalities': ['lidar', 'camera', 'gnss', 'imu', 'v2x'],
                'output_dim': 256
            },
            'sa_bft': {
                'class': 'AdvancedSABFTValidator',
                'params': MODEL_PARAMS['sa_bft'],
                'description': 'Semantic-Aware Byzantine Fault-Tolerant consensus mechanism',
                'max_peers': 20,
                'output': 'consensus_state'
            },
            'ktf_transformer': {
                'class': 'AdvancedKTFTransformer',
                'params': MODEL_PARAMS['ktf_transformer'],
                'description': 'Kinodynamic Trajectory Forecaster with physics constraints',
                'prediction_modes': 6,
                'horizon': 30
            }
        },
        'total_params': TOTAL_PARAMS,
        'architecture': 'Vertically Integrated Dual-AI',
        'framework': 'PyTorch'
    }

def create_nexus_fusion_model(device='cpu'):
    """
    Create complete NexusFusion model pipeline.
    
    Args:
        device (str): Target device ('cpu', 'cuda', 'mps')
        
    Returns:
        dict: Dictionary containing all three models
    """
    models = {
        'mmf_gnn': create_mmf_gnn().to(device),
        'sa_bft': create_sa_bft().to(device),
        'ktf_transformer': create_ktf_transformer().to(device)
    }
    
    # Set models to evaluation mode by default
    for model in models.values():
        model.eval()
    
    return models
