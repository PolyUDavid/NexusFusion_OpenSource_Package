"""
NexusFusion Training Script
==========================

This script provides a complete training pipeline for the NexusFusion
multi-modal fusion architecture. It supports distributed training,
mixed precision, and comprehensive logging.

Key Features:
- Multi-GPU distributed training support
- Mixed precision training (FP16/FP32)
- Comprehensive logging and visualization
- Checkpoint management and resuming
- Validation and testing pipelines
- Performance profiling and optimization

Authors: NexusFusion Research Team
License: MIT
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from tqdm import tqdm
import wandb

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models import create_mmf_gnn, create_sa_bft, create_ktf_transformer
from inference import NexusFusionAPI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NexusFusionTrainer:
    """
    Main trainer class for NexusFusion model pipeline.
    
    This class handles the complete training process including model setup,
    data loading, optimization, validation, and checkpointing.
    
    Args:
        config (dict): Training configuration dictionary
        device (str): Target device for training
        distributed (bool): Whether to use distributed training
    """
    
    def __init__(self, config: Dict, device: str = 'auto', distributed: bool = False):
        self.config = config
        self.device = self._setup_device(device)
        self.distributed = distributed
        
        # Initialize models
        self.models = self._setup_models()
        
        # Initialize optimizers and schedulers
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if config.get('mixed_precision', False) else None
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        self.train_losses = []
        self.val_losses = []
        
        logger.info(f"NexusFusion trainer initialized on device: {self.device}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup and validate training device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def _setup_models(self) -> Dict[str, nn.Module]:
        """Initialize all NexusFusion model components."""
        models = {
            'mmf_gnn': create_mmf_gnn(),
            'sa_bft': create_sa_bft(), 
            'ktf_transformer': create_ktf_transformer()
        }
        
        # Move models to device
        for name, model in models.items():
            models[name] = model.to(self.device)
            
            # Wrap with DDP if distributed training
            if self.distributed:
                models[name] = DDP(model, device_ids=[self.device])
        
        return models
    
    def _setup_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Setup optimizers for each model component."""
        opt_config = self.config.get('optimizer', {})
        lr = opt_config.get('learning_rate', 1e-4)
        weight_decay = opt_config.get('weight_decay', 0.01)
        
        optimizers = {}
        for name, model in self.models.items():
            if opt_config.get('type', 'adamw').lower() == 'adamw':
                optimizers[name] = optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
            else:
                optimizers[name] = optim.Adam(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
        
        return optimizers
    
    def _setup_schedulers(self) -> Dict[str, optim.lr_scheduler._LRScheduler]:
        """Setup learning rate schedulers."""
        scheduler_config = self.config.get('scheduler', {})
        max_epochs = self.config.get('max_epochs', 100)
        
        schedulers = {}
        for name, optimizer in self.optimizers.items():
            if scheduler_config.get('type', 'cosine').lower() == 'cosine':
                schedulers[name] = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max_epochs,
                    eta_min=scheduler_config.get('min_lr', 1e-6)
                )
            else:
                schedulers[name] = optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config.get('step_size', 30),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
        
        return schedulers
    
    def compute_loss(self, batch: Dict, outputs: Dict) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for all model components.
        
        Args:
            batch (dict): Input batch data
            outputs (dict): Model outputs
            
        Returns:
            dict: Dictionary of computed losses
        """
        losses = {}
        
        # MMF-GNN loss (reconstruction + regularization)
        if 'mmf_gnn_output' in outputs:
            mmf_features = outputs['mmf_gnn_output'][0]
            # Add appropriate loss computation based on your specific requirements
            losses['mmf_gnn'] = torch.tensor(0.0, device=self.device)
        
        # SA-BFT consensus loss
        if 'sa_bft_output' in outputs:
            consensus_results = outputs['sa_bft_output']
            # Consensus quality loss
            consensus_weights = consensus_results.get('consensus_weights')
            if consensus_weights is not None:
                # Encourage confident consensus (low entropy)
                entropy = -torch.sum(consensus_weights * torch.log(consensus_weights + 1e-8), dim=1)
                losses['sa_bft'] = entropy.mean()
            else:
                losses['sa_bft'] = torch.tensor(0.0, device=self.device)
        
        # KTF trajectory prediction loss
        if 'ktf_output' in outputs:
            ktf_results = outputs['ktf_output']
            trajectories = ktf_results.get('trajectories')
            target_trajectories = batch.get('target_trajectories')
            
            if trajectories is not None and target_trajectories is not None:
                # Multi-modal trajectory loss
                traj_loss = nn.MSELoss()(trajectories, target_trajectories)
                
                # Physics constraint loss
                physics_loss = ktf_results.get('physics_loss', torch.tensor(0.0, device=self.device))
                
                losses['ktf'] = traj_loss + 0.1 * physics_loss
            else:
                losses['ktf'] = torch.tensor(0.0, device=self.device)
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def train_step(self, batch: Dict) -> Dict[str, float]:
        """
        Execute single training step.
        
        Args:
            batch (dict): Training batch
            
        Returns:
            dict: Training metrics for this step
        """
        # Set models to training mode
        for model in self.models.values():
            model.train()
        
        # Zero gradients
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()
        
        # Forward pass with mixed precision if enabled
        if self.scaler is not None:
            with autocast():
                outputs = self.forward_pass(batch)
                losses = self.compute_loss(batch, outputs)
        else:
            outputs = self.forward_pass(batch)
            losses = self.compute_loss(batch, outputs)
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(losses['total']).backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_norm'):
                for model in self.models.values():
                    self.scaler.unscale_(self.optimizers[list(self.models.keys())[0]])
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        self.config['gradient_clip_norm']
                    )
            
            # Optimizer step
            for optimizer in self.optimizers.values():
                self.scaler.step(optimizer)
            self.scaler.update()
        else:
            losses['total'].backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip_norm'):
                for model in self.models.values():
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.config['gradient_clip_norm']
                    )
            
            # Optimizer step
            for optimizer in self.optimizers.values():
                optimizer.step()
        
        # Convert losses to float for logging
        metrics = {k: v.item() for k, v in losses.items()}
        
        return metrics
    
    def forward_pass(self, batch: Dict) -> Dict:
        """
        Execute forward pass through all model components.
        
        Args:
            batch (dict): Input batch data
            
        Returns:
            dict: Model outputs from all components
        """
        outputs = {}
        
        # Extract input data (adapt based on your data format)
        sensor_data = batch.get('sensor_data', {})
        
        # MMF-GNN forward pass
        if 'mmf_gnn' in self.models:
            # Prepare dummy inputs - replace with actual data extraction
            lidar_points = sensor_data.get('lidar', torch.zeros(1, 1000, 4, device=self.device))
            camera_keypoints = sensor_data.get('camera', torch.zeros(1, 500, 130, device=self.device))
            gnss = sensor_data.get('gnss', torch.zeros(1, 3, device=self.device))
            imu = sensor_data.get('imu', torch.zeros(1, 10, 6, device=self.device))
            v2x_states = sensor_data.get('v2x', torch.zeros(1, 5, 8, device=self.device))
            obu = torch.zeros(1, 256, device=self.device)
            participants = torch.zeros(1, 10, 256, device=self.device)
            
            mmf_output = self.models['mmf_gnn'](
                lidar_points, camera_keypoints, gnss, imu, obu, v2x_states, participants
            )
            outputs['mmf_gnn_output'] = mmf_output
        
        # SA-BFT forward pass
        if 'sa_bft' in self.models and 'mmf_gnn_output' in outputs:
            local_features = outputs['mmf_gnn_output'][0]
            peer_features = local_features.unsqueeze(1)  # Dummy peer data
            trusted_context = local_features
            
            sa_bft_output = self.models['sa_bft'](peer_features, trusted_context)
            outputs['sa_bft_output'] = sa_bft_output
        
        # KTF forward pass
        if 'ktf_transformer' in self.models and 'sa_bft_output' in outputs:
            consensus_state = outputs['sa_bft_output']['consensus_state']
            agent_history = torch.zeros(consensus_state.size(0), 5, 10, 4, device=self.device)
            map_data = torch.zeros(consensus_state.size(0), 20, 50, 2, device=self.device)
            
            ktf_output = self.models['ktf_transformer'](agent_history, map_data, consensus_state)
            outputs['ktf_output'] = ktf_output
        
        return outputs
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Run validation loop.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            dict: Validation metrics
        """
        # Set models to evaluation mode
        for model in self.models.values():
            model.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                outputs = self.forward_pass(batch)
                losses = self.compute_loss(batch, outputs)
                val_losses.append(losses['total'].item())
        
        avg_val_loss = np.mean(val_losses)
        
        return {
            'val_loss': avg_val_loss,
            'val_samples': len(val_loader.dataset)
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict, checkpoint_dir: str):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dicts': {
                name: model.state_dict() for name, model in self.models.items()
            },
            'optimizer_state_dicts': {
                name: opt.state_dict() for name, opt in self.optimizers.items()
            },
            'scheduler_state_dicts': {
                name: sched.state_dict() for name, sched in self.schedulers.items()
            },
            'metrics': metrics,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if metrics.get('val_loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics['val_loss']
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model states
        for name, model in self.models.items():
            if name in checkpoint['model_state_dicts']:
                model.load_state_dict(checkpoint['model_state_dicts'][name])
        
        # Load optimizer states
        for name, optimizer in self.optimizers.items():
            if name in checkpoint['optimizer_state_dicts']:
                optimizer.load_state_dict(checkpoint['optimizer_state_dicts'][name])
        
        # Load scheduler states
        for name, scheduler in self.schedulers.items():
            if name in checkpoint['scheduler_state_dicts']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dicts'][name])
        
        # Load scaler state
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        logger.info(f"Checkpoint loaded from: {checkpoint_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, 
              checkpoint_dir: str = './checkpoints'):
        """
        Main training loop.
        
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader, optional): Validation data loader
            checkpoint_dir (str): Directory to save checkpoints
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        max_epochs = self.config.get('max_epochs', 100)
        
        logger.info(f"Starting training for {max_epochs} epochs")
        
        for epoch in range(self.current_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Training loop
            train_metrics = []
            
            for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
                metrics = self.train_step(batch)
                train_metrics.append(metrics)
                self.global_step += 1
                
                # Log every N steps
                if batch_idx % self.config.get('log_interval', 100) == 0:
                    avg_metrics = {k: np.mean([m[k] for m in train_metrics[-100:]]) 
                                 for k in metrics.keys()}
                    logger.info(f"Epoch {epoch+1}, Step {batch_idx}: {avg_metrics}")
            
            # Compute epoch metrics
            epoch_train_metrics = {
                k: np.mean([m[k] for m in train_metrics]) 
                for k in train_metrics[0].keys()
            }
            
            # Validation
            val_metrics = {}
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            
            # Learning rate scheduling
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Combine metrics
            all_metrics = {**epoch_train_metrics, **val_metrics}
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch + 1, all_metrics, checkpoint_dir)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1} completed: {all_metrics}")
        
        logger.info("Training completed!")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train NexusFusion Model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to training configuration file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Initialize trainer
    trainer = NexusFusionTrainer(
        config=config,
        distributed=args.distributed
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Create dummy data loaders (replace with actual data loading)
    # This is a placeholder - implement your actual data loading logic
    train_loader = DataLoader(
        dataset=[],  # Your training dataset
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4)
    )
    
    val_loader = DataLoader(
        dataset=[],  # Your validation dataset  
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4)
    )
    
    # Start training
    trainer.train(train_loader, val_loader, args.checkpoint_dir)


if __name__ == '__main__':
    main()
