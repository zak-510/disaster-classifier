import os
import yaml
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import wandb
from typing import Dict, Any, Optional

class BaseModel:
    def __init__(self, config_path: str, model_name: str):
        """
        Initialize base model with configuration and logging setup.
        
        Args:
            config_path: Path to YAML configuration file
            model_name: Name of the model (localization or damage_classification)
        """
        self.config = self._load_config(config_path)
        self.model_config = self.config[model_name]
        self.logging_config = self.config['logging']
        
        # Setup logging
        self._setup_logging()
        
        # Setup checkpoint directory
        self.checkpoint_dir = Path(self.logging_config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics tracking
        self.best_metric = float('-inf') if self.logging_config['mode'] == 'max' else float('inf')
        self.current_epoch = 0
        
        # Setup tensorboard
        if self.logging_config['tensorboard']:
            self.writer = SummaryWriter(log_dir=str(self.checkpoint_dir / 'tensorboard'))
        
        # Setup wandb
        if self.logging_config['wandb']:
            wandb.init(
                project=self.logging_config['wandb_project'],
                config=self.config
            )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.checkpoint_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_metric': self.best_metric
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint if metric improved
        if self._is_best_metric():
            torch.save(checkpoint, self.checkpoint_dir / 'best.pth')
            self.logger.info(f"New best model saved with metric: {self.best_metric:.4f}")
    
    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       checkpoint_path: Optional[str] = None) -> int:
        """Load model checkpoint."""
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / 'latest.pth'
        
        if not os.path.exists(checkpoint_path):
            self.logger.warning(f"No checkpoint found at {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        return self.current_epoch
    
    def _is_best_metric(self) -> bool:
        """Check if current metric is the best so far."""
        if self.logging_config['mode'] == 'max':
            return self.current_metric > self.best_metric
        return self.current_metric < self.best_metric
    
    def update_metrics(self, metrics: Dict[str, float], phase: str = 'train'):
        """Update and log metrics."""
        self.current_metric = metrics[self.logging_config['monitor']]
        
        # Update best metric if improved
        if self._is_best_metric():
            self.best_metric = self.current_metric
        
        # Log to tensorboard
        if self.logging_config['tensorboard']:
            for name, value in metrics.items():
                self.writer.add_scalar(f'{phase}/{name}', value, self.current_epoch)
        
        # Log to wandb
        if self.logging_config['wandb']:
            wandb.log({f'{phase}/{name}': value for name, value in metrics.items()},
                     step=self.current_epoch)
        
        # Log to console
        metrics_str = ' - '.join([f'{name}: {value:.4f}' for name, value in metrics.items()])
        self.logger.info(f'Epoch {self.current_epoch} - {phase} - {metrics_str}')
    
    def close(self):
        """Clean up resources."""
        if self.logging_config['tensorboard']:
            self.writer.close()
        if self.logging_config['wandb']:
            wandb.finish() 