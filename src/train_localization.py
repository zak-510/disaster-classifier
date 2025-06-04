import os
import yaml
import torch
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb

from models.localization.unet import UNet
from data.xbd_dataset import XBDDataset
from utils.augmentations import get_localization_augmentations
from utils.metrics import calculate_localization_metrics
from models.base_model import BaseModel

def parse_args():
    parser = argparse.ArgumentParser(description='Train building localization model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to xBD dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--test_mode', action='store_true', help='Run in test mode with small dataset')
    return parser.parse_args()

def train_epoch(model, dataloader, optimizer, device, base_model):
    model.train()
    epoch_metrics = []
    
    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        preds = model(images)
        loss = torch.nn.BCELoss()(preds, masks)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        metrics = calculate_localization_metrics(preds, masks)
        metrics['loss'] = loss.item()
        epoch_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in epoch_metrics])
        for k in epoch_metrics[0].keys()
    }
    
    return avg_metrics

def validate(model, dataloader, device, base_model):
    model.eval()
    epoch_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            preds = model(images)
            loss = torch.nn.BCELoss()(preds, masks)
            
            # Calculate metrics
            metrics = calculate_localization_metrics(preds, masks)
            metrics['loss'] = loss.item()
            epoch_metrics.append(metrics)
    
    # Average metrics
    avg_metrics = {
        k: np.mean([m[k] for m in epoch_metrics])
        for k in epoch_metrics[0].keys()
    }
    
    return avg_metrics

def main():
    args = parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize base model for logging and checkpointing
    base_model = BaseModel(args.config, 'localization')
    
    # Create datasets
    train_transform = get_localization_augmentations(config['localization']['data'], 'train')
    val_transform = get_localization_augmentations(config['localization']['data'], 'val')
    
    train_dataset = XBDDataset(
        args.data_dir,
        transform=train_transform,
        phase='train'
    )
    val_dataset = XBDDataset(
        args.data_dir,
        transform=val_transform,
        phase='val'
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['localization']['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['localization']['training']['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = UNet(
        in_channels=3,
        out_channels=1,
        features=config['localization']['model'].get('features', [64, 128, 256, 512, 1024])
    ).to(device)
    
    # Create optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=config['localization']['training']['learning_rate'],
        weight_decay=config['localization']['training']['weight_decay']
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['localization']['training']['num_epochs']
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        start_epoch = base_model.load_checkpoint(model, optimizer, scheduler, args.resume)
    
    # Training loop
    for epoch in range(start_epoch, config['localization']['training']['num_epochs']):
        base_model.current_epoch = epoch
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, base_model)
        base_model.update_metrics(train_metrics, 'train')
        
        # Validate
        val_metrics = validate(model, val_loader, device, base_model)
        base_model.update_metrics(val_metrics, 'val')
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['save_frequency'] == 0:
            base_model.save_checkpoint(model, optimizer, scheduler)
    
    # Save final model
    base_model.save_checkpoint(model, optimizer, scheduler)
    base_model.close()

if __name__ == '__main__':
    main() 