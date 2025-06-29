import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import datetime

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model, calculate_iou
from data_processing.localization_data import get_proper_data_loaders

def train_model():
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device = torch.device('cuda')
    print(f'Using device: {device}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
    
    torch.cuda.empty_cache()
    
    data_dir = os.path.join(project_root, 'Data')
    if not os.path.exists(data_dir):
        print(f'Data directory {data_dir} not found!')
        return
    
    # Create model and move to GPU
    model = create_model().to(device)
    
    # Use weighted BCE loss with reduced weight to prevent NaN
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([4.0]).to(device))
    
    # Use AdamW with lower learning rate and gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,  # Reduced from 1e-3
        weight_decay=0.01,
        eps=1e-8  # Increased epsilon for stability
    )
    
    # Smaller batch size but more accumulation steps
    batch_size = 2  # Reduced from 4
    accumulation_steps = 8  # Increased from 4, effective batch size still 16
    print(f'Batch size: {batch_size} (effective: {batch_size * accumulation_steps})')
    
    train_loader, val_loader = get_proper_data_loaders(
        data_dir,
        batch_size=batch_size,
        num_workers=2,
        prefetch_factor=2
    )
    
    if train_loader is None:
        print('Failed to create proper data loaders!')
        return
    
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    # Setup mixed precision training
    scaler = GradScaler(
        init_scale=2**10,  # Start with a lower scale
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=100
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=5e-4,
        epochs=30,
        steps_per_epoch=len(train_loader) // accumulation_steps,
        pct_start=0.3,  # Longer warmup
        div_factor=10,
        final_div_factor=100
    )
    
    epochs = 30
    best_iou = 0.0
    patience = 0
    max_patience = 7
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(project_root, 'models', 'checkpoints', 'new_training')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print('\nStarting training...')
    print('=' * 60)
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current learning rate: {current_lr:.6f}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        optimizer.zero_grad(set_to_none=True)
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader, desc='Training')):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            # Mixed precision training
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks.float()) / accumulation_steps
            
            if torch.isnan(loss):
                print(f'NaN loss at batch {batch_idx}, skipping...')
                optimizer.zero_grad(set_to_none=True)
                continue
            
            # Gradient accumulation
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients more aggressively
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            with torch.no_grad():
                train_loss += loss.item() * accumulation_steps
                batch_iou = calculate_iou(outputs, masks)
                train_iou += batch_iou
            
            # Print progress and clear memory periodically
            if batch_idx % 100 == 0:
                print(f'\nBatch {batch_idx}/{len(train_loader)}')
                print(f'Loss: {loss.item():.4f} | IoU: {batch_iou:.4f}')
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc='Validation'):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks.float())
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print('-' * 60)
        print(f'Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}')
        print('-' * 60)
        
        # Save model if validation IoU improves
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            patience = 0
            
            # Save timestamped version
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch:02d}_{timestamp}.pth')
            
            # Also save as best model
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            
            # Save full checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_iou': best_iou,
                'training_config': {
                    'batch_size': batch_size,
                    'accumulation_steps': accumulation_steps,
                    'learning_rate': current_lr
                }
            }
            
            # Save both versions
            torch.save(checkpoint, model_path)
            torch.save(checkpoint, best_model_path)
            
            print(f'✨ New best model saved! IoU: {best_iou:.4f}')
            print(f'   Saved to: {model_path}')
            print(f'   Best model also saved to: {best_model_path}')
        else:
            patience += 1
            if patience >= max_patience:
                print(f'Early stopping triggered! No improvement for {max_patience} epochs.')
                break
        
        # Print memory stats
        max_mem = torch.cuda.max_memory_allocated(0) // 1024**2
        print(f'Max GPU Memory: {max_mem}MB')
        torch.cuda.empty_cache()
    
    print(f'\nTraining complete! Best IoU: {best_iou:.4f}')

if __name__ == '__main__':
    train_model() 