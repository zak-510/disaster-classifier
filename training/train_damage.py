import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import numpy as np
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.damage_model import create_damage_model, calculate_accuracy
from data_processing.damage_data import get_damage_data_loaders

def test_configuration(batch_size, learning_rate, max_batches=50):
    """Test a specific batch size and learning rate configuration"""
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    data_dir = os.path.join(project_root, 'Data')
    model = create_damage_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5, eps=1e-8)
    
    train_loader, _ = get_damage_data_loaders(
        data_dir, 
        batch_size=batch_size, 
        patch_size=64,
        num_workers=2,
        prefetch_factor=1
    )
    
    if train_loader is None:
        return False, 0, 0, 0
    
    model.train()
    successful_batches = 0
    nan_batches = 0
    total_loss = 0.0
    total_acc = 0.0
    
    for batch_idx, (patches, labels) in enumerate(train_loader):
        if batch_idx >= max_batches:
            break
            
        patches = patches.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        outputs = model(patches)
        
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            nan_batches += 1
            continue
        
        loss = criterion(outputs, labels)
        
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
            nan_batches += 1
            continue
        
        loss.backward()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        if grad_norm > 10.0 or torch.isnan(grad_norm):
            nan_batches += 1
            continue
        
        optimizer.step()
        
        successful_batches += 1
        total_loss += loss.item()
        total_acc += calculate_accuracy(outputs, labels)
    
    if successful_batches > 0:
        avg_loss = total_loss / successful_batches
        avg_acc = total_acc / successful_batches
        success_rate = successful_batches / (successful_batches + nan_batches) * 100
        return True, success_rate, avg_loss, avg_acc
    else:
        return False, 0, 0, 0

def find_optimal_settings():
    print(f'PROGRESSIVE PARAMETER OPTIMIZATION')
    print('=' * 60)
    
    # Test configurations in order of increasing aggressiveness
    test_configs = [
        # (batch_size, learning_rate, description)
        (8, 1e-6, "Ultra-Conservative (baseline)"),
        (8, 5e-6, "Conservative LR increase"),
        (8, 1e-5, "Moderate LR"),
        (8, 2e-5, "Higher LR"),
        (8, 5e-5, "Aggressive LR"),
        (8, 1e-4, "Very Aggressive LR"),
        (16, 1e-6, "Double batch size, reset LR"),
        (16, 1e-5, "Double batch + moderate LR"),
        (16, 5e-5, "Double batch + aggressive LR"),
        (32, 1e-6, "Large batch, conservative LR"),
        (32, 1e-5, "Large batch + moderate LR"),
        (32, 2e-5, "Large batch + higher LR"),
    ]
    
    results = []
    
    for batch_size, lr, description in test_configs:
        print(f'\nTesting: {description}')
        print(f'Batch size: {batch_size}, Learning rate: {lr}')
        
        success = test_configuration(batch_size, lr, max_batches=100)
        
        if success:
            status = "STABLE"
            results.append((batch_size, lr, description))
        else:
            if len(results) > 0:
                print(f"Found instability, stopping at {len(results)} stable configs")
                break
            status = "UNSTABLE" if success else "FAILED"
        
        print(f'Result: {status}')
        
        torch.cuda.empty_cache()
    
    print('\nSTABILITY TEST RESULTS:')
    print('=' * 40)
    for batch_size, lr, desc in results:
        print(f'STABLE: {desc} (BS={batch_size}, LR={lr})')
    
    print(f"STABLE CONFIGURATIONS FOUND: {len(results)}")
    
    if results:
        # Use the most aggressive stable configuration
        best_batch_size, best_lr, best_desc = results[-1]
        print(f'\nOPTIMAL CONFIGURATION SELECTED:')
        print(f'Configuration: {best_desc}')
        print(f'Batch size: {best_batch_size}')
        print(f'Learning rate: {best_lr}')
        print('\nProceeding with optimized training using these parameters...')
        
        return best_batch_size, best_lr
    else:
        print("NO STABLE CONFIGURATIONS FOUND")
        print("Falling back to ultra-conservative settings:")
        return 8, 1e-6

def run_optimized_training(batch_size, learning_rate, epochs=10):
    """Run full training with optimized parameters"""
    print(f'\nRUNNING OPTIMIZED TRAINING')
    print('=' * 60)
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {learning_rate:.0e}')
    print(f'Epochs: {epochs}')
    
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    
    data_dir = os.path.join(project_root, 'Data')
    model = create_damage_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4, eps=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    train_loader, val_loader = get_damage_data_loaders(
        data_dir, 
        batch_size=batch_size, 
        patch_size=64,
        num_workers=4,
        prefetch_factor=2
    )
    
    if train_loader is None:
        print('Failed to create data loaders!')
        return
    
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        # Training
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        successful_batches = 0
        failed_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (patches, labels) in enumerate(progress_bar):
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 100:
                failed_batches += 1
                if failed_batches > 10:  # Too many failures
                    print(f"\nERROR: Too many failed batches ({failed_batches}), stopping epoch")
                    break
                continue
            
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            if torch.isnan(grad_norm) or grad_norm > 10.0:
                failed_batches += 1
                continue
                
            optimizer.step()
            
            successful_batches += 1
            train_loss += loss.item()
            train_acc += calculate_accuracy(outputs, labels)
            
            if batch_idx % 200 == 0:
                current_acc = train_acc / successful_batches if successful_batches > 0 else 0
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.4f}',
                    'Failed': failed_batches
                })
        
        if successful_batches == 0:
            print(f"ERROR: Epoch {epoch+1} failed completely, stopping training")
            break
            
        avg_train_loss = train_loss / successful_batches
        avg_train_acc = train_acc / successful_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        
        with torch.no_grad():
            for patches, labels in tqdm(val_loader, desc='Validation'):
                patches = patches.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(patches)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}')
        
        # Save model checkpoint
        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            
            # Ensure the directory exists
            weights_dir = os.path.join(project_root, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            model_save_path = os.path.join(weights_dir, 'best_damage_model_optimized.pth')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'batch_size': batch_size,
                'learning_rate': learning_rate
            }, model_save_path)
            
            print(f'Best model saved to {model_save_path} (Accuracy: {best_acc:.4f})')
        
        scheduler.step(avg_val_loss)
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_acc:.4f} ({best_acc*100:.1f}%)')

if __name__ == '__main__':
    # Find optimal parameters
    optimal_batch_size, optimal_lr = find_optimal_settings()
    
    # Run training with optimal parameters
    run_optimized_training(optimal_batch_size, optimal_lr, epochs=15) 