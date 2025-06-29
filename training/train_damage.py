import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import time
import numpy as np
import sys
from sklearn.metrics import f1_score
from collections import Counter

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.damage_model import create_damage_model
from data_processing.damage_data import get_damage_data_loaders

class FocalLoss(nn.Module):
    """Focal Loss with improved numerical stability"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Apply log_softmax for better numerical stability
        log_probs = nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)
        
        # Get the log probability of the target class
        target_probs = probs[torch.arange(probs.shape[0]), targets]
        
        # Calculate focal weight
        focal_weight = (1 - target_probs) ** self.gamma
        
        # Calculate cross entropy with class weights
        if self.alpha is not None:
            alpha_weight = self.alpha[targets]
            focal_weight = focal_weight * alpha_weight
        
        loss = -focal_weight * log_probs[torch.arange(probs.shape[0]), targets]
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def calculate_f1_score(outputs, targets):
    """Calculate F1 score for batch"""
    with torch.no_grad():
        pred_classes = torch.argmax(outputs, dim=1).cpu().numpy()
        target_classes = targets.cpu().numpy()
        
        # Calculate macro F1 score (average of per-class F1 scores)
        f1_macro = f1_score(target_classes, pred_classes, average='macro', zero_division=0)
        # Calculate weighted F1 score (weighted by class frequency)
        f1_weighted = f1_score(target_classes, pred_classes, average='weighted', zero_division=0)
        
        return f1_macro, f1_weighted

def calculate_class_weights(dataset):
    """Calculate class weights based on inverse frequency"""
    class_counts = torch.zeros(4)
    for label in dataset.labels:
        class_counts[label] += 1
    
    # Prevent division by zero
    class_counts = torch.clamp(class_counts, min=1.0)
    
    # Calculate weights as inverse of frequency with smoothing
    weights = 1.0 / torch.log1p(class_counts)  # Using log1p for smoother weights
    weights = weights / weights.min()  # Normalize relative to minimum class weight
    
    print("Class distribution and weights:")
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    for i, (name, count, weight) in enumerate(zip(class_names, class_counts, weights)):
        print(f"  {name}: {int(count)} samples (weight: {weight:.3f})")
    
    return weights

def apply_smote_balancing(patches, labels, random_state=42):
    """Apply SMOTE to balance minority classes"""
    print("Applying SMOTE for class balancing...")
    
    # Flatten patches for SMOTE
    patches_flat = np.array(patches).reshape(len(patches), -1)
    
    # Apply SMOTE with conservative sampling strategy
    smote = SMOTE(
        sampling_strategy={
            1: min(30000, len(patches) // 4),  # minor-damage
            2: min(30000, len(patches) // 4),  # major-damage  
            3: min(30000, len(patches) // 4)   # destroyed
        },
        random_state=random_state,
        k_neighbors=3
    )
    
    try:
        patches_resampled, labels_resampled = smote.fit_resample(patches_flat, labels)
        
        # Reshape back to original patch dimensions
        original_shape = patches[0].shape
        patches_resampled = patches_resampled.reshape(-1, *original_shape)
        
        print(f"SMOTE applied: {len(patches)} -> {len(patches_resampled)} samples")
        print("New class distribution:")
        counter = Counter(labels_resampled)
        class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
        for i, name in enumerate(class_names):
            print(f"  {name}: {counter[i]}")
        
        return patches_resampled.tolist(), labels_resampled.tolist()
    except Exception as e:
        print(f"SMOTE failed: {e}. Using original data.")
        return patches, labels

def run_optimized_training():
    """Run training with optimal hyperparameters"""
    print(f'OPTIMIZED DAMAGE CLASSIFICATION TRAINING')
    print('=' * 60)
    
    # Optimal hyperparameters with anti-overfitting measures
    batch_size = 32  # Keep moderate for stability
    learning_rate = 1e-3  # Reduced to prevent overfitting
    epochs = 40  # Full training with early stopping
    early_stop_patience = 3  # More aggressive early stopping
    patch_size = 64  # Keep at 64 to reduce memory usage
    
    # Regularization parameters
    weight_decay = 5e-4  # Increased L2 regularization
    dropout_rate = 0.4  # Increased dropout
    
    print(f'Batch size: {batch_size}')
    print(f'Learning rate: {learning_rate:.0e}')
    print(f'Epochs: {epochs}')
    print(f'Early stopping patience: {early_stop_patience}')
    print(f'Patch size: {patch_size}x{patch_size}')
    print(f'Weight decay: {weight_decay}')
    print(f'Dropout rate: {dropout_rate}')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    torch.cuda.empty_cache()
    
    # Load data with increased patch size
    data_dir = os.path.join(project_root, 'Data')
    train_loader, val_loader = get_damage_data_loaders(
        data_dir, 
        batch_size=batch_size, 
        patch_size=patch_size,
        num_workers=2,  # Reduced to prevent memory issues
        prefetch_factor=1  # Reduced to prevent memory issues
    )
    
    if train_loader is None:
        print('Failed to create data loaders!')
        return
    
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Validation samples: {len(val_loader.dataset)}')
    print(f'Training batches: {len(train_loader)}')
    print(f'Validation batches: {len(val_loader)}')
    
    # Calculate class weights for weighted loss
    print('\nCalculating class weights...')
    class_weights = calculate_class_weights(train_loader.dataset)
    class_weights = class_weights.to(device)
    
    # Create model with increased dropout
    model = create_damage_model(dropout_rate=dropout_rate).to(device)
    
    # Use Focal Loss for better class imbalance handling
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    
    # Optimizer with strong regularization
    optimizer = optim.AdamW(  # AdamW for better weight decay
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Multi-step learning rate scheduler for better convergence
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[10, 20, 30], 
        gamma=0.5
    )
    
    best_f1_weighted = 0.0
    best_f1_macro = 0.0
    epochs_without_improvement = 0
    
    # Track both metrics for better stopping criteria
    best_combined_score = 0.0  # Combination of macro and weighted F1
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_f1_macro = 0.0
        train_f1_weighted = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc='Training')
        for batch_idx, (patches, labels) in enumerate(progress_bar):
            patches = patches.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(patches)
            loss = criterion(outputs, labels)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Gradient clipping (less aggressive)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            if torch.isnan(grad_norm) or grad_norm > 50.0:
                print(f"Warning: Large gradient norm {grad_norm:.2f} at batch {batch_idx}")
                continue
                
            optimizer.step()
            
            train_loss += loss.item()
            f1_macro, f1_weighted = calculate_f1_score(outputs, labels)
            train_f1_macro += f1_macro
            train_f1_weighted += f1_weighted
            num_batches += 1
            
            # Update progress bar
            if batch_idx % 50 == 0:
                current_loss = train_loss / num_batches
                current_f1_macro = train_f1_macro / num_batches
                current_f1_weighted = train_f1_weighted / num_batches
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'F1-Macro': f'{current_f1_macro:.3f}',
                    'F1-Weighted': f'{current_f1_weighted:.3f}',
                    'GradNorm': f'{grad_norm:.2f}'
                })
        
        if num_batches == 0:
            print(f"ERROR: No successful batches in epoch {epoch+1}")
            break
            
        avg_train_loss = train_loss / num_batches
        avg_train_f1_macro = train_f1_macro / num_batches
        avg_train_f1_weighted = train_f1_weighted / num_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_f1_macro = 0.0
        val_f1_weighted = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for patches, labels in tqdm(val_loader, desc='Validation'):
                patches = patches.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(patches)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                f1_macro, f1_weighted = calculate_f1_score(outputs, labels)
                val_f1_macro += f1_macro
                val_f1_weighted += f1_weighted
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_f1_macro = val_f1_macro / val_batches
        avg_val_f1_weighted = val_f1_weighted / val_batches
        
        print(f'Train Loss: {avg_train_loss:.4f}, Train F1-Macro: {avg_train_f1_macro:.3f}, Train F1-Weighted: {avg_train_f1_weighted:.3f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val F1-Macro: {avg_val_f1_macro:.3f}, Val F1-Weighted: {avg_val_f1_weighted:.3f}')
        
        # Combined metric: balance both macro and weighted F1
        combined_score = 0.6 * avg_val_f1_weighted + 0.4 * avg_val_f1_macro
        print(f'Combined Score: {combined_score:.3f} (0.6*weighted + 0.4*macro)')
        
        # Save model checkpoint using combined metric
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            best_f1_weighted = avg_val_f1_weighted
            best_f1_macro = avg_val_f1_macro
            epochs_without_improvement = 0  # Reset counter
            
            # Ensure the models/weights directory exists
            weights_dir = os.path.join(project_root, 'models', 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            
            model_save_path = os.path.join(weights_dir, 'best_damage_model_optimized.pth')
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1_weighted': best_f1_weighted,
                'best_f1_macro': best_f1_macro,
                'best_combined_score': best_combined_score,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'patch_size': patch_size,
                'class_weights': class_weights.cpu() if class_weights.is_cuda else class_weights,
                'weight_decay': weight_decay
            }, model_save_path)
            
            print(f'✓ Best model saved! Combined: {best_combined_score:.3f}, F1-W: {best_f1_weighted:.3f}, F1-M: {best_f1_macro:.3f} (Epoch {epoch+1})')
        else:
            epochs_without_improvement += 1
            print(f'No improvement for {epochs_without_improvement}/{early_stop_patience} epochs')
            
            # Early stopping check
            if epochs_without_improvement >= early_stop_patience:
                print(f'\n🛑 Early stopping triggered! No improvement for {early_stop_patience} epochs.')
                print(f'Best Combined Score: {best_combined_score:.3f}')
                print(f'Best F1-Weighted: {best_f1_weighted:.3f}, Best F1-Macro: {best_f1_macro:.3f}')
                break
        
        # Step scheduler
        scheduler.step()
    
    print(f'\nTraining completed!')
    print(f'Best Combined Score: {best_combined_score:.3f}')
    print(f'Best F1-Weighted: {best_f1_weighted:.3f} ({best_f1_weighted*100:.1f}%)')
    print(f'Best F1-Macro: {best_f1_macro:.3f} ({best_f1_macro*100:.1f}%)')

if __name__ == '__main__':
    run_optimized_training() 