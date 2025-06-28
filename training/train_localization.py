import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import sys

# Add project root to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from models.model import create_model, calculate_iou
from data_processing.localization_data import get_proper_data_loaders

def train_model():
    device = torch.device('cuda')
    print(f'Using device: {device}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB')
    
    torch.cuda.empty_cache()
    
    data_dir = os.path.join(project_root, 'Data')
    if not os.path.exists(data_dir):
        print(f'Data directory {data_dir} not found!')
        return
    
    model = create_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    batch_size = 2
    print(f'Batch size: {batch_size}')
    
    train_loader, val_loader = get_proper_data_loaders(data_dir, batch_size=batch_size)
    
    if train_loader is None:
        print('Failed to create proper data loaders!')
        return
    
    print(f'Training samples: {len(train_loader.dataset)}')
    
    epochs = 5
    best_iou = 0.0
    
    for epoch in range(epochs):
        print(f'\nEpoch {epoch+1}/{epochs}')
        
        model.train()
        train_loss = 0.0
        train_iou = 0.0
        
        for batch_idx, (images, masks) in enumerate(tqdm(train_loader)):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            if torch.isnan(loss):
                print(f'NaN loss at batch {batch_idx}')
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            batch_iou = calculate_iou(outputs, masks)
            train_iou += batch_iou
            
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_iou = train_iou / len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                
                val_loss += loss.item()
                val_iou += calculate_iou(outputs, masks)
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)
        
        print(f'Train Loss: {avg_train_loss:.4f} | Train IoU: {avg_train_iou:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f} | Val IoU: {avg_val_iou:.4f}')
        
        if avg_val_iou > best_iou:
            best_iou = avg_val_iou
            weights_dir = os.path.join(project_root, 'weights')
            os.makedirs(weights_dir, exist_ok=True)
            model_path = os.path.join(weights_dir, 'best_model_proper.pth')
            torch.save(model.state_dict(), model_path)
            print(f'Best IoU: {best_iou:.4f} - Model saved to {model_path}!')
        
        max_mem = torch.cuda.max_memory_allocated(0) // 1024**2
        print(f'Max GPU Memory: {max_mem}MB')
        torch.cuda.empty_cache()
    
    print(f'\nTraining complete! Best IoU: {best_iou:.4f}')

if __name__ == '__main__':
    train_model() 