import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from model import create_model
from localization_data import XBDDataset
import os

def run_inference():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = create_model().to(device)
    
    checkpoint_path = 'checkpoints/extended/model_epoch_20.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f'Loaded localization model from {checkpoint_path}')
    else:
        print('No trained model found!')
        return
    
    model.eval()
    
    data_dir = './Data'
    test_dataset = XBDDataset(data_dir, 'test')
    
    if len(test_dataset) == 0:
        print('No test samples found!')
        return
    
    os.makedirs('outputs', exist_ok=True)
    
    num_samples = min(10, len(test_dataset))
    
    with torch.no_grad():
        for i in range(num_samples):
            image, mask = test_dataset[i]
            
            image_batch = image.unsqueeze(0).to(device)
            
            prediction = model(image_batch)
            prediction = torch.sigmoid(prediction)
            
            image_np = image.permute(1, 2, 0).cpu().numpy()
            mask_np = mask.squeeze().cpu().numpy()
            pred_np = prediction.squeeze().cpu().numpy()
            
            pred_binary = (pred_np > 0.5).astype(np.uint8)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            axes[0].imshow(image_np)
            axes[0].set_title('Satellite Image')
            axes[0].axis('off')
            
            axes[1].imshow(mask_np, cmap='gray')
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
            
            axes[2].imshow(pred_binary, cmap='gray')
            axes[2].set_title('Prediction')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(f'outputs/inference_{i:03d}.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f'Saved inference_{i:03d}.png')
    
    print(f'Inference complete! Generated {num_samples} visualization images in outputs/')

if __name__ == '__main__':
    run_inference() 