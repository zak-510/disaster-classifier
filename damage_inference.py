import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, jaccard_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from damage_model import create_damage_model, calculate_accuracy
from damage_data import get_damage_data_loaders


def run_damage_inference():
    device = torch.device('cuda')
    print(f'DAMAGE CLASSIFIER INFERENCE')
    print('=' * 60)
    
    # Load the trained model
    model_path = './weights/best_damage_model_optimized.pth'
    if not os.path.exists(model_path):
        print(f'ERROR: Model file {model_path} not found!')
        return
    
    model = create_damage_model().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f'SUCCESS: Model loaded successfully!')
    print(f'Model accuracy: {checkpoint["best_acc"]:.3f}')
    print(f'Epochs trained: {checkpoint["epoch"]}')
    
    # Create test data loader
    data_dir = './Data'
    batch_size = 64
    
    try:
        _, test_loader = get_damage_data_loaders(
            data_dir, 
            batch_size=batch_size, 
            patch_size=64
        )
    except Exception as e:
        print('ERROR: Failed to create test data loader!')
        print(f'Error: {e}')
        return
    
    if test_loader is None:
        print('ERROR: Test loader is None')
        return
    
    print(f'Test samples: {len(test_loader.dataset)}')
    
    # Class names
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    num_classes = len(class_names)
    
    # Inference
    model.eval()
    all_predictions = []
    all_targets = []
    all_probs = []
    
    print(f'\nPROCESSING: Running inference on test set...')
    
    with torch.no_grad():
        for batch_idx, (patches, labels) in enumerate(tqdm(test_loader, desc='Inference')):
            patches = patches.to(device)
            labels = labels.to(device)
            
            outputs = model(patches)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    
    # Per-class metrics
    per_class_precision = precision_score(targets, predictions, average=None)
    per_class_recall = recall_score(targets, predictions, average=None)
    per_class_f1 = f1_score(targets, predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Calculate IoU for each class
    intersection = np.diag(cm)
    union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
    iou_per_class = intersection / (union + 1e-8)
    mean_iou = np.mean(iou_per_class)
    
    print(f'\nRESULTS: INFERENCE RESULTS:')
    print('=' * 60)
    print(f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)')
    print(f'Weighted Precision: {precision:.4f}')
    print(f'Weighted Recall: {recall:.4f}')
    print(f'Weighted F1-Score: {f1:.4f}')
    print(f'Mean IoU: {mean_iou:.4f} ({mean_iou*100:.1f}%)')
    
    print(f'\nPer-Class Performance:')
    for i, class_name in enumerate(class_names):
        print(f'  {class_name}:')
        print(f'    Precision: {per_class_precision[i]:.4f}')
        print(f'    Recall: {per_class_recall[i]:.4f}')
        print(f'    F1-Score: {per_class_f1[i]:.4f}')
        print(f'    IoU: {iou_per_class[i]:.4f}')
    
    print(f'\nDETAILED CLASSIFICATION REPORT:')
    print('=' * 60)
    print(classification_report(targets, predictions, target_names=class_names, digits=4))
    
    # Print confusion matrix
    print(f'\nCONFUSION MATRIX:')
    print('=' * 60)
    print('Predicted →')
    header = ''.join([f'{name:>12}' for name in class_names])
    print(f'{"Actual ↓":>12}{header}')
    for i, class_name in enumerate(class_names):
        row = ''.join([f'{cm[i,j]:>12}' for j in range(len(class_names))])
        print(f'{class_name:>12}{row}')
    
    # Performance assessment
    print(f'\nPERFORMANCE ASSESSMENT:')
    print('=' * 60)
    
    target_acc = 0.70  # 70% accuracy target
    target_iou = 0.50  # 50% IoU target
    
    if accuracy >= target_acc:
        print(f'SUCCESS: ACCURACY TARGET MET: {accuracy*100:.1f}% >= {target_acc*100:.0f}%')
    else:
        print(f'ERROR: Accuracy below target: {accuracy*100:.1f}% < {target_acc*100:.0f}%')
    
    if mean_iou >= target_iou:
        print(f'SUCCESS: IoU TARGET MET: {mean_iou*100:.1f}% >= {target_iou*100:.0f}%')
    else:
        print(f'ERROR: IoU below target: {mean_iou*100:.1f}% < {target_iou*100:.0f}%')
    
    if accuracy >= target_acc and mean_iou >= target_iou:
        print(f'SUCCESS: Model ready for production use')
    else:
        print(f'\nGOOD PROGRESS: Model shows strong performance')
        print(f'RECOMMENDATION: Consider additional training or fine-tuning for target achievement')
    
    # Save detailed results
    results_dir = './damage_inference_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Save summary
    summary_path = os.path.join(results_dir, 'inference_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('DAMAGE CLASSIFIER INFERENCE RESULTS\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Model: {model_path}\n')
        f.write(f'Test samples: {len(all_targets)}\n')
        f.write(f'Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)\n')
        f.write(f'Mean IoU: {mean_iou:.4f} ({mean_iou*100:.1f}%)\n\n')
        f.write('Per-Class IoU:\n')
        for name, iou in zip(class_names, iou_per_class):
            f.write(f'  {name}: {iou:.4f} ({iou*100:.1f}%)\n')
        f.write('\n' + classification_report(targets, predictions, target_names=class_names, digits=4))
    
    print(f'\n📁 Results saved to: {results_dir}/')
    print(f'   - inference_summary.txt: Detailed metrics')
    
    print(f'\nSUCCESS: READY FOR PIPELINE INTEGRATION!')
    print('=' * 60)
    
    print(f'\nSUMMARY: Model performance metrics collected for analysis.')
    
    return {
        'accuracy': accuracy,
        'iou': mean_iou,
        'class_ious': iou_per_class,
        'confusion_matrix': cm,
        'predictions': predictions,
        'targets': targets
    }

if __name__ == '__main__':
    results = run_damage_inference()
    
    if results and results['accuracy'] >= 0.70:
        print(f'\nREADY FOR PIPELINE INTEGRATION!')
        print(f'The trained model can now be used for damage assessment in the full pipeline.')
    else:
        print(f'\nModel performance metrics collected for analysis.') 