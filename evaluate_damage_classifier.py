import torch
import os
import json
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from data_processing.damage_data import DamageDataset
from models.damage_model import create_damage_model

def evaluate_damage_classifier():
    print("DAMAGE CLASSIFIER EVALUATION (Ground Truth Buildings)")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    model = create_damage_model().to(device)
    checkpoint_path = os.path.join('weights', 'best_damage.pth')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nLoaded model checkpoint with validation metrics:")
    print(f"F1-Weighted: {checkpoint['best_f1_weighted']:.3f}")
    print(f"F1-Macro: {checkpoint['best_f1_macro']:.3f}")
    
    # Create test dataset using ground truth crops
    test_dataset = DamageDataset('Data', split='test', augment=False, patch_size=64)
    print(f"\nTest set size: {len(test_dataset)} building patches")
    
    # Lists to store results
    results = []
    all_preds = []
    all_labels = []
    
    # Evaluate
    print("\nEvaluating model...")
    with torch.no_grad():
        for idx in tqdm(range(len(test_dataset))):
            patch, label = test_dataset[idx]
            patch = patch.unsqueeze(0).to(device)  # Add batch dimension
            
            # Get prediction and confidence
            outputs = model(patch)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(outputs, dim=1).item()
            confidence = probs[0][pred].item()
            
            # Store results
            all_preds.append(pred)
            all_labels.append(label.item())
            
            results.append({
                'image_idx': idx,
                'true_label': label.item(),
                'predicted_label': pred,
                'confidence': confidence
            })
    
    # Calculate metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print("\nResults on Ground Truth Building Patches:")
    print(f"F1-Macro: {f1_macro:.3f}")
    print(f"F1-Weighted: {f1_weighted:.3f}")
    
    # Print confusion matrix
    class_names = ['no-damage', 'minor-damage', 'major-damage', 'destroyed']
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print("Predicted →")
    print("Actual ↓")
    print(" " * 12 + " ".join(f"{name:>12}" for name in class_names))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:12} " + " ".join(f"{count:12d}" for count in row))
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df['true_label_name'] = results_df['true_label'].map(
        {i: name for i, name in enumerate(class_names)}
    )
    results_df['predicted_label_name'] = results_df['predicted_label'].map(
        {i: name for i, name in enumerate(class_names)}
    )
    
    output_path = 'test_results/damage_classifier_evaluation.csv'
    os.makedirs('test_results', exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == '__main__':
    evaluate_damage_classifier() 