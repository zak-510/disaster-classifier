"""
Damage classification model based on ResNet50.
Provides functionality for damage assessment in disaster response imagery.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

class DamageClassifier(nn.Module):
    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predict(self, x: torch.Tensor) -> Dict[str, Any]:
        with torch.no_grad():
            logits = self(x)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]
            
        return {
            'class': pred_class.item(),
            'confidence': confidence.item(),
            'probabilities': probs[0].tolist()
        }
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: Optional[str] = None) -> 'DamageClassifier':
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        model = cls()
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        return model 