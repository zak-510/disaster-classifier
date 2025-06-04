import torch
import numpy as np
import pytest
from src.utils.metrics import (
    calculate_localization_metrics,
    calculate_classification_metrics,
    calculate_damage_level_metrics,
    calculate_confusion_matrix
)

def test_localization_metrics():
    # Create sample data
    pred_masks = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]]],
        [[[0.0, 1.0], [1.0, 0.0]]]
    ])
    true_masks = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]]],
        [[[0.0, 1.0], [1.0, 0.0]]]
    ])
    
    # Calculate metrics
    metrics = calculate_localization_metrics(pred_masks, true_masks)
    
    # Check metrics
    assert metrics['iou'] == 1.0
    assert metrics['f1_score'] == 1.0

def test_classification_metrics():
    # Create sample data
    preds = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    targets = torch.tensor([0, 1, 2])
    
    # Calculate metrics
    metrics = calculate_classification_metrics(preds, targets)
    
    # Check metrics
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_score'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['recall'] == 1.0
    assert all(metrics[f'f1_class_{i}'] == 1.0 for i in range(3))

def test_damage_level_metrics():
    # Create sample data
    preds = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    targets = torch.tensor([0, 1, 2])
    damage_levels = ['no-damage', 'minor', 'major']
    
    # Calculate metrics
    metrics = calculate_damage_level_metrics(preds, targets, damage_levels)
    
    # Check metrics
    assert metrics['accuracy'] == 1.0
    assert metrics['f1_score'] == 1.0
    assert metrics['f1_no-damage'] == 1.0
    assert metrics['f1_minor'] == 1.0
    assert metrics['f1_major'] == 1.0

def test_confusion_matrix():
    # Create sample data
    preds = torch.tensor([
        [0.8, 0.1, 0.1],
        [0.1, 0.8, 0.1],
        [0.1, 0.1, 0.8]
    ])
    targets = torch.tensor([0, 1, 2])
    
    # Calculate confusion matrix
    matrix = calculate_confusion_matrix(preds, targets, num_classes=3)
    
    # Check matrix
    expected = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    assert np.array_equal(matrix, expected)

def test_localization_metrics_partial_match():
    # Create sample data with partial match
    pred_masks = torch.tensor([
        [[[1.0, 0.0], [0.0, 1.0]]],
        [[[0.0, 1.0], [1.0, 0.0]]]
    ])
    true_masks = torch.tensor([
        [[[1.0, 1.0], [0.0, 1.0]]],
        [[[0.0, 1.0], [1.0, 1.0]]]
    ])
    
    # Calculate metrics
    metrics = calculate_localization_metrics(pred_masks, true_masks)
    
    # Check metrics
    assert 0.0 < metrics['iou'] < 1.0
    assert 0.0 < metrics['f1_score'] < 1.0 