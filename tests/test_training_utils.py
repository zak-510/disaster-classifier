import pytest
import torch
import numpy as np
from src.utils.training_utils import (
    calculate_iou,
    calculate_dice_score,
    EarlyStopping,
    LRScheduler,
    save_checkpoint,
    load_checkpoint
)

def test_calculate_iou():
    pred = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 1]]).float()
    target = torch.tensor([[0, 1, 1], [1, 1, 0], [0, 1, 1]]).float()
    iou = calculate_iou(pred, target)
    assert isinstance(iou, float)
    assert 0 <= iou <= 1

def test_calculate_dice_score():
    pred = torch.tensor([[0, 1, 1], [1, 1, 1], [0, 1, 1]]).float()
    target = torch.tensor([[0, 1, 1], [1, 1, 0], [0, 1, 1]]).float()
    dice = calculate_dice_score(pred, target)
    assert isinstance(dice, float)
    assert 0 <= dice <= 1

def test_early_stopping():
    early_stopping = EarlyStopping(patience=2, min_delta=0.01)
    
    # Should not stop
    assert not early_stopping(val_loss=1.0)
    assert not early_stopping(val_loss=0.9)
    
    # Should stop (no improvement > min_delta for patience steps)
    assert not early_stopping(val_loss=0.89)
    assert early_stopping(val_loss=0.89)

def test_lr_scheduler():
    scheduler = LRScheduler(
        mode='min',
        factor=0.1,
        patience=2,
        min_lr=1e-6
    )
    
    current_lr = 0.001
    # No reduction yet
    current_lr = scheduler(val_loss=1.0, current_lr=current_lr)
    assert current_lr == 0.001
    
    # Should reduce after patience steps without improvement
    current_lr = scheduler(val_loss=1.0, current_lr=current_lr)
    current_lr = scheduler(val_loss=1.0, current_lr=current_lr)
    assert current_lr == 0.0001

def test_checkpoint_save_load(tmp_path, sample_config):
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    
    checkpoint_path = tmp_path / "checkpoint.pt"
    
    # Save checkpoint
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=10,
        loss=0.5,
        config=sample_config
    )
    
    # Load checkpoint
    loaded = load_checkpoint(checkpoint_path)
    assert loaded["epoch"] == 10
    assert loaded["loss"] == 0.5
    assert loaded["config"] == sample_config

def test_invalid_checkpoint_path():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("nonexistent.pt")

@pytest.mark.parametrize("mode,better_loss", [
    ('min', 0.8),  # Lower loss is better
    ('max', 1.2)   # Higher score is better
])
def test_lr_scheduler_modes(mode, better_loss):
    scheduler = LRScheduler(
        mode=mode,
        factor=0.1,
        patience=1,
        min_lr=1e-6
    )
    
    current_lr = 0.001
    scheduler(val_loss=1.0, current_lr=current_lr)
    new_lr = scheduler(val_loss=better_loss, current_lr=current_lr)
    
    # LR should not change when metric improves
    assert new_lr == current_lr 