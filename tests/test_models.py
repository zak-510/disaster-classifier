import pytest
import torch
from src.models.localization import UNet
from src.models.damage_classifier import DamageClassifier

def test_unet_forward(sample_config):
    model = UNet(
        encoder=sample_config["model"]["encoder"],
        num_classes=sample_config["model"]["num_classes"]
    )
    x = torch.randn(2, 3, 512, 512)
    y = model(x)
    assert y.shape == (2, 1, 512, 512)

def test_unet_loss():
    model = UNet(encoder="resnet34", num_classes=1)
    criterion = model.get_loss_fn()
    
    pred = torch.sigmoid(torch.randn(2, 1, 512, 512))
    target = torch.randint(0, 2, (2, 1, 512, 512)).float()
    
    loss = criterion(pred, target)
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad

def test_damage_classifier_forward():
    model = DamageClassifier(num_classes=4)  # No damage, minor, major, destroyed
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.shape == (2, 4)

def test_model_save_load(tmp_path, sample_config):
    # Test UNet
    unet = UNet(
        encoder=sample_config["model"]["encoder"],
        num_classes=sample_config["model"]["num_classes"]
    )
    save_path = tmp_path / "unet.pt"
    torch.save(unet.state_dict(), save_path)
    
    loaded_unet = UNet(
        encoder=sample_config["model"]["encoder"],
        num_classes=sample_config["model"]["num_classes"]
    )
    loaded_unet.load_state_dict(torch.load(save_path))
    
    # Test classifier
    classifier = DamageClassifier(num_classes=4)
    save_path = tmp_path / "classifier.pt"
    torch.save(classifier.state_dict(), save_path)
    
    loaded_classifier = DamageClassifier(num_classes=4)
    loaded_classifier.load_state_dict(torch.load(save_path))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_model_cuda():
    model = UNet(encoder="resnet34", num_classes=1)
    model.cuda()
    x = torch.randn(2, 3, 512, 512).cuda()
    y = model(x)
    assert y.is_cuda 