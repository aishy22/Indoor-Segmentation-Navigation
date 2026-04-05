# src/model.py
# src/model.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
import sys

sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

try:
    from config import Config
except ImportError:
    class Config:
        NUM_CLASSES = 4
        CLASS_NAMES = ['floor', 'obstacle/wall', 'door', 'no-go']
        ENCODER = 'resnet50'
        ENCODER_WEIGHTS = 'imagenet'

class IndoorSegmentationModel(nn.Module):
    """
    U-Net with ResNet50 backbone for indoor scene segmentation
    Outputs 4 classes: floor, obstacle/wall, door, no-go
    """
    
    def __init__(self, num_classes=None, encoder_name=None):
        super(IndoorSegmentationModel, self).__init__()
        
        self.num_classes = num_classes or getattr(Config, 'NUM_CLASSES', 4)
        self.encoder_name = encoder_name or getattr(Config, 'ENCODER', 'resnet50')
        encoder_weights = getattr(Config, 'ENCODER_WEIGHTS', 'imagenet')
        
        # Create U-Net model with pre-trained ResNet50 encoder
        self.model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=self.num_classes,
            activation=None,
            decoder_use_batchnorm=True,
            decoder_dropout=0.3,  # Add dropout to prevent overfitting
        )
        
        print(f"✓ Created U-Net model with {self.encoder_name} encoder")
        print(f"  Input: RGB images (3 channels)")
        print(f"  Output: {self.num_classes} classes")
        print(f"  Pre-trained weights: {encoder_weights}")
        
    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        """Get prediction classes (not logits)"""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class IndoorSegmentationLoss(nn.Module):
    """Combined loss: CrossEntropy + Dice"""
    
    def __init__(self, use_dice_loss=True, dice_weight=0.3, class_weights=None):
        super(IndoorSegmentationLoss, self).__init__()
        
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        ce_loss = self.ce_loss(predictions, targets)
        
        if self.use_dice_loss:
            dice_loss = self.dice_loss(predictions, targets)
            total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
            return total_loss, ce_loss, dice_loss
        else:
            return ce_loss, ce_loss, torch.tensor(0.0)
    
    def dice_loss(self, predictions, targets, smooth=1e-6):
        probs = torch.softmax(predictions, dim=1)
        targets_one_hot = torch.eye(predictions.shape[1], device=targets.device)[targets]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        dice_scores = []
        for class_idx in range(predictions.shape[1]):
            prob = probs[:, class_idx, :, :]
            target = targets_one_hot[:, class_idx, :, :]
            
            intersection = (prob * target).sum(dim=(1, 2))
            union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


def create_model(pretrained=True):
    model = IndoorSegmentationModel()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    return model


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Model")
    print("=" * 50)
    
    model = create_model()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("\n✓ Model test complete!")
