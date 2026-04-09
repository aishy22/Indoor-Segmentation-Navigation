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
        IMAGE_SIZE = 256


class IndoorSegmentationModel(nn.Module):
    """
    U-Net with ResNet50 backbone for indoor scene segmentation.
    Outputs 4 classes: floor, obstacle/wall, door, no-go.
    """

    def __init__(self, num_classes=None, encoder_name=None, encoder_weights=None):
        super(IndoorSegmentationModel, self).__init__()

        self.num_classes    = num_classes    or getattr(Config, 'NUM_CLASSES', 4)
        self.encoder_name   = encoder_name   or getattr(Config, 'ENCODER', 'resnet50')
        encoder_weights     = encoder_weights or getattr(Config, 'ENCODER_WEIGHTS', 'imagenet')

        # U-Net with pre-trained ResNet50 encoder
        self.model = smp.Unet(
            encoder_name=self.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=self.num_classes,
            activation=None,             # Raw logits — required for CrossEntropyLoss
            decoder_use_batchnorm=True,
            decoder_dropout=0.3,         # ← consistent with train.py
        )

        print(f"✓ Created U-Net model with {self.encoder_name} encoder")
        print(f"  Input           : RGB images (3 channels)")
        print(f"  Output          : {self.num_classes} classes → {getattr(Config, 'CLASS_NAMES', '')}")
        print(f"  Encoder weights : {encoder_weights}")
        print(f"  Decoder dropout : 0.3")

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        """Return predicted class index per pixel (not logits)."""
        with torch.no_grad():
            logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class IndoorSegmentationLoss(nn.Module):
    """
    Combined loss: CrossEntropy + Dice.
    CE handles per-pixel classification; Dice handles class imbalance.

    Returns: (total_loss, ce_loss, dice_loss)
    """

    def __init__(self, use_dice_loss=True, dice_weight=0.3, class_weights=None):
        super(IndoorSegmentationLoss, self).__init__()

        self.use_dice_loss = use_dice_loss
        self.dice_weight   = dice_weight

        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights if class_weights is not None else None
        )

        print(f"✓ Loss function: CrossEntropy"
              f"{' + Dice (weight=' + str(dice_weight) + ')' if use_dice_loss else ''}")
        if class_weights is not None:
            print(f"  Class weights: {class_weights.tolist()}")

    def forward(self, predictions, targets):
        ce   = self.ce_loss(predictions, targets)

        if self.use_dice_loss:
            dice  = self._dice_loss(predictions, targets)
            total = (1 - self.dice_weight) * ce + self.dice_weight * dice
            return total, ce, dice
        else:
            return ce, ce, torch.tensor(0.0, device=predictions.device)

    def _dice_loss(self, predictions, targets, smooth=1e-6):
        """
        Soft Dice loss averaged across all classes.
        Uses one-hot encoding of targets for per-class computation.
        """
        num_classes = predictions.shape[1]
        probs = torch.softmax(predictions, dim=1)

        # One-hot encode targets: (B, H, W) → (B, C, H, W)
        targets_one_hot = torch.eye(num_classes, device=targets.device)[targets]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dice_scores = []
        for c in range(num_classes):
            prob   = probs[:, c, :, :]
            target = targets_one_hot[:, c, :, :]

            intersection = (prob * target).sum(dim=(1, 2))
            union        = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
            dice         = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)

        mean_dice = torch.stack(dice_scores).mean()
        return 1.0 - mean_dice


def create_model(pretrained=True):
    """
    Convenience function to instantiate and summarize the model.

    Args:
        pretrained: If True, uses ImageNet weights for encoder.
                    If False, trains encoder from scratch.
    Returns:
        model: IndoorSegmentationModel instance
    """
    encoder_weights = 'imagenet' if pretrained else None
    model = IndoorSegmentationModel(encoder_weights=encoder_weights)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Statistics:")
    print(f"  Total parameters     : {total_params:,}")
    print(f"  Trainable parameters : {trainable_params:,}")

    return model


def create_loss(class_weights=None):
    """
    Convenience function to instantiate the combined loss.

    Args:
        class_weights: Optional tensor of per-class weights for CrossEntropy.
                       Recommended: torch.tensor([0.8, 0.5, 1.5, 2.0])
    Returns:
        criterion: IndoorSegmentationLoss instance
    """
    return IndoorSegmentationLoss(
        use_dice_loss=True,
        dice_weight=0.3,
        class_weights=class_weights
    )


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Model")
    print("=" * 50)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    size   = getattr(Config, 'IMAGE_SIZE', 256)

    # Test model
    model = create_model(pretrained=True).to(device)

    x = torch.randn(2, 3, size, size).to(device)
    with torch.no_grad():
        y = model(x)

    print(f"\nForward pass:")
    print(f"  Input shape  : {x.shape}")
    print(f"  Output shape : {y.shape}")   # Expected: (2, 4, 256, 256)

    # Test prediction
    preds = model.predict(x.to(device))
    print(f"  Predict shape: {preds.shape}")  # Expected: (2, 256, 256)

    # Test loss
    class_weights = torch.tensor([0.8, 0.5, 1.5, 2.0]).to(device)
    criterion = create_loss(class_weights=class_weights)

    dummy_targets = torch.randint(0, 4, (2, size, size)).to(device)
    total, ce, dice = criterion(y, dummy_targets)

    print(f"\nLoss test:")
    print(f"  Total loss : {total.item():.4f}")
    print(f"  CE loss    : {ce.item():.4f}")
    print(f"  Dice loss  : {dice.item():.4f}")

    print("\n✓ Model test complete!")
