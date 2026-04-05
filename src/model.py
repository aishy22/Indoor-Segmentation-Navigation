# src/model.py
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import Config

class IndoorSegmentationModel(nn.Module):
    """
    U-Net with ResNet50 backbone for indoor scene segmentation
    Outputs 4 classes: floor, obstacle/wall, door, no-go
    """
    
    def __init__(self, num_classes=Config.NUM_CLASSES, encoder_name=Config.ENCODER):
        super(IndoorSegmentationModel, self).__init__()
        
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        
        # Create U-Net model with pre-trained ResNet50 encoder
        self.model = smp.Unet(
            encoder_name=encoder_name,           # ResNet50 backbone
            encoder_weights=Config.ENCODER_WEIGHTS,  # ImageNet pre-trained weights
            in_channels=3,                        # RGB images
            classes=num_classes,                   # 4 output classes
            activation=None,                       # No activation (will use CrossEntropyLoss)
            decoder_use_batchnorm=True,            # Use batch norm in decoder
            decoder_attention_type=None,           # No attention (simpler model)
        )
        
        print(f"✓ Created U-Net model with {encoder_name} encoder")
        print(f"  Input: RGB images (3 channels)")
        print(f"  Output: {num_classes} classes ({Config.CLASS_NAMES})")
        print(f"  Pre-trained weights: {Config.ENCODER_WEIGHTS}")
        
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, 3, H, W)
        Returns:
            Output tensor of shape (batch_size, num_classes, H, W)
        """
        return self.model(x)
    
    def get_features(self, x):
        """
        Get encoder features (useful for visualization or analysis)
        Args:
            x: Input tensor
        Returns:
            Dictionary of encoder features
        """
        return self.model.encoder(x)
    
    def predict(self, x):
        """
        Get prediction classes (not logits)
        Args:
            x: Input tensor
        Returns:
            Class predictions of shape (batch_size, H, W)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

class IndoorSegmentationLoss(nn.Module):
    """
    Combined loss function for segmentation
    Combines CrossEntropyLoss with optional Dice loss
    """
    
    def __init__(self, use_dice_loss=True, dice_weight=0.3, class_weights=None):
        super(IndoorSegmentationLoss, self).__init__()
        
        self.use_dice_loss = use_dice_loss
        self.dice_weight = dice_weight
        
        # Cross-entropy loss (with optional class weights)
        if class_weights is not None:
            self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Model outputs (logits) of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W)
        """
        # Cross-entropy loss
        ce_loss = self.ce_loss(predictions, targets)
        
        if self.use_dice_loss:
            # Dice loss for better boundary segmentation
            dice_loss = self.dice_loss(predictions, targets)
            total_loss = (1 - self.dice_weight) * ce_loss + self.dice_weight * dice_loss
            return total_loss, ce_loss, dice_loss
        else:
            return ce_loss, ce_loss, torch.tensor(0.0)
    
    def dice_loss(self, predictions, targets, smooth=1e-6):
        """
        Compute Dice loss for multi-class segmentation
        """
        # Convert to probabilities
        probs = torch.softmax(predictions, dim=1)
        
        # Convert targets to one-hot
        targets_one_hot = torch.eye(predictions.shape[1], device=targets.device)[targets]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)
        
        # Compute Dice score for each class
        dice_scores = []
        for class_idx in range(predictions.shape[1]):
            prob = probs[:, class_idx, :, :]
            target = targets_one_hot[:, class_idx, :, :]
            
            intersection = (prob * target).sum(dim=(1, 2))
            union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
            
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)
        
        # Average Dice score across classes and batch
        mean_dice = torch.stack(dice_scores).mean()
        
        # Dice loss = 1 - Dice score
        return 1.0 - mean_dice

def create_model(pretrained=True):
    """
    Factory function to create the model
    """
    model = IndoorSegmentationModel()
    
    if pretrained:
        print("Using pre-trained ImageNet weights")
    else:
        print("Training from scratch (no pre-trained weights)")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    
    return model

def test_model():
    """
    Quick test to verify model works
    """
    print("=" * 50)
    print("Testing IndoorSegmentationModel")
    print("=" * 50)
    
    # Create model
    model = create_model(pretrained=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 256, 256).to(device)
    
    print(f"\nInput shape: {input_tensor.shape}")
    
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Output shape: {output.shape}")
    print(f"Output min: {output.min():.3f}, max: {output.max():.3f}")
    
    # Test prediction
    pred_classes = model.predict(input_tensor)
    print(f"Prediction shape: {pred_classes.shape}")
    print(f"Prediction unique values: {torch.unique(pred_classes)}")
    
    # Test loss function
    print("\n" + "=" * 50)
    print("Testing Loss Function")
    print("=" * 50)
    
    criterion = IndoorSegmentationLoss(use_dice_loss=True)
    targets = torch.randint(0, 4, (batch_size, 256, 256)).to(device)
    
    total_loss, ce_loss, dice_loss = criterion(output, targets)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"CE loss: {ce_loss.item():.4f}")
    print(f"Dice loss: {dice_loss.item():.4f}")
    
    print("\n✓ Model test complete!")
    
    return model

# Run test if script is executed directly
if __name__ == "__main__":
    model = test_model()
    
    # Print model architecture summary
    print("\n" + "=" * 50)
    print("Model Architecture Summary")
    print("=" * 50)
    print(model)
