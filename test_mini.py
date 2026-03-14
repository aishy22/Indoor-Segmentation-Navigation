# test_mini.py
import torch
from src.config import Config
from src.data_prep import get_dataloaders
from src.model import IndoorSegmentationModel

print("Testing with minimal settings...")
print(f"Batch size: {Config.BATCH_SIZE}")
print(f"Train samples: {Config.MAX_TRAIN_SAMPLES}")
print(f"Val samples: {Config.MAX_VAL_SAMPLES}")

# Get dataloaders
train_loader, val_loader = get_dataloaders(
    batch_size=Config.BATCH_SIZE,
    num_workers=0
)

# Create model
model = IndoorSegmentationModel()
print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test one batch
print("\nTesting one training batch...")
for batch_idx, (images, masks) in enumerate(train_loader):
    print(f"Batch {batch_idx + 1}:")
    print(f"  Images shape: {images.shape}")
    print(f"  Masks shape: {masks.shape}")
    print(f"  Masks unique values: {torch.unique(masks)}")
    
    # Forward pass
    outputs = model(images)
    print(f"  Output shape: {outputs.shape}")
    
    if batch_idx >= 0:  # Just test first batch
        break

print("\n✓ Test complete!")
