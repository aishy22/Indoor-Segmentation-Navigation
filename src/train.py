# src/train.py - Final Working Version
# Trains on full ADE20K dataset with pre-trained weights

import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

# ============================================
# CONFIGURATION
# ============================================
class Config:
    DATA_PATH = '/content/ADEChallengeData2016'
    MODEL_SAVE_PATH = '/content/Indoor-Segmentation-Navigation/models'
    OUTPUT_PATH = '/content/Indoor-Segmentation-Navigation/outputs'
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================
# CLASS MAPPING
# ============================================
def create_class_mapping():
    class_mapping = {}
    for i in range(1, 151):
        class_mapping[i] = 1  # Default: wall
    
    # Floor (class 0)
    floor_ids = [4, 12, 14, 29, 53, 95, 7, 30, 55]
    for idx in floor_ids:
        class_mapping[idx] = 0
    
    # Door (class 2)
    door_ids = [15, 59]
    for idx in door_ids:
        class_mapping[idx] = 2
    
    # No-go (class 3)
    nogo_ids = [54, 60, 97, 61, 17, 22, 27, 50, 69, 72, 105, 110, 114, 122]
    for idx in nogo_ids:
        class_mapping[idx] = 3
    
    return class_mapping

# ============================================
# DATASET CLASS
# ============================================
class ADE20KDataset(Dataset):
    def __init__(self, split='train', transform=None):
        self.split = split
        self.transform = transform
        self.class_mapping = create_class_mapping()
        
        base_path = Config.DATA_PATH
        self.images_dir = f'{base_path}/images/{split}'
        self.masks_dir = f'{base_path}/annotations/{split}'
        
        self.images = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        print(f"Loaded {len(self.images)} {split} images")
    
    def __len__(self):
        return len(self.images)
    
    def apply_mapping(self, mask):
        result = np.ones_like(mask, dtype=np.uint8)
        for orig, new in self.class_mapping.items():
            result[mask == orig] = new
        return result
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name.replace('.jpg', '.png'))
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = self.apply_mapping(mask)
        
        if self.transform:
            transformed = self.transform(image=img, mask=mask)
            img = transformed['image']
            mask = transformed['mask']
            mask = mask.long()
        
        return img, mask

# ============================================
# MAIN TRAINING
# ============================================
def main():
    # Create directories
    os.makedirs(Config.MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
    
    # Transforms
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Datasets
    print("📁 Creating datasets...")
    train_dataset = ADE20KDataset('training', transform=train_transform)
    val_dataset = ADE20KDataset('validation', transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Model
    print(f"\n🔧 Using device: {Config.DEVICE}")
    model = smp.Unet(
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=3,
        classes=4,
        decoder_dropout=0.3,
    ).to(Config.DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    class_weights = torch.tensor([0.8, 0.5, 1.5, 2.0]).to(Config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print(f"   Training images: {len(train_dataset)}")
    print(f"   Validation images: {len(val_dataset)}")
    print(f"   Epochs: {Config.EPOCHS}")
    print("="*60)
    
    for epoch in range(1, Config.EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{Config.EPOCHS}")
        for images, masks in pbar:
            images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(Config.DEVICE), masks.to(Config.DEVICE)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(Config.MODEL_SAVE_PATH, 'best_model.pth'))
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")
    
    print(f"\n✅ Training complete! Best val loss: {best_val_loss:.4f}")
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Config.OUTPUT_PATH, 'training_history.png'))
    plt.show()

if __name__ == "__main__":
    main()

def main():
    trainer = Trainer()
    model = trainer.train()
    print("\n✅ Training script completed!")


if __name__ == "__main__":
    main()
