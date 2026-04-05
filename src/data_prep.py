# src/data_prep.py

import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import pickle
import albumentations as A
from albumentations.pytorch import ToTensorV2
import sys

# Add path for Colab
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

try:
    from config import Config
except ImportError:
    class Config:
        BASE_PATH = '/content/Indoor-Segmentation-Navigation'
        DATA_PATH = '/content/ADEChallengeData2016'
        IMAGE_SIZE = 256
        CLASS_NAMES = ['floor', 'obstacle/wall', 'door', 'no-go']

class IndoorNavigationDataset(Dataset):
    """
    Custom dataset for indoor navigation with 4 classes:
    0: floor
    1: obstacle/wall
    2: door
    3: no-go (stairs, fragile zones, etc.)
    """
    
    def __init__(self, root_dir=None, transform=None, is_train=True, max_samples=None):
        if root_dir is None:
            root_dir = getattr(Config, 'DATA_PATH', '/content/ADEChallengeData2016')
        
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Load the class mapping
        base_path = getattr(Config, 'BASE_PATH', '/content/Indoor-Segmentation-Navigation')
        mapping_path = os.path.join(base_path, 'ade_to_nav_mapping.pkl')
        
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.class_mapping = pickle.load(f)
            print(f"Loaded class mapping with {len(self.class_mapping)} ADE classes")
        else:
            print(f"Warning: Mapping file not found at {mapping_path}")
            self.class_mapping = {}
        
        # Set paths based on train/val split
        if is_train:
            self.images_dir = os.path.join(root_dir, 'images/training')
            self.masks_dir = os.path.join(root_dir, 'annotations/training')
        else:
            self.images_dir = os.path.join(root_dir, 'images/validation')
            self.masks_dir = os.path.join(root_dir, 'annotations/validation')
        
        # Debug prints
        print(f"\nLooking for images in: {self.images_dir}")
        print(f"Looking for masks in: {self.masks_dir}")
        print(f"Images directory exists: {os.path.exists(self.images_dir)}")
        print(f"Masks directory exists: {os.path.exists(self.masks_dir)}")
        
        # Get all image files
        self.image_files = []
        self.mask_files = []
        
        if os.path.exists(self.images_dir) and os.path.exists(self.masks_dir):
            all_images = sorted(os.listdir(self.images_dir))
            print(f"Found {len(all_images)} files in images directory")
            
            if max_samples:
                all_images = all_images[:max_samples]
                print(f"Using {len(all_images)} images (limited by max_samples)")
            
            if len(all_images) > 0:
                print(f"Sample image files: {all_images[:5]}")
            
            for img_file in all_images:
                if img_file.endswith(('.jpg', '.jpeg')):
                    mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                    mask_path = os.path.join(self.masks_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.image_files.append(os.path.join(self.images_dir, img_file))
                        self.mask_files.append(mask_path)
            
            print(f"Loaded {len(self.image_files)} images from {'training' if is_train else 'validation'} set")
        else:
            print(f"Warning: Could not find image or mask directories")
    
    def remap_classes(self, mask):
        """Remap ADE20K class IDs to 4 navigation classes"""
        remapped = np.ones_like(mask, dtype=np.uint8)  # Default to class 1 (wall)
        
        for ade_class, nav_class in self.class_mapping.items():
            remapped[mask == ade_class] = nav_class
        
        return remapped
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_files[idx])
        if image is None:
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize
        image = cv2.resize(image, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        # Remap classes
        mask = self.remap_classes(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            mask = mask.long()
        
        return image, mask


def get_dataloaders(batch_size=8, num_workers=2, max_train_samples=None, max_val_samples=None):
    """Create train and validation dataloaders"""
    
    # Define transforms
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = IndoorNavigationDataset(transform=train_transform, is_train=True, max_samples=max_train_samples)
    
    print("\nCreating validation dataset...")
    val_dataset = IndoorNavigationDataset(transform=val_transform, is_train=False, max_samples=max_val_samples)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 50)
    print("Testing Indoor Navigation Dataset")
    print("=" * 50)
    
    dataset = IndoorNavigationDataset(is_train=True, max_samples=5)
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        image, mask = dataset[0]
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img_display = image.permute(1, 2, 0).numpy()
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=3)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
        print("\n✓ Dataset test complete!")
