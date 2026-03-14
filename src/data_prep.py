# src/data_prep.py
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
import pickle
from src.config import Config

class IndoorNavigationDataset(Dataset):
    """
    Custom dataset for indoor navigation with 4 classes:
    0: floor
    1: obstacle/wall
    2: door
    3: no-go (stairs, fragile zones, etc.)
    """
    
    def __init__(self, root_dir=Config.DATA_PATH, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # Load the class mapping
        mapping_path = os.path.join(Config.BASE_PATH, 'ade_to_nav_mapping.pkl')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.class_mapping = pickle.load(f)
            print(f"Loaded class mapping with {len(self.class_mapping)} ADE classes")
        else:
            print("Warning: Mapping file not found. Please run class_mapper.py first")
            self.class_mapping = {}
        
        # Set paths based on train/val split
        if is_train:
            self.images_dir = os.path.join(root_dir, 'images/training')
            self.masks_dir = os.path.join(root_dir, 'annotations/training')
            # Set max_samples from Config
            self.max_samples = getattr(Config, 'MAX_TRAIN_SAMPLES', None)
        else:
            self.images_dir = os.path.join(root_dir, 'images/validation')
            self.masks_dir = os.path.join(root_dir, 'annotations/validation')
            # Set max_samples from Config
            self.max_samples = getattr(Config, 'MAX_VAL_SAMPLES', None)
        
        # Debug prints
        print(f"\nLooking for images in: {self.images_dir}")
        print(f"Looking for masks in: {self.masks_dir}")
        print(f"Images directory exists: {os.path.exists(self.images_dir)}")
        print(f"Masks directory exists: {os.path.exists(self.masks_dir)}")
        if self.max_samples:
            print(f"Max samples: {self.max_samples} (for testing)")
        
        # Get all image files
        self.image_files = []
        self.mask_files = []
        
        if os.path.exists(self.images_dir) and os.path.exists(self.masks_dir):
            # Get all image files
            all_images = sorted(os.listdir(self.images_dir))
            print(f"Found {len(all_images)} files in images directory")
            
            if self.max_samples:
                all_images = all_images[:self.max_samples]
                print(f"Using {len(all_images)} images for testing")
            
            if len(all_images) > 0:
                print(f"Sample image files: {all_images[:5]}")
            
            # Filter to only include images that have corresponding masks
            for i, img_file in enumerate(all_images):
                # Check if we've reached the max samples limit (additional safeguard)
                if self.max_samples and i >= self.max_samples:
                    print(f"Reached max samples limit ({self.max_samples})")
                    break
                    
                if img_file.endswith(('.jpg', '.jpeg')):
                    # Convert .jpg to .png for mask files
                    mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                    mask_path = os.path.join(self.masks_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.image_files.append(os.path.join(self.images_dir, img_file))
                        self.mask_files.append(mask_path)
                        
                elif img_file.endswith('.png'):
                    # If image is .png, keep same extension
                    mask_file = img_file
                    mask_path = os.path.join(self.masks_dir, mask_file)
                    
                    if os.path.exists(mask_path):
                        self.image_files.append(os.path.join(self.images_dir, img_file))
                        self.mask_files.append(mask_path)
            
            print(f"Loaded {len(self.image_files)} images from {'training' if is_train else 'validation'} set")
        else:
            print(f"Warning: Could not find image or mask directories")
            # Try to find the correct paths
            self.find_and_load_files()
   
    
    def find_and_load_files(self):
        """Try to find the dataset in alternative locations"""
        print("\nTrying to find dataset in alternative locations...")
        
        # Common alternative paths
        possible_paths = [
            ('images/training', 'annotations/training'),
            ('training/images', 'training/annotations'),
            ('train/images', 'train/annotations'),
            ('img/train', 'ann/train'),
            ('rgb/train', 'seg/train'),
        ]
        
        for img_subpath, mask_subpath in possible_paths:
            test_img_dir = os.path.join(self.root_dir, img_subpath)
            test_mask_dir = os.path.join(self.root_dir, mask_subpath)
            
            if os.path.exists(test_img_dir) and os.path.exists(test_mask_dir):
                print(f"Found alternative paths!")
                print(f"  Images: {test_img_dir}")
                print(f"  Masks: {test_mask_dir}")
                
                self.images_dir = test_img_dir
                self.masks_dir = test_mask_dir
                
                # Load files from these paths
                all_images = sorted(os.listdir(self.images_dir))
                for img_file in all_images:
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        mask_file = img_file
                        mask_path = os.path.join(self.masks_dir, mask_file)
                        
                        if os.path.exists(mask_path):
                            self.image_files.append(os.path.join(self.images_dir, img_file))
                            self.mask_files.append(mask_path)
                
                print(f"Loaded {len(self.image_files)} images from alternative paths")
                return
        
        print("Could not find dataset in alternative locations")
    
    def remap_classes(self, mask):
        """
        Remap ADE20K class IDs to your 4 navigation classes
        """
        # Create output mask with default class 1 (obstacle/wall)
        remapped = np.ones_like(mask, dtype=np.uint8) * 1  # Default to class 1
        
        # Apply the mapping
        for ade_class, nav_class in self.class_mapping.items():
            remapped[mask == ade_class] = nav_class
        
        # Ensure mask values are within 0-3
        remapped = np.clip(remapped, 0, 3)
        
        return remapped
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_files[idx])
        if image is None:
            print(f"Warning: Could not load image {self.image_files[idx]}")
            # Return a dummy tensor if image fails to load
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (segmentation annotation)
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Could not load mask {self.mask_files[idx]}")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Remap classes
        mask = self.remap_classes(mask)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Convert to tensor if no transform
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask).long()
        
        return image, mask

def get_dataloaders(batch_size=8, num_workers=2):
    """
    Create train and validation dataloaders
    """
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Define transforms - simpler for testing
    train_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    # Create datasets
    print("\nCreating training dataset...")
    train_dataset = IndoorNavigationDataset(transform=train_transform, is_train=True)
    
    print("\nCreating validation dataset...")
    val_dataset = IndoorNavigationDataset(transform=val_transform, is_train=False)
    
    # Create dataloaders with memory optimization
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=False,  # Disable pin_memory for CPU
        prefetch_factor=None  # Remove prefetching
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    return train_loader, val_loader
# Test the dataset
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 50)
    print("Testing Indoor Navigation Dataset")
    print("=" * 50)
    
    # Test with no transform first
    dataset = IndoorNavigationDataset(transform=None, is_train=True)
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        # Get a sample
        image, mask = dataset[0]
        
        print(f"Image shape: {image.shape}")
        print(f"Mask shape: {mask.shape}")
        print(f"Mask unique values: {torch.unique(mask)}")
        
        # Count pixels per class
        print(f"\nPixel distribution:")
        for class_id in range(4):
            count = (mask == class_id).sum().item()
            percentage = 100 * count / (mask.shape[0] * mask.shape[1])
            print(f"  Class {class_id} ({Config.CLASS_NAMES[class_id]}): {count} pixels ({percentage:.2f}%)")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Image (convert from tensor CHW to HWC)
        img_display = image.permute(1, 2, 0).numpy()
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Mask
        mask_display = mask.numpy()
        im = axes[1].imshow(mask_display, cmap='tab10', vmin=0, vmax=3)
        axes[1].set_title('Segmentation Mask')
        axes[1].axis('off')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1], ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(Config.CLASS_NAMES)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(Config.OUTPUT_PATH, 'dataset_sample.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Sample saved to '{output_path}'")
        
        # Test dataloader
        print("\n" + "=" * 50)
        print("Testing DataLoader")
        print("=" * 50)
        
        train_loader, val_loader = get_dataloaders(batch_size=4)
        
        if len(train_loader) > 0:
            # Get a batch
            batch_images, batch_masks = next(iter(train_loader))
            print(f"Batch images shape: {batch_images.shape}")
            print(f"Batch masks shape: {batch_masks.shape}")
            print(f"Batch masks unique values: {torch.unique(batch_masks)}")
            
            print("\n✓ Dataset and dataloader test complete!")
        else:
            print("\n✗ No training data loaded")
    else:
        print("\n✗ No images found. Please check your dataset path.")
        print(f"Current data path: {Config.DATA_PATH}")
        
        # List what's in the data directory
        print("\nContents of data directory:")
        if os.path.exists(Config.DATA_PATH):
            for item in os.listdir(Config.DATA_PATH):
                print(f"  - {item}")
        
        print("\nTroubleshooting tips:")
        print("1. Run 'python src/class_mapper.py' first to create the mapping")
        print("2. Check if the dataset exists at:", Config.DATA_PATH)
        print("3. Verify the folder structure:")
        print("   Should have: images/training/ and annotations/training/")
        print("4. List contents: ls -la", Config.DATA_PATH)
