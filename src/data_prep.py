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
        BATCH_SIZE = 16
        NUM_WORKERS = 2
        CLASS_NAMES = ['floor', 'obstacle/wall', 'door', 'no-go']


class IndoorNavigationDataset(Dataset):
    """
    Custom dataset for indoor navigation with 4 classes:
        0: floor
        1: obstacle/wall  (default)
        2: door
        3: no-go (stairs, hazardous zones, etc.)
    """

    def __init__(self, root_dir=None, transform=None, is_train=True, max_samples=None):
        if root_dir is None:
            root_dir = getattr(Config, 'DATA_PATH', '/content/ADEChallengeData2016')

        self.root_dir   = root_dir
        self.transform  = transform
        self.is_train   = is_train
        self.image_size = getattr(Config, 'IMAGE_SIZE', 256)  # ← use Config value

        # Load class mapping — raise error if missing so training doesn't silently fail
        base_path    = getattr(Config, 'BASE_PATH', '/content/Indoor-Segmentation-Navigation')
        mapping_path = os.path.join(base_path, 'ade_to_nav_mapping.pkl')

        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.class_mapping = pickle.load(f)
            print(f"✓ Loaded class mapping with {len(self.class_mapping)} ADE classes")
        else:
            raise FileNotFoundError(
                f"Mapping file not found at: {mapping_path}\n"
                f"  → Run class_mapper.py first to generate it."
            )

        # Set image and mask directories
        split = 'training' if is_train else 'validation'
        self.images_dir = os.path.join(root_dir, 'images', split)
        self.masks_dir  = os.path.join(root_dir, 'annotations', split)

        print(f"\nLooking for images in : {self.images_dir}")
        print(f"Looking for masks in  : {self.masks_dir}")
        print(f"Images dir exists     : {os.path.exists(self.images_dir)}")
        print(f"Masks dir exists      : {os.path.exists(self.masks_dir)}")

        # Collect matched image–mask pairs
        self.image_files = []
        self.mask_files  = []

        if os.path.exists(self.images_dir) and os.path.exists(self.masks_dir):
            all_images = sorted(os.listdir(self.images_dir))
            print(f"Found {len(all_images)} files in images directory")

            if max_samples:
                all_images = all_images[:max_samples]
                print(f"Using {len(all_images)} images (limited by max_samples)")

            if all_images:
                print(f"Sample image files: {all_images[:5]}")

            for img_file in all_images:
                if img_file.endswith(('.jpg', '.jpeg')):
                    mask_file = img_file.replace('.jpg', '.png').replace('.jpeg', '.png')
                    mask_path = os.path.join(self.masks_dir, mask_file)
                    if os.path.exists(mask_path):
                        self.image_files.append(os.path.join(self.images_dir, img_file))
                        self.mask_files.append(mask_path)

            print(f"✓ Loaded {len(self.image_files)} image-mask pairs from '{split}' set")
        else:
            print("⚠ Warning: Could not find image or mask directories.")

    def remap_classes(self, mask):
        """Remap ADE20K class IDs to 4 navigation classes."""
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
            # Skip corrupted file — return next valid sample
            print(f"⚠ Warning: Could not load image: {self.image_files[idx]}, skipping.")
            return self.__getitem__((idx + 1) % len(self))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠ Warning: Could not load mask: {self.mask_files[idx]}, skipping.")
            return self.__getitem__((idx + 1) % len(self))

        # Resize using Config.IMAGE_SIZE
        size  = self.image_size
        image = cv2.resize(image, (size, size))
        mask  = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)

        # Remap ADE20K classes → 4 navigation classes
        mask = self.remap_classes(mask)

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask  = augmented['mask'].long()
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask  = torch.from_numpy(mask).long()

        return image, mask


def get_dataloaders(
    batch_size=None,
    num_workers=None,
    max_train_samples=None,
    max_val_samples=None
):
    """Create train and validation dataloaders using Config values by default."""
    # Use Config values unless explicitly overridden
    batch_size  = batch_size  or getattr(Config, 'BATCH_SIZE',  16)
    num_workers = num_workers or getattr(Config, 'NUM_WORKERS', 2)

    # Transforms — no A.Resize needed since __getitem__ already resizes
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    print("\nCreating training dataset...")
    train_dataset = IndoorNavigationDataset(
        transform=train_transform,
        is_train=True,
        max_samples=max_train_samples
    )

    print("\nCreating validation dataset...")
    val_dataset = IndoorNavigationDataset(
        transform=val_transform,
        is_train=False,
        max_samples=max_val_samples
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    print(f"\n✓ Train batches : {len(train_loader)}")
    print(f"✓ Val batches   : {len(val_loader)}")

    return train_loader, val_loader


# ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 50)
    print("Testing Indoor Navigation Dataset")
    print("=" * 50)

    dataset = IndoorNavigationDataset(is_train=True, max_samples=5)
    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        image, mask = dataset[0]

        # Denormalize for display (undo ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])

        if torch.is_tensor(image):
            img_np   = image.permute(1, 2, 0).numpy()
            img_np   = (img_np * std + mean).clip(0, 1)  # ← denormalize
            mask_np  = mask.numpy()
        else:
            img_np  = image
            mask_np = mask

        print(f"Image shape       : {img_np.shape}")
        print(f"Mask shape        : {mask_np.shape}")
        print(f"Mask unique values: {np.unique(mask_np)}")

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask_np, cmap='tab10', vmin=0, vmax=3)
        axes[1].set_title('Segmentation Mask\n(0=floor, 1=wall, 2=door, 3=no-go)')
        axes[1].axis('off')

        plt.tight_layout()
        plt.savefig('dataset_test.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("\n✓ Dataset test complete!")
    else:
        print("No images loaded. Check dataset path and mapping file.")
