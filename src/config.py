# src/config.py

import torch
import os

class Config:
    # Paths for Colab
    BASE_PATH = '/content/Indoor-Segmentation-Navigation'
    DATA_PATH = '/content/ADEChallengeData2016'
    MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'models')
    OUTPUT_PATH = os.path.join(BASE_PATH, 'outputs')

    # Dataset
    NUM_CLASSES = 4
    CLASS_NAMES = ['floor', 'obstacle/wall', 'door', 'no-go']
    IMAGE_SIZE = 256

    # Training
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    NUM_WORKERS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'

    # Cost Map for A*
    COST_MAPPING = {0: 1.0, 1: float('inf'), 2: 0.8, 3: float('inf')}

    @classmethod
    def setup(cls):
        """Create necessary directories. Call this once at the start of train.py."""
        os.makedirs(cls.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(cls.OUTPUT_PATH, exist_ok=True)
        print(f"✓ Directories ready:")
        print(f"  Model save path : {cls.MODEL_SAVE_PATH}")
        print(f"  Output path     : {cls.OUTPUT_PATH}")

    @classmethod
    def print_config(cls):
        print("=" * 60)
        print("Configuration Settings")
        print("=" * 60)
        print(f"  Data path       : {cls.DATA_PATH}")
        print(f"  Model save path : {cls.MODEL_SAVE_PATH}")
        print(f"  Device          : {cls.DEVICE}")
        print(f"  Encoder         : {cls.ENCODER} ({cls.ENCODER_WEIGHTS})")
        print(f"  Batch size      : {cls.BATCH_SIZE}")
        print(f"  Epochs          : {cls.EPOCHS}")
        print(f"  Learning rate   : {cls.LEARNING_RATE}")
        print(f"  Image size      : {cls.IMAGE_SIZE}x{cls.IMAGE_SIZE}")
        print(f"  Num classes     : {cls.NUM_CLASSES} → {cls.CLASS_NAMES}")
        print(f"  Cost mapping    : {cls.COST_MAPPING}")
        print("=" * 60)


if __name__ == "__main__":
    Config.setup()
    Config.print_config()
         
        
