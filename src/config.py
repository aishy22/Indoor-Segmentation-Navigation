# src/config.py
import torch
import os

class Config:
    # Paths for Colab
    BASE_PATH = '/content/Indoor-Segmentation-Navigation'
    DATA_PATH = '/content/ADEChallengeData2016'
    MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'models')
    OUTPUT_PATH = os.path.join(BASE_PATH, 'outputs')
    
    # Create directories
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
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
    def print_config(cls):
        print("="*60)
        print("Configuration Settings")
        print("="*60)
        print(f"Data path: {cls.DATA_PATH}")
        print(f"Model save path: {cls.MODEL_SAVE_PATH}")
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Device: {cls.DEVICE}")
