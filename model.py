# src/config.py
import torch
import os

class Config:
    # ======================
    # Paths Configuration
    # ======================
    
    # Base path - automatically gets your home directory
    BASE_PATH = os.path.expanduser("~/Documents/indoor_segmentation_project")
    
    # Data path - where ADE20K dataset is stored
    DATA_PATH = os.path.join(BASE_PATH, "data/ADEChallengeData2016")
    
    # Model save path - where trained models will be saved
    MODEL_SAVE_PATH = os.path.join(BASE_PATH, "models")
    
    # Output path - where visualizations and results will be saved
    OUTPUT_PATH = os.path.join(BASE_PATH, "outputs")
    
    # Create directories if they don't exist
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # ======================
    # Dataset Configuration
    # ======================
    
    # Number of classes for indoor navigation
    # 0: floor
    # 1: obstacle/wall
    # 2: door
    # 3: no-go (stairs, fragile zones)
    NUM_CLASSES = 4
    
    # Class names for reference and visualization
    CLASS_NAMES = ['floor', 'obstacle/wall', 'door', 'no-go']
    
    # Class colors for visualization (RGB format)
    CLASS_COLORS = [
        [0, 255, 0],      # floor: green
        [255, 0, 0],      # obstacle/wall: red
        [255, 255, 0],    # door: yellow
        [255, 0, 255]     # no-go: magenta
    ]
    
    # Image size for training (will resize images to this)
    IMAGE_SIZE = 256
    
    # ======================
    # Training Configuration
    # ======================
    
    # Batch size (adjust based on your GPU memory)
    BATCH_SIZE = 2
    
    # Number of training epochs
    EPOCHS = 2
    
    # Learning rate
    LEARNING_RATE = 1e-4

    # Add these lines for testing with small dataset
    MAX_TRAIN_SAMPLES = 20  # Only use 20 training images
    MAX_VAL_SAMPLES = 5     # Only use 5 validation images    

    # Learning rate scheduler
    SCHEDULER_STEP_SIZE = 10  # Reduce LR every 10 epochs
    SCHEDULER_GAMMA = 0.5     # Multiply LR by 0.5
    
    # Number of workers for data loading
    NUM_WORKERS = 0
    
    # Device (auto-detects CUDA if available)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ======================
    # Model Configuration
    # ======================
    
    # Model architecture
    ENCODER = 'resnet50'           # Backbone network
    ENCODER_WEIGHTS = 'imagenet'   # Pre-trained weights
    
    # ======================
    # Cost Map Configuration (for A* integration)
    # ======================
    
    # Cost mapping for each class
    # These values will be used to create cost maps for path planning
    COST_MAPPING = {
        0: 1.0,           # floor: lowest cost (most traversable)
        1: float('inf'),  # obstacle/wall: blocked (infinite cost)
        2: 0.8,           # door: low cost (preferred)
        3: float('inf')   # no-go: blocked (infinite cost)
    }
    
    # ======================
    # Logging and Checkpointing
    # ======================
    
    # How often to print training progress (in batches)
    LOG_INTERVAL = 50
    
    # How often to save model checkpoints (in epochs)
    SAVE_INTERVAL = 5
    
    # Whether to save best model based on validation loss
    SAVE_BEST_MODEL = True
    
    # ======================
    # Visualization
    # ======================
    
    # Whether to save visualization during validation
    SAVE_VISUALIZATIONS = True
    
    # Number of validation samples to visualize
    NUM_VISUALIZATIONS = 4
    
    # ======================
    # Class Mapping (ADE20K to your classes)
    # ======================
    
    # This will be populated by class_mapper.py
    # But here's a default mapping based on common ADE20K class IDs
    # You may need to adjust these based on your actual dataset
    DEFAULT_ADE_MAPPING = {
        # Floor classes (map to 0)
        2: 0,    # floor
        8: 0,    # carpet
        12: 0,   # rug
        15: 0,   # flooring
        
        # Wall/Obstacle classes (map to 1)
        1: 1,    # wall
        3: 1,    # ceiling
        4: 1,    # column
        5: 1,    # furniture
        6: 1,    # table
        7: 1,    # chair
        9: 1,    # sofa
        10: 1,   # bed
        
        # Door classes (map to 2)
        11: 2,   # door
        13: 2,   # doorway
        14: 2,   # gate
        
        # No-go classes (map to 3)
        16: 3,   # stairs
        17: 3,   # staircase
        18: 3,   # fragile
        19: 3,   # restricted
    }
    
    # ======================
    # Utility Functions
    # ======================
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("=" * 60)
        print("Configuration Settings")
        print("=" * 60)
        print(f"Base path: {cls.BASE_PATH}")
        print(f"Data path: {cls.DATA_PATH}")
        print(f"Model save path: {cls.MODEL_SAVE_PATH}")
        print(f"Output path: {cls.OUTPUT_PATH}")
        print("-" * 60)
        print(f"Number of classes: {cls.NUM_CLASSES}")
        print(f"Class names: {cls.CLASS_NAMES}")
        print(f"Image size: {cls.IMAGE_SIZE}")
        print("-" * 60)
        print(f"Batch size: {cls.BATCH_SIZE}")
        print(f"Epochs: {cls.EPOCHS}")
        print(f"Learning rate: {cls.LEARNING_RATE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Encoder: {cls.ENCODER}")
        print("-" * 60)
        print(f"Cost mapping: {cls.COST_MAPPING}")
        print("=" * 60)
    
    @classmethod
    def get_class_weights(cls):
        """
        Calculate class weights for handling imbalanced datasets
        This can be used with weighted loss functions
        """
        # This is a placeholder - you might want to calculate actual
        # class weights based on your dataset statistics
        return torch.tensor([1.0, 1.0, 1.5, 2.0])  # Example weights
    
    @classmethod
    def save_config(cls, filepath):
        """Save configuration to a text file"""
        with open(filepath, 'w') as f:
            f.write("Configuration Settings\n")
            f.write("=" * 60 + "\n")
            for key, value in cls.__dict__.items():
                if not key.startswith('__') and not callable(value):
                    f.write(f"{key}: {value}\n")
        print(f"Configuration saved to {filepath}")

# ======================
# Test the configuration
# ======================
if __name__ == "__main__":
    # Print all configuration settings
    Config.print_config()
    
    # Save config to file
    config_path = os.path.join(Config.OUTPUT_PATH, 'config.txt')
    Config.save_config(config_path)
    
    print(f"\n✓ Configuration test complete!")
    print(f"  Device: {Config.DEVICE}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
