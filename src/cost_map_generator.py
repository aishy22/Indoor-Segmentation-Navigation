# src/cost_map_generator.py
# Generates cost maps for A* path planning

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import os
import sys

class CostMapGenerator:
    """
    Generate cost maps from segmentation for A* path planning
    
    Cost values:
    - 1.0: Floor (safe to traverse)
    - 0.8: Door (preferred path)
    - inf: Wall/No-go (blocked)
    """
    
    COST_MAPPING = {
        0: 1.0,           # floor
        1: float('inf'),  # wall/obstacle
        2: 0.8,           # door
        3: float('inf')   # no-go (stairs, fragile)
    }
    
    CLASS_NAMES = ['floor', 'wall', 'door', 'no-go']
    
    def __init__(self, model_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = smp.Unet('resnet50', encoder_weights=None, classes=4).to(self.device)
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print("✅ Model loaded successfully!")
    
    def preprocess_image(self, image):
        """Preprocess image for model input"""
        # Resize
        img_resized = cv2.resize(image, (256, 256))
        # Normalize
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Convert to tensor
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        return img_tensor.unsqueeze(0)
    
    def predict(self, image):
        """Predict segmentation for an image"""
        img_tensor = self.preprocess_image(image).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        return pred
    
    def generate_cost_map(self, image):
        """Generate cost map from input image"""
        h, w = image.shape[:2]
        
        # Predict segmentation
        pred = self.predict(image)
        
        # Resize back to original size
        pred_full = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Generate cost map
        cost_map = np.zeros_like(pred_full, dtype=np.float32)
        for class_id, cost in self.COST_MAPPING.items():
            cost_map[pred_full == class_id] = cost
        
        return cost_map, pred_full
    
    def visualize(self, image, cost_map, segmentation, save_path=None):
        """Visualize original image, segmentation, and cost map"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Segmentation
        axes[1].imshow(segmentation, cmap='tab10', vmin=0, vmax=3)
        axes[1].set_title('Segmentation\n(0:floor, 1:wall, 2:door, 3:no-go)')
        axes[1].axis('off')
        
        # Cost map
        cost_viz = np.copy(cost_map)
        cost_viz[cost_viz == float('inf')] = 2
        im = axes[2].imshow(cost_viz, cmap='hot', vmin=0, vmax=2)
        axes[2].set_title('Cost Map\n(darker = better, red = blocked)')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], ticks=[0, 1, 2])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def save_cost_map(self, cost_map, save_path):
        """Save cost map to file"""
        np.save(save_path, cost_map)
        print(f"✅ Cost map saved to: {save_path}")

# ============================================
# EXAMPLE USAGE
# ============================================
if __name__ == "__main__":
    # Initialize generator
    generator = CostMapGenerator('/content/Indoor-Segmentation-Navigation/models/best_model.pth')
    
    # Test on an image
    test_image_path = '/content/ADEChallengeData2016/images/validation/ADE_val_00000137.jpg'
    
    if os.path.exists(test_image_path):
        image = cv2.imread(test_image_path)
        cost_map, segmentation = generator.generate_cost_map(image)
        generator.visualize(image, cost_map, segmentation, save_path='/content/cost_map_result.png')
        generator.save_cost_map(cost_map, '/content/cost_map.npy')
