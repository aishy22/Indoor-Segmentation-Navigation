"""
Inference script for indoor segmentation
Usage: python inference.py --img_path image.jpg --model_path models/large_indoor_best.pth
"""

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import segmentation_models_pytorch as smp

def load_model(model_path, device='cuda'):
    """Load trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = smp.Unet('resnet50', encoder_weights=None, classes=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, image_path, device='cuda', visualize=True):
    """Run inference on an image"""
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    
    # Preprocess
    img_resized = cv2.resize(img_rgb, (256, 256))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    # Resize back to original
    pred_full = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Generate cost map
    COST_MAP = {0: 1.0, 1: float('inf'), 2: 0.8, 3: float('inf')}
    cost_map = np.zeros_like(pred_full, dtype=np.float32)
    for class_id, cost in COST_MAP.items():
        cost_map[pred_full == class_id] = cost
    
    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_rgb)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_full, cmap='tab10', vmin=0, vmax=3)
        axes[1].set_title('Segmentation\n0:floor, 1:wall, 2:door, 3:no-go')
        axes[1].axis('off')
        
        cost_viz = np.copy(cost_map)
        cost_viz[cost_viz == float('inf')] = 2
        im = axes[2].imshow(cost_viz, cmap='hot', vmin=0, vmax=2)
        axes[2].set_title('Cost Map (darker = better)')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], ticks=[0, 1, 2])
        
        plt.tight_layout()
        plt.show()
    
    return pred_full, cost_map

def main():
    parser = argparse.ArgumentParser(description='Indoor segmentation inference')
    parser.add_argument('--img_path', required=True, help='Path to input image')
    parser.add_argument('--model_path', default='models/large_indoor_best.pth', help='Path to model')
    parser.add_argument('--output_dir', default='outputs', help='Directory to save outputs')
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = load_model(args.model_path, device)
    segmentation, cost_map = predict(model, args.img_path, device)
    
    # Save outputs
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.img_path))[0]
    np.save(os.path.join(args.output_dir, f'{base_name}_cost.npy'), cost_map)
    cv2.imwrite(os.path.join(args.output_dir, f'{base_name}_seg.png'), segmentation)
    print(f"Saved outputs to {args.output_dir}")

if __name__ == '__main__':
    main()
