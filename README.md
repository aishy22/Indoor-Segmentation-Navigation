# Indoor Navigation Segmentation
## Overview

This project provides a complete pipeline for indoor robot navigation using **semantic segmentation**. The system identifies 4 key navigation classes from any indoor image and generates optimal, safe paths using an enhanced A* algorithm with configurable safety margins.

## Navigation Classes
- **0: floor** - Traversable (cost 1.0)
- **1: wall** - Blocked (cost inf)
- **2: door** - Preferred path (cost 0.8)
- **3: no-go** - Blocked (cost inf)

## How to Run 
### ============================================
### INDOOR NAVIGATION - UPLOAD & PROCESS
### ============================================
```bash
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

print("="*60)
print("INDOOR NAVIGATION SYSTEM")
print("="*60)

# Step 1: Setup
!pip install -q segmentation-models-pytorch albumentations opencv-python matplotlib numpy gdown scipy

# Step 2: Clone or use existing repo
if not os.path.exists('/content/Indoor-Segmentation-Navigation'):
    !git clone https://github.com/aishy22/Indoor-Segmentation-Navigation.git
%cd /content/Indoor-Segmentation-Navigation

# Step 3: Add to path and import YOUR files
sys.path.insert(0, 'src')
from cost_map_generator import ColorCodedCostMapGenerator
from astar_planner import plan_path

# Step 4: Download model (if not exists)
!gdown "https://drive.google.com/uc?id=1TArpiIxo_Vt1rLNZtFdhnS_66N2Hh7n-" -O models/best_model.pth -q

# Step 5: Load model
generator = ColorCodedCostMapGenerator('models/best_model.pth')

# Step 6: Upload and process
print("\n📸 Upload your indoor image:")
uploaded = files.upload()

for filename in uploaded.keys():
    print(f"\nProcessing: {filename}")
    image = cv2.imread(filename)
    h, w = image.shape[:2]
    
    # Generate cost map
    results = generator.process_image(image, save_prefix=f'output_{filename[:-4]}')
    cost_map = results['cost_map']
    
    # Plan path 
    start = (int(h * 0.7), int(w * 0.2))
    goal = (int(h * 0.8), int(w * 0.7))
    path, planner = plan_path(cost_map, image, start, goal, safety_margin=25)
    
    # Show results
    generator.visualize(results, save_path=f'result_{filename[:-4]}.png')
    
    # Download
    files.download(f'output_{filename[:-4]}_cost_map.npy')
    files.download(f'result_{filename[:-4]}.png')

print("\n✅ Done!")
```
## Challenges and Solution
**Large Dataset**:The ADE20K dataset's 20GB size with 22,000+ images posed significant storage and memory challenges. Using Google Colab Pro's high-RAM environment and GPU acceleration, this implemented efficient PyTorch DataLoaders with batch processing to stream data dynamically during training. This approach eliminated memory bottlenecks and enabled smooth training on the full dataset.

**Path Hugging Walls**:Standard A* path planning prioritizes shortest distance, often producing paths that hug dangerously close to walls and obstacles. This project solved this by implementing a distance transform that calculates each cell's proximity to obstacles, then adding penalty costs for cells within a 25-pixel safety margin. This creates a "safety bubble" around obstacles, ensuring paths maintain safe distance while still finding efficient routes.

**Overfitting**:

## Future Improvements 
**Real-time video processing**: Extend from single images to live video streams by optimizing inference speed and implementing frame-to-frame consistency for smoother robot navigation.

**Dynamic obstacle avoidance**: Incorporate real-time object detection to handle moving obstacles like people and pets, allowing the robot to dynamically re-plan paths.

**Better door detection**:Collect more door examples to improve door detection accuracy, currently the weakest class due to limited training samples.

## Applications
**Home robots**: Roomba-style vacuums that understand room layouts

**Hospital robots**: Deliver medicine without bumping into equipment

**Warehouse robots**: Navigate between shelves safely

**Assisstive robots**: Help visually impaired people navigate indoors
