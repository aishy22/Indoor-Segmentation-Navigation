# Indoor Navigation Segmentation
## 1. Overview

This project provides a complete pipeline for indoor robot navigation using **semantic segmentation** with **A-star path planning**. The system identifies 4 key navigation classes from any indoor image and instead of relying solely on raw images or geometric obstacle detection, the system understands the semantic context of the scene to generate a semantically safe and optimal path.

## 2. Project Pipeline

The navigation system operates in four main stages:
1. **Perception:** A U-Net model with a ResNet50 encoder performs pixel-wise scene classification on an input image.
2. **Semantic Mapping:** The predicted segmentation mask is converted into a traversal cost map, assigning low travel costs to floors and doors, and infinite costs to walls and hazards.
3. **Safety Margin:** A distance transform inflates obstacles to ensure the robot maintains a safe distance from walls.
4. **Path Planning:** A safety-aware A* algorithm calculates the shortest optimal path from the start point to the goal point on the cost map.

## 3. Repository Structure

The project is modular, with core logic separated into specific scripts within the `src/` directory.

```bash
Indoor-Segmentation-Navigation/
├── src/
│   ├── data_prep.py          # loads ADE20K images and masks
│   ├── class_mapper.py       # remaps 150 ADE20K classes to 4 nav classes
│   ├── model.py              # U-Net + ResNet50 segmentation model
│   ├── train.py              # training and validation pipeline
│   ├── cost_map_generator.py # creates traversability cost maps
│   ├── astar_planner.py      # path planning on the cost map
│   └── config.py             # global settings
├── models/
│   └── .gitkeep
├── README.md
└── requirements.txt
```

## 4. Dataset and Class Mapping

The model is trained on the **ADE20K** dataset. Because ADE20K has 150 classes, we use `class_mapper.py` to compress them into **4 navigation-relevant classes** before training:

| Class ID | Semantic Class | Traversal Cost | Meaning |
| :---: | :--- | :--- | :--- |
| **0** | **Floor** (carpet, rug, etc.) | `1.0` | Safe to traverse |
| **1** | **Wall/Obstacle** (furniture, etc.) | `inf` | Blocked / Collision |
| **2** | **Door** (doorway, etc.) | `0.8` | Passable / Preferred transition |
| **3** | **No-Go Zone** (stairs, water, etc.)| `inf` | Hazardous / Blocked |

---
## 5. Installation

1. **Clone the repository:**
   ```bash
   !git clone https://github.com/aishy22/Indoor-Segmentation-Navigation.git
   %cd Indoor-Segmentation-Navigation
   ```
2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   !pip install -r requirements.txt
   ```
3. **Download Trained Weights:**
   Download the trained `best_model.pth` file from the provided Google Drive link and place it inside the `models/` directory.
First, update gdown to avoid Google Drive permission errors.
   ```bash
   !pip install --upgrade gdown
   ```
   Then download the model.
   ```bash
   !gdown "1lDgvHLZvAGs4j0RmcRjJPvc3ZRmh2beS" -O models/best_model.pth
   ```

## 6. How to Use
1. **Segmentation and cost map generator**:

```bash
import sys
import cv2
import numpy as np
from google.colab import files

sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')

from cost_map_generator import ColorCodedCostMapGenerator

print("⏳ Loading model...")
model_path = '/content/Indoor-Segmentation-Navigation/models/best_model.pth'

generator = ColorCodedCostMapGenerator(model_path=model_path)
print("✅ Model loaded!")


# STEP 1 — Upload image
print("📁 Please upload your image...")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
print(f"✅ Uploaded: {filename}")

# STEP 2 — Load image
image = cv2.imread(filename)
if image is None:
    print("❌ Could not load image. Try uploading again.")
else:
    print(f"✅ Image loaded! Shape: {image.shape}")

    # STEP 3 — Run cost map pipeline
    results = generator.process_image(image, save_prefix='/content/output')

    # STEP 4 — Visualize
    generator.visualize(results, save_path='/content/output_figure.png')

    # STEP 5 — Print statistics
    generator.print_statistics(results)

    # STEP 6 — Download outputs back to your PC
    print("\n📥 Downloading output files...")
    files.download('/content/output_figure.png')
    files.download('/content/output_colored_segmentation.png')
    files.download('/content/output_cost_viz.png')
    files.download('/content/output_cost_map.npy')
    print("✅ All done!")
```
This will output `costmap.npy` (the numerical grid) and visualization images.

2. **Plan a Safe Path**:
 
```bash
import numpy as np
from astar_planner import plan_path

# Extract cost map and image dimensions from previous step
cost_map = results['cost_map']
h, w     = image.shape[:2]

# Goal: Route from bottom-left floor → bottom-right floor
# Auto-snap both to the nearest valid traversable point
traversable = np.argwhere(~np.isinf(cost_map))
bottom      = traversable[traversable[:, 0] > int(h * 0.7)]

idx_s = np.argmin(np.abs(bottom[:, 1] - int(w * 0.1)))
idx_g = np.argmin(np.abs(bottom[:, 1] - int(w * 0.6)))

start = tuple(bottom[idx_s])
goal  = tuple(bottom[idx_g])

print(f"✅ Start : {start}")
print(f"✅ Goal  : {goal}")

# Run the safety-aware A* planner
path, planner = plan_path(
    cost_map      = cost_map,
    image         = image,
    start         = start,
    goal          = goal,
    safety_margin = 20,    # smaller margin since floor area is narrow
    save_path     = '/content/astar_result.png'
)
```

This will automatically find the best floor pixels on the left and right sides of the image, navigate around any no-go obstacles, and save the final visual output

## 7. Sample Results
<img width="1389" height="1126" alt="image" src="https://github.com/user-attachments/assets/1defba1d-c355-4de7-8018-7e4f9b68ce27" />
<img width="1389" height="1190" alt="image" src="https://github.com/user-attachments/assets/70306457-375f-403c-926f-35f6fa10ada3" />


## 8. Challenges and Solution

**Large Dataset**:The ADE20K dataset's 20GB size with 22,000+ images posed significant storage and memory challenges. Using Google Colab Pro's high-RAM environment and GPU acceleration, this implemented efficient PyTorch DataLoaders with batch processing to stream data dynamically during training. This approach eliminated memory bottlenecks and enabled smooth training on the full dataset.

**Path Hugging Walls**:Standard A* path planning prioritizes shortest distance, often producing paths that hug dangerously close to walls and obstacles. This project solved this by implementing a distance transform that calculates each cell's proximity to obstacles, then adding penalty costs for cells within a 25-pixel safety margin. This creates a "safety bubble" around obstacles, ensuring paths maintain safe distance while still finding efficient routes.

**Overfitting**:Implemented extensive data augmentation (flips, rotations, color jitter) and added a 0.3 dropout rate to the decoder to force the network to learn generalized features instead of exact pixel patterns.

## 9. Future Improvements 

**Real-time video processing**: Extend from single images to live video streams by optimizing inference speed and implementing frame-to-frame consistency for smoother robot navigation.

**Dynamic obstacle avoidance**: Incorporate real-time object detection to handle moving obstacles like people and pets, allowing the robot to dynamically re-plan paths.

**Better door detection**:Collect more door examples to improve door detection accuracy, currently the weakest class due to limited training samples.

## 9.Applications
**Home robots**: Roomba-style vacuums that understand room layouts

**Hospital robots**: Deliver medicine without bumping into equipment

**Warehouse robots**: Navigate between shelves safely

**Assisstive robots**: Help visually impaired people navigate indoors


## 👥 Contributors
* **Group 10**
* Prithula Prashun Aishy 
* WU Chun Him 
