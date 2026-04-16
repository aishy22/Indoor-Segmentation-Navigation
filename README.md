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
   git clone https://github.com/aishy22/Indoor-Segmentation-Navigation.git
   cd Indoor-Segmentation-Navigation
   ```
2. **Install dependencies:**
   It is recommended to use a virtual environment.
   ```bash
   pip install -r requirements.txt
   ```
3. **Download Trained Weights:**
   Download the trained `best_model.pth` file from the provided Google Drive link (insert your link here) and place it inside the `models/` directory.

## 6. How to Use
1. **Generate a Cost Map from an Image**:
   To segment an image and generate its traversal cost map, use the cost map generator:
```bash
python src/cost_map_generator.py --image path/to/input.jpg --model models/best_model.pth
```
This will output `costmap.npy` (the numerical grid) and visualization images.

2. **Plan a Safe Path**:
   Run the A* planner on the generated cost map. You can optionally specify the start and goal coordinates `(y, x)` and the safety margin.
```bash
python src/astar_planner.py \
    --cost_map costmap.npy \
    --image path/to/input.jpg \
    --start "200,50" \
    --goal "220,200" \
    --margin 25
```

This will output the optimal path coordinates and save a visualization plot `path_result.png` showing the route, the cost map, and the safety distance map.

## 7. Sample Results

## 8. Challenges and Solution

**Large Dataset**:The ADE20K dataset's 20GB size with 22,000+ images posed significant storage and memory challenges. Using Google Colab Pro's high-RAM environment and GPU acceleration, this implemented efficient PyTorch DataLoaders with batch processing to stream data dynamically during training. This approach eliminated memory bottlenecks and enabled smooth training on the full dataset.

**Path Hugging Walls**:Standard A* path planning prioritizes shortest distance, often producing paths that hug dangerously close to walls and obstacles. This project solved this by implementing a distance transform that calculates each cell's proximity to obstacles, then adding penalty costs for cells within a 25-pixel safety margin. This creates a "safety bubble" around obstacles, ensuring paths maintain safe distance while still finding efficient routes.

**Overfitting**:

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
