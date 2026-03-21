"""
Cost map generator for A* path planning
Converts segmentation output to cost map for robot navigation
"""

import numpy as np

class CostMapGenerator:
    """
    Generate cost maps from segmentation for A* path planning
    
    Cost values:
    - 1.0: Floor (safe to traverse)
    - 0.8: Door (preferred path)
    - inf: Blocked (wall, no-go zones)
    """
    
    COST_MAPPING = {
        0: 1.0,           # floor
        1: float('inf'),  # wall/obstacle
        2: 0.8,           # door
        3: float('inf')   # no-go (stairs, fragile)
    }
    
    @classmethod
    def generate(cls, segmentation):
        """
        Convert segmentation to cost map
        
        Args:
            segmentation: 2D numpy array with class IDs (0-3)
            
        Returns:
            cost_map: 2D numpy array with traversal costs
        """
        cost_map = np.zeros_like(segmentation, dtype=np.float32)
        for class_id, cost in cls.COST_MAPPING.items():
            cost_map[segmentation == class_id] = cost
        return cost_map
    
    @classmethod
    def save(cls, cost_map, filepath):
        """Save cost map to file"""
        np.save(filepath, cost_map)
    
    @classmethod
    def load(cls, filepath):
        """Load cost map from file"""
        return np.load(filepath)
    
    @classmethod
    def get_traversable_mask(cls, cost_map):
        """Get mask of traversable cells (not blocked)"""
        return cost_map != float('inf')
    
    @classmethod
    def get_floor_mask(cls, cost_map):
        """Get mask of floor cells"""
        return cost_map == 1.0
    
    @classmethod
    def get_door_mask(cls, cost_map):
        """Get mask of door cells"""
        return cost_map == 0.8

# Example usage
if __name__ == '__main__':
    # Create sample segmentation
    sample_seg = np.array([
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3]
    ])
    
    cost_map = CostMapGenerator.generate(sample_seg)
    print("Sample cost map:")
    print(cost_map)
