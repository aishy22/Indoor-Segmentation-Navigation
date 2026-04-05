"""
A* Path Planner for Indoor Navigation
=====================================
Plans optimal paths using cost maps from segmentation.
Maintains safety margin from walls and obstacles.

Usage:
    from astar_planner import plan_path, SafetyAStarPlanner
    
    # Method 1: Quick planning
    path, planner = plan_path(cost_map, image, start, goal)
    
    # Method 2: Full control
    planner = SafetyAStarPlanner(cost_map, safety_margin=25)
    path, cost = planner.find_path(start, goal)
    planner.visualize(image, path, start, goal)
"""

import numpy as np
import heapq
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import distance_transform_edt


class SafetyAStarPlanner:
    """
    A* path planner with safety margin from obstacles.
    Maintains minimum distance from walls and no-go zones.
    """
    
    def __init__(self, cost_map, safety_margin=25, obstacle_weight=5.0):
        """
        Initialize the safety-aware planner.
        
        Args:
            cost_map: Cost map from segmentation (1.0=floor, 0.8=door, inf=blocked)
            safety_margin: Minimum pixels to keep from obstacles (default: 25)
            obstacle_weight: Extra cost multiplier for cells near obstacles (default: 5.0)
        """
        self.base_cost_map = cost_map
        self.height, self.width = cost_map.shape
        self.safety_margin = safety_margin
        self.obstacle_weight = obstacle_weight
        
        # Create obstacle mask
        self.obstacle_mask = cost_map == float('inf')
        
        # Calculate distance to obstacles
        self.distance_map = distance_transform_edt(~self.obstacle_mask)
        
        # Create safety-enhanced cost map
        self.cost_map = self._create_safety_cost_map()
        self.traversable_mask = self.cost_map != float('inf')
        
        print(f"✅ Safety planner initialized:")
        print(f"   Safety margin: {safety_margin} pixels")
        print(f"   Traversable area: {100 * np.sum(self.traversable_mask) / (self.height*self.width):.1f}%")
    
    def _create_safety_cost_map(self):
        """Add safety penalties near obstacles."""
        safety_cost = np.zeros_like(self.base_cost_map, dtype=np.float32)
        
        for y in range(self.height):
            for x in range(self.width):
                if not self.obstacle_mask[y, x]:
                    dist = self.distance_map[y, x]
                    if dist < self.safety_margin:
                        penalty = self.obstacle_weight * (1 - dist / self.safety_margin)
                        safety_cost[y, x] = penalty
        
        final_cost = self.base_cost_map.copy()
        mask = final_cost != float('inf')
        final_cost[mask] += safety_cost[mask]
        return final_cost
    
    def get_neighbors(self, node):
        """Get 4-directional neighbors (up, down, left, right)."""
        y, x = node
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                if self.traversable_mask[ny, nx]:
                    neighbors.append((ny, nx))
        return neighbors
    
    def heuristic(self, node, goal):
        """Manhattan distance heuristic."""
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    
    def find_valid_point(self, point, search_radius=100):
        """
        Find nearest traversable point if the given point is blocked.
        
        Args:
            point: (y, x) tuple
            search_radius: Radius to search
        
        Returns:
            Valid (y, x) point or None if not found
        """
        y, x = point
        
        # Check if point is valid
        if 0 <= y < self.height and 0 <= x < self.width:
            if self.traversable_mask[y, x]:
                return (y, x)
        
        # Search in expanding square
        for r in range(1, search_radius):
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < self.height and 0 <= nx < self.width:
                        if self.traversable_mask[ny, nx]:
                            return (ny, nx)
        return None
    
    def find_path(self, start, goal, auto_adjust=True):
        """
        Find optimal safe path using A* algorithm.
        
        Args:
            start: (y, x) tuple for start position
            goal: (y, x) tuple for goal position
            auto_adjust: Automatically adjust start/goal if blocked
        
        Returns:
            path: list of (y, x) points from start to goal
            total_cost: total traversal cost
        """
        original_start = start
        original_goal = goal
        
        # Adjust points if needed
        if auto_adjust:
            start = self.find_valid_point(start)
            goal = self.find_valid_point(goal)
            
            if start is None:
                print(f"❌ Could not find valid start point near {original_start}")
                return None, float('inf')
            if goal is None:
                print(f"❌ Could not find valid goal point near {original_goal}")
                return None, float('inf')
            
            if start != original_start:
                print(f"📍 Start adjusted: {original_start} → {start}")
            if goal != original_goal:
                print(f"📍 Goal adjusted: {original_goal} → {goal}")
        else:
            if not self.traversable_mask[start]:
                print(f"❌ Start point {start} is blocked!")
                return None, float('inf')
            if not self.traversable_mask[goal]:
                print(f"❌ Goal point {goal} is blocked!")
                return None, float('inf')
        
        # A* algorithm
        open_set = [(0, 0, start)]
        counter = 1
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        while open_set:
            current_f, _, current = heapq.heappop(open_set)
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, g_score[goal]
            
            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.cost_map[neighbor]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1
        
        print("❌ No path found between start and goal!")
        return None, float('inf')
    
    def get_safety_map(self):
        """Get safety distance map (0=unsafe, 1=safe)."""
        return np.clip(self.distance_map / self.safety_margin, 0, 1)
    
    def visualize(self, image, path, start, goal, save_path=None):
        """
        Visualize path on original image, safety map, and cost map.
        
        Args:
            image: Original BGR image
            path: List of (y, x) points
            start: (y, x) start point
            goal: (y, x) goal point
            save_path: Optional path to save figure
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Original image with path
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image with Path')
        axes[0, 0].axis('off')
        
        if path:
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            axes[0, 0].plot(path_x, path_y, 'cyan', linewidth=2.5, alpha=0.9, label='A* Path')
        axes[0, 0].plot(start[1], start[0], 'go', markersize=12, label='Start')
        axes[0, 0].plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
        axes[0, 0].legend()
        
        # 2. Safety distance map
        safety_viz = self.get_safety_map()
        im1 = axes[0, 1].imshow(safety_viz, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 1].set_title(f'Safety Map\n(Green=Safe, Red=Dangerous)\nMargin={self.safety_margin}px')
        axes[0, 1].axis('off')
        if path:
            axes[0, 1].plot(path_x, path_y, 'cyan', linewidth=2, alpha=0.9)
        axes[0, 1].plot(start[1], start[0], 'go', markersize=12)
        axes[0, 1].plot(goal[1], goal[0], 'ro', markersize=12)
        plt.colorbar(im1, ax=axes[0, 1])
        
        # 3. Cost map with path
        cost_viz = np.copy(self.cost_map)
        cost_viz[cost_viz == float('inf')] = cost_viz[cost_viz != float('inf')].max() + 1
        im2 = axes[1, 0].imshow(cost_viz, cmap='hot')
        axes[1, 0].set_title('Cost Map (with Safety Penalty)')
        axes[1, 0].axis('off')
        if path:
            axes[1, 0].plot(path_x, path_y, 'cyan', linewidth=2, alpha=0.9)
        axes[1, 0].plot(start[1], start[0], 'go', markersize=12)
        axes[1, 0].plot(goal[1], goal[0], 'ro', markersize=12)
        plt.colorbar(im2, ax=axes[1, 0])
        
        # 4. Legend
        legend_elements = [
            mpatches.Patch(color='green', label='Floor (cost 1.0) - Safe'),
            mpatches.Patch(color='yellow', label='Door (cost 0.8) - Preferred'),
            mpatches.Patch(color='red', label='Wall/No-go (cost inf) - Blocked'),
            mpatches.Patch(color='cyan', label='A* Path'),
            mpatches.Patch(color='green', label='Start Point'),
            mpatches.Patch(color='red', label='Goal Point')
        ]
        axes[1, 1].legend(handles=legend_elements, loc='center', fontsize=12)
        axes[1, 1].set_title('Legend')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Figure saved: {save_path}")
        plt.show()
    
    def print_stats(self, path, start, goal):
        """Print path statistics."""
        if path is None:
            print("No path to analyze")
            return
        
        total_cost = 0
        min_distances = []
        
        for point in path:
            total_cost += self.cost_map[point]
            dist = self.distance_map[point]
            min_distances.append(dist)
        
        print("\n" + "="*50)
        print("PATH STATISTICS")
        print("="*50)
        print(f"  Path length: {len(path)} steps")
        print(f"  Total cost: {total_cost:.2f}")
        print(f"  Start: {start}")
        print(f"  Goal: {goal}")
        print(f"  Min distance to obstacle: {min(min_distances):.1f} pixels")
        print(f"  Avg distance to obstacle: {np.mean(min_distances):.1f} pixels")
        print(f"  Safety margin: {self.safety_margin} pixels")


# ============================================
# CONVENIENCE FUNCTIONS
# ============================================

def plan_path(cost_map, image, start=None, goal=None, safety_margin=25, save_path=None):
    """
    Quick function to plan a path from cost map and image.
    
    Args:
        cost_map: Cost map from segmentation
        image: Original BGR image
        start: (y, x) start point (default: bottom-left)
        goal: (y, x) goal point (default: bottom-right)
        safety_margin: Safety margin in pixels
        save_path: Where to save visualization
    
    Returns:
        path: List of waypoints
        planner: SafetyAStarPlanner instance
    """
    h, w = image.shape[:2]
    
    # Default start and goal
    if start is None:
        start = (int(h * 0.7), int(w * 0.2))
    if goal is None:
        goal = (int(h * 0.8), int(w * 0.7))
    
    print(f"\n📍 Start: {start}")
    print(f"📍 Goal: {goal}")
    
    # Create planner
    planner = SafetyAStarPlanner(cost_map, safety_margin=safety_margin)
    
    # Find path
    path, total_cost = planner.find_path(start, goal)
    
    if path:
        planner.visualize(image, path, start, goal, save_path=save_path)
        planner.print_stats(path, start, goal)
    else:
        print("❌ No path found!")
    
    return path, planner


# ============================================
# MAIN - Command Line Interface
# ============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='A* path planning for indoor navigation')
    parser.add_argument('--cost_map', type=str, required=True, help='Path to cost map .npy file')
    parser.add_argument('--image', type=str, required=True, help='Path to original image')
    parser.add_argument('--start', type=str, default=None, help='Start point as "y,x"')
    parser.add_argument('--goal', type=str, default=None, help='Goal point as "y,x"')
    parser.add_argument('--margin', type=int, default=25, help='Safety margin in pixels')
    parser.add_argument('--output', type=str, default='path_result.png', help='Output image path')
    
    args = parser.parse_args()
    
    # Load cost map and image
    cost_map = np.load(args.cost_map)
    image = cv2.imread(args.image)
    
    # Parse start and goal
    start = None
    goal = None
    if args.start:
        y, x = map(int, args.start.split(','))
        start = (y, x)
    if args.goal:
        y, x = map(int, args.goal.split(','))
        goal = (y, x)
    
    # Plan path
    path, planner = plan_path(cost_map, image, start, goal, args.margin, args.output)
    
    if path:
        np.save('optimal_path.npy', np.array(path))
        print("\n✅ Path saved to: optimal_path.npy")
