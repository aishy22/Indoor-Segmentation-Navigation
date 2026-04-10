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
            cost_map      : Cost map from segmentation (float64:
                            1.0=floor, 0.8=door, inf=blocked)
            safety_margin : Minimum pixels to keep from obstacles (default: 25)
            obstacle_weight: Extra cost multiplier for cells near obstacles (default: 5.0)
        """
        self.base_cost_map   = cost_map.astype(np.float64)   # FIX 4: enforce float64
        self.height, self.width = cost_map.shape
        self.safety_margin   = safety_margin
        self.obstacle_weight = obstacle_weight

        # FIX 1: use np.isinf() instead of == float('inf')
        self.obstacle_mask = np.isinf(self.base_cost_map)

        # Distance from every cell to nearest obstacle
        self.distance_map = distance_transform_edt(~self.obstacle_mask)

        # Safety-enhanced cost map and traversable mask
        self.cost_map        = self._create_safety_cost_map()
        self.traversable_mask = ~np.isinf(self.cost_map)     # FIX 1

        print(f"✅ Safety planner initialized:")
        print(f"   Safety margin    : {safety_margin} pixels")
        print(f"   Traversable area : "
              f"{100 * np.sum(self.traversable_mask) / (self.height * self.width):.1f}%")

    # ------------------------------------------------------------------
    def _create_safety_cost_map(self):
        """Add safety penalties near obstacles — fully vectorized (FIX 2)."""
        safety_cost = np.zeros_like(self.base_cost_map, dtype=np.float64)  # FIX 4

        # FIX 2: replace pixel-by-pixel loop with vectorized operation
        near_obstacle = (~self.obstacle_mask) & (self.distance_map < self.safety_margin)
        safety_cost[near_obstacle] = self.obstacle_weight * (
            1.0 - self.distance_map[near_obstacle] / self.safety_margin
        )

        final_cost = self.base_cost_map.copy()
        passable   = ~np.isinf(final_cost)                   # FIX 1
        final_cost[passable] += safety_cost[passable]
        return final_cost

    # ------------------------------------------------------------------
    def get_neighbors(self, node):
        """Get 4-directional neighbors (up, down, left, right)."""
        y, x      = node
        neighbors = []
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.height and 0 <= nx < self.width:
                if self.traversable_mask[ny, nx]:
                    neighbors.append((ny, nx))
        return neighbors

    # ------------------------------------------------------------------
    def heuristic(self, node, goal):
        """Manhattan distance heuristic."""
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    # ------------------------------------------------------------------
    def find_valid_point(self, point, search_radius=100):
        """
        Find nearest traversable point if the given point is blocked.
        Uses distance_transform_edt for fast nearest-neighbour lookup (FIX 3).

        Args:
            point        : (y, x) tuple
            search_radius: pixel radius to search

        Returns:
            Valid (y, x) point, or None if not found
        """
        y, x = point

        # Already valid?
        if 0 <= y < self.height and 0 <= x < self.width:
            if self.traversable_mask[y, x]:
                return (y, x)

        # FIX 3: vectorized nearest traversable pixel via distance transform
        dist_to_traversable = distance_transform_edt(~self.traversable_mask)

        y1 = max(0, y - search_radius)
        y2 = min(self.height, y + search_radius)
        x1 = max(0, x - search_radius)
        x2 = min(self.width,  x + search_radius)

        region   = dist_to_traversable[y1:y2, x1:x2]
        local_y, local_x = np.unravel_index(np.argmin(region), region.shape)
        ny, nx   = y1 + local_y, x1 + local_x

        return (ny, nx) if self.traversable_mask[ny, nx] else None

    # ------------------------------------------------------------------
    def find_path(self, start, goal, auto_adjust=True):
        """
        Find optimal safe path using A* algorithm.

        Args:
            start       : (y, x) start position
            goal        : (y, x) goal position
            auto_adjust : Automatically snap start/goal to nearest traversable cell

        Returns:
            path       : list of (y, x) points from start to goal (or None)
            total_cost : total traversal cost
        """
        original_start, original_goal = start, goal

        if auto_adjust:
            start = self.find_valid_point(start)
            goal  = self.find_valid_point(goal)

            if start is None:
                print(f"❌ Could not find valid start near {original_start}")
                return None, float('inf')
            if goal is None:
                print(f"❌ Could not find valid goal near {original_goal}")
                return None, float('inf')

            if start != original_start:
                print(f"📍 Start adjusted : {original_start} → {start}")
            if goal != original_goal:
                print(f"📍 Goal adjusted  : {original_goal} → {goal}")
        else:
            # FIX 5: correct 2-D indexing for traversable_mask check
            if not self.traversable_mask[start[0], start[1]]:
                print(f"❌ Start point {start} is blocked!")
                return None, float('inf')
            if not self.traversable_mask[goal[0], goal[1]]:
                print(f"❌ Goal point {goal} is blocked!")
                return None, float('inf')

        # ---- A* ----
        open_set  = [(0, 0, start)]
        counter   = 1
        came_from = {}
        g_score   = {start: 0.0}
        f_score   = {start: self.heuristic(start, goal)}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path, g_score[goal]

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + self.cost_map[neighbor[0], neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor]   = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        print("❌ No path found between start and goal!")
        return None, float('inf')

    # ------------------------------------------------------------------
    def get_safety_map(self):
        """Return safety distance map normalised to [0, 1] (0=unsafe, 1=safe)."""
        return np.clip(self.distance_map / self.safety_margin, 0, 1)

    # ------------------------------------------------------------------
    def visualize(self, image, path, start, goal, save_path=None):
        """
        Visualize path on original image, safety map, and cost map.

        Args:
            image    : Original BGR image (numpy array)
            path     : List of (y, x) waypoints
            start    : (y, x) start point
            goal     : (y, x) goal point
            save_path: Optional file path to save the figure
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # ---------- 1. Original image with path ----------
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Original Image with Path', fontsize=14)
        axes[0, 0].axis('off')

        if path:
            path_y = [p[0] for p in path]
            path_x = [p[1] for p in path]
            axes[0, 0].plot(path_x, path_y, 'cyan', linewidth=2.5,
                            alpha=0.9, label='A* Path')
        axes[0, 0].plot(start[1], start[0], 'go', markersize=12, label='Start')
        axes[0, 0].plot(goal[1],  goal[0],  'ro', markersize=12, label='Goal')
        axes[0, 0].legend()

        # ---------- 2. Safety distance map ----------
        safety_viz = self.get_safety_map()
        im1 = axes[0, 1].imshow(safety_viz, cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, 1].set_title(
            f'Safety Map\n(Green=Safe, Red=Dangerous)\nMargin={self.safety_margin}px',
            fontsize=12)
        axes[0, 1].axis('off')
        if path:
            axes[0, 1].plot(path_x, path_y, 'cyan', linewidth=2, alpha=0.9)
        axes[0, 1].plot(start[1], start[0], 'go', markersize=12)
        axes[0, 1].plot(goal[1],  goal[0],  'ro', markersize=12)
        plt.colorbar(im1, ax=axes[0, 1])

        # ---------- 3. Cost map with path ----------
        cost_viz = self.cost_map.copy()
        # FIX 1: use np.isinf() for safe inf replacement
        finite_max = cost_viz[~np.isinf(cost_viz)].max() if np.any(~np.isinf(cost_viz)) else 1.0
        cost_viz[np.isinf(cost_viz)] = finite_max + 1.0
        im2 = axes[1, 0].imshow(cost_viz, cmap='hot')
        axes[1, 0].set_title('Cost Map (with Safety Penalty)', fontsize=14)
        axes[1, 0].axis('off')
        if path:
            axes[1, 0].plot(path_x, path_y, 'cyan', linewidth=2, alpha=0.9)
        axes[1, 0].plot(start[1], start[0], 'go', markersize=12)
        axes[1, 0].plot(goal[1],  goal[0],  'ro', markersize=12)
        plt.colorbar(im2, ax=axes[1, 0])

        # ---------- 4. Legend ----------
        legend_elements = [
            mpatches.Patch(color='green',   label='Floor (cost 1.0) — Safe'),
            mpatches.Patch(color='yellow',  label='Door  (cost 0.8) — Preferred'),
            mpatches.Patch(color='red',     label='Wall/No-go (cost inf) — Blocked'),
            mpatches.Patch(color='cyan',    label='A* Path'),
            mpatches.Patch(color='green',   label='Start Point'),
            mpatches.Patch(color='red',     label='Goal Point'),
        ]
        axes[1, 1].legend(handles=legend_elements, loc='center', fontsize=12)
        axes[1, 1].set_title('Legend', fontsize=14)
        axes[1, 1].axis('off')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   Figure saved: {save_path}")
        plt.show()

    # ------------------------------------------------------------------
    def print_stats(self, path, start, goal):
        """Print path length, cost, and safety statistics."""
        if path is None:
            print("No path to analyse.")
            return

        costs         = [self.cost_map[p[0], p[1]] for p in path]
        distances     = [self.distance_map[p[0], p[1]] for p in path]

        print("\n" + "=" * 50)
        print("PATH STATISTICS")
        print("=" * 50)
        print(f"  Path length              : {len(path)} steps")
        print(f"  Total cost               : {sum(costs):.2f}")
        print(f"  Start                    : {start}")
        print(f"  Goal                     : {goal}")
        print(f"  Min distance to obstacle : {min(distances):.1f} pixels")
        print(f"  Avg distance to obstacle : {np.mean(distances):.1f} pixels")
        print(f"  Safety margin            : {self.safety_margin} pixels")


# ============================================================
# CONVENIENCE FUNCTION
# ============================================================

def plan_path(cost_map, image, start=None, goal=None,
              safety_margin=25, save_path=None):
    """
    Quick function: plan a path from a cost map and image.

    Args:
        cost_map      : float64 cost map from ColorCodedCostMapGenerator
        image         : Original BGR image
        start         : (y, x) start point  (default: lower-left quadrant)
        goal          : (y, x) goal  point  (default: lower-right quadrant)
        safety_margin : Safety margin in pixels
        save_path     : Optional path to save the visualisation figure

    Returns:
        path   : list of (y, x) waypoints
        planner: SafetyAStarPlanner instance
    """
    h, w = image.shape[:2]

    if start is None:
        start = (int(h * 0.7), int(w * 0.2))
    if goal is None:
        goal  = (int(h * 0.8), int(w * 0.7))

    print(f"\n📍 Start : {start}")
    print(f"📍 Goal  : {goal}")

    planner        = SafetyAStarPlanner(cost_map, safety_margin=safety_margin)
    path, total_cost = planner.find_path(start, goal)

    if path:
        print(f"✅ Path found! Steps: {len(path)}, Total cost: {total_cost:.2f}")
        planner.visualize(image, path, start, goal, save_path=save_path)
        planner.print_stats(path, start, goal)
    else:
        print("❌ No path found!")

    return path, planner


# ============================================================
# MAIN — Command Line Interface
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='A* path planning for indoor navigation')
    parser.add_argument('--cost_map', type=str, required=True,
                        help='Path to cost map .npy file')
    parser.add_argument('--image',    type=str, required=True,
                        help='Path to original image')
    parser.add_argument('--start',    type=str, default=None,
                        help='Start point as "y,x"')
    parser.add_argument('--goal',     type=str, default=None,
                        help='Goal point as "y,x"')
    parser.add_argument('--margin',   type=int, default=25,
                        help='Safety margin in pixels')
    parser.add_argument('--output',   type=str, default='path_result.png',
                        help='Output figure path')
    args = parser.parse_args()

    cost_map = np.load(args.cost_map)
    image    = cv2.imread(args.image)
    if image is None:
        print(f"❌ Could not load image: {args.image}")
        exit(1)

    start = goal = None
    if args.start:
        y, x  = map(int, args.start.split(','))
        start = (y, x)
    if args.goal:
        y, x = map(int, args.goal.split(','))
        goal = (y, x)

    path, planner = plan_path(cost_map, image, start, goal,
                              args.margin, args.output)
    if path:
        np.save('optimal_path.npy', np.array(path))
        print("\n✅ Path saved to: optimal_path.npy")
