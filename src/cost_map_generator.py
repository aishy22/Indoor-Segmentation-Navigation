# src/cost_map_generator.py

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import segmentation_models_pytorch as smp
import os


class ColorCodedCostMapGenerator:
    """
    Color-coded cost map generator for indoor navigation.
    Converts indoor scene images to segmentation masks and cost maps.
    """

    # Class definitions
    CLASS_NAMES = {
        0: 'Floor',
        1: 'Wall',
        2: 'Door',
        3: 'No-go'
    }

    # FIX 1: Colors stored in RGB order (safe for matplotlib).
    # Convert to BGR using cv2.cvtColor before any cv2.imwrite call.
    CLASS_COLORS = {
        0: [0, 255, 0],    # GREEN   - floor (safe)
        1: [255, 0, 0],    # RED     - wall (blocked)
        2: [255, 255, 0],  # YELLOW  - door (preferred)
        3: [255, 0, 255]   # MAGENTA - no-go (blocked)
    }

    # Cost values for A* planning
    COST_VALUES = {
        0: 1.0,           # floor - traversable
        1: float('inf'),  # wall - blocked
        2: 0.8,           # door - preferred
        3: float('inf')   # no-go - blocked
    }

    def __init__(self, model_path, device='cuda'):
        """
        Initialize the cost map generator.

        Args:
            model_path: Path to the trained .pth model file
            device: 'cuda' for GPU, 'cpu' for CPU
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load model architecture
        self.model = smp.Unet('resnet50', encoder_weights=None, classes=4).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Fix keys (remove 'model.' prefix if present)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v
            else:
                new_state_dict[k] = v

        self.model.load_state_dict(new_state_dict)
        self.model.eval()
        print("✅ Model loaded successfully!")

    def predict(self, image):
        """
        Predict segmentation mask for an image.

        Args:
            image: Input image (numpy array, BGR format from cv2.imread)

        Returns:
            segmentation: 2D array with class IDs (0-3)
        """
        img_resized = cv2.resize(image, (256, 256))

        # FIX 3: Convert BGR → RGB before feeding to model.
        # Deep learning models (trained on ImageNet-style data) expect RGB input.
        # Skipping this causes wrong predictions since Red and Blue channels are swapped.
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_normalized = img_rgb.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

        # Resize back to original size
        h, w = image.shape[:2]
        pred_full = cv2.resize(pred.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

        return pred_full

    def get_colored_segmentation(self, segmentation):
        """
        Convert segmentation mask to color-coded image (RGB).

        Args:
            segmentation: 2D array with class IDs (0-3)

        Returns:
            colored: RGB image with class colors (safe for matplotlib.imshow)
        """
        h, w = segmentation.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in self.CLASS_COLORS.items():
            colored[segmentation == class_id] = color

        return colored  # RGB — do NOT pass directly to cv2.imwrite

    def get_cost_map(self, segmentation):
        """
        Generate cost map for A* path planning.

        Args:
            segmentation: 2D array with class IDs (0-3)

        Returns:
            cost_map: 2D float64 array with cost values
                      (1.0 = floor, 0.8 = door, inf = blocked)
        """
        h, w = segmentation.shape
        # FIX 4: Use float64 instead of float32.
        # float32 can store inf but comparisons and downstream processing
        # are safer and more portable with float64.
        cost_map = np.zeros((h, w), dtype=np.float64)

        for class_id, cost in self.COST_VALUES.items():
            cost_map[segmentation == class_id] = cost

        return cost_map

    def get_cost_map_visualization(self, cost_map):
        """
        Convert cost map to RGB color visualization.

        Args:
            cost_map: 2D array with cost values

        Returns:
            cost_viz: RGB image (green=floor, yellow=door, red=blocked)
        """
        cost_viz = np.zeros((cost_map.shape[0], cost_map.shape[1], 3), dtype=np.uint8)
        cost_viz[cost_map == 1.0] = [0, 255, 0]    # Green  - floor (RGB)
        cost_viz[cost_map == 0.8] = [255, 255, 0]  # Yellow - door  (RGB)
        # FIX 2: Use np.isinf() instead of == float('inf').
        # Comparing float arrays with == float('inf') is unreliable across
        # platforms and numpy dtypes. np.isinf() is explicit and always correct.
        cost_viz[np.isinf(cost_map)] = [255, 0, 0]  # Red - blocked (RGB)

        return cost_viz  # RGB — do NOT pass directly to cv2.imwrite

    def process_image(self, image, save_prefix='output'):
        """
        Complete pipeline: predict, generate all outputs, and save files.

        Args:
            image: Input image (numpy array, BGR format from cv2.imread)
            save_prefix: Prefix for saved files

        Returns:
            results: Dictionary containing all outputs
        """
        # Convert to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate outputs
        segmentation = self.predict(image)
        colored_seg = self.get_colored_segmentation(segmentation)  # RGB
        cost_map = self.get_cost_map(segmentation)
        cost_viz = self.get_cost_map_visualization(cost_map)       # RGB

        # Save cost map array (float64 preserves inf safely)
        np.save(f'{save_prefix}_cost_map.npy', cost_map)

        # FIX 1 (applied here): Convert RGB → BGR before saving with cv2.imwrite.
        # cv2.imwrite expects BGR channel order; our arrays are in RGB.
        cv2.imwrite(f'{save_prefix}_colored_segmentation.png',
                    cv2.cvtColor(colored_seg, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f'{save_prefix}_cost_viz.png',
                    cv2.cvtColor(cost_viz, cv2.COLOR_RGB2BGR))

        return {
            'image': image_rgb,
            'segmentation': segmentation,
            'colored_segmentation': colored_seg,   # RGB
            'cost_map': cost_map,
            'cost_visualization': cost_viz          # RGB
        }

    def visualize(self, results, save_path=None):
        """
        Display all results side by side with legend.

        Args:
            results: Dictionary from process_image()
            save_path: Optional path to save the figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Original image
        axes[0, 0].imshow(results['image'])
        axes[0, 0].set_title('Original Image', fontsize=14)
        axes[0, 0].axis('off')

        # Colored segmentation (already RGB — safe for imshow)
        axes[0, 1].imshow(results['colored_segmentation'])
        axes[0, 1].set_title('Segmentation\n(GREEN=floor, RED=wall, YELLOW=door, MAGENTA=no-go)',
                              fontsize=12)
        axes[0, 1].axis('off')

        # Cost map visualization (already RGB — safe for imshow)
        axes[1, 0].imshow(results['cost_visualization'])
        axes[1, 0].set_title('Cost Map\n(GREEN=floor, YELLOW=door, RED=blocked)', fontsize=12)
        axes[1, 0].axis('off')

        # Legend
        legend_elements = [
            mpatches.Patch(color='green',   label='Floor (cost=1.0) - Safe to traverse'),
            mpatches.Patch(color='red',     label='Wall (cost=inf) - Blocked'),
            mpatches.Patch(color='yellow',  label='Door (cost=0.8) - Preferred path'),
            mpatches.Patch(color='magenta', label='No-go (cost=inf) - Blocked')
        ]
        axes[1, 1].legend(handles=legend_elements, loc='center', fontsize=12)
        axes[1, 1].set_title('Legend', fontsize=14)
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    def print_statistics(self, results):
        """
        Print class distribution and cost map statistics.

        Args:
            results: Dictionary from process_image()
        """
        segmentation = results['segmentation']
        total_pixels = segmentation.size

        print("\n" + "=" * 50)
        print("CLASS DISTRIBUTION")
        print("=" * 50)

        for class_id in range(4):
            count = np.sum(segmentation == class_id)
            percentage = 100 * count / total_pixels
            print(f"  {self.CLASS_NAMES[class_id]}: {percentage:.1f}% ({count:,} pixels)")

        cost_map = results['cost_map']
        traversable = np.sum((cost_map == 1.0) | (cost_map == 0.8))
        # FIX 2 (also applied here): np.isinf() instead of == float('inf')
        blocked = np.sum(np.isinf(cost_map))

        print("\n" + "=" * 50)
        print("COST MAP STATISTICS")
        print("=" * 50)
        print(f"  Traversable area: {100 * traversable / total_pixels:.1f}%")
        print(f"  Blocked area:     {100 * blocked / total_pixels:.1f}%")


# ============================================
# MAIN - Command Line Interface
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate cost maps for indoor navigation')
    parser.add_argument('--model',  type=str, required=True, help='Path to trained model (.pth)')
    parser.add_argument('--image',  type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, default='output', help='Output file prefix')
    parser.add_argument('--device', type=str, default='cuda',   help='Device (cuda/cpu)')

    args = parser.parse_args()

    # Initialize generator
    generator = ColorCodedCostMapGenerator(args.model, device=args.device)

    # Load image (cv2.imread returns BGR — expected by this class)
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ Could not load image: {args.image}")
        exit(1)

    results = generator.process_image(image, save_prefix=args.output)
    generator.visualize(results, save_path=f'{args.output}_figure.png')
    generator.print_statistics(results)

    print(f"\n✅ Done! Files saved:")
    print(f"   - {args.output}_cost_map.npy")
    print(f"   - {args.output}_colored_segmentation.png")
    print(f"   - {args.output}_cost_viz.png")
    print(f"   - {args.output}_figure.png")
