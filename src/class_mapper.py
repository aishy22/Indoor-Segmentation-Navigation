# src/class_mapper.py
import numpy as np
import os
import pickle
import sys

# Add path setup for Colab
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation')
sys.path.insert(0, '/content/Indoor-Segmentation-Navigation/src')

# Import config - now works in Colab
try:
    from config import Config
except ImportError:
    # Fallback config for when running standalone
    class Config:
        BASE_PATH = '/content/Indoor-Segmentation-Navigation'
        DATA_PATH = '/content/ADEChallengeData2016'
        OUTPUT_PATH = '/content/outputs'
        CLASS_NAMES = ['floor', 'obstacle/wall', 'door', 'no-go']

class ADEClassMapper:
    """
    Maps ADE20K's 150 classes to 4 navigation classes:
    0: floor
    1: obstacle/wall
    2: door
    3: no-go (stairs, fragile zones, etc.)
    """
    
    def __init__(self):
        self.ade_class_names = self.load_ade_classes()
        self.mapping = self.create_mapping()
        
    def load_ade_classes(self):
        """Load ADE20K class names from objectInfo150.txt"""
        class_names = {}
        # Use DATA_PATH from config, with fallback
        data_path = getattr(Config, 'DATA_PATH', '/content/ADEChallengeData2016')
        object_info_path = os.path.join(data_path, 'objectInfo150.txt')
        
        print(f"Looking for objectInfo150.txt at: {object_info_path}")
        
        try:
            with open(object_info_path, 'r') as f:
                # Skip header line
                next(f)
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 5:
                        idx = int(parts[0])
                        name = parts[4].strip().lower()
                        class_names[idx] = name
            print(f"✓ Loaded {len(class_names)} ADE20K classes")
            
            # Print first few classes to verify
            print("\nFirst 10 classes:")
            for i, (cid, name) in enumerate(list(class_names.items())[:10]):
                print(f"  ID {cid}: {name}")
                
        except FileNotFoundError:
            print(f"✗ Could not find {object_info_path}")
            print("Looking for dataset in common locations...")
            
            # Try to find the dataset
            possible_paths = [
                '/content/ADEChallengeData2016/objectInfo150.txt',
                '/content/ADEChallengeData2016/objectInfo150.txt',
                '/content/data/ADEChallengeData2016/objectInfo150.txt',
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    object_info_path = path
                    print(f"Found at: {path}")
                    try:
                        with open(object_info_path, 'r') as f:
                            next(f)
                            for line in f:
                                parts = line.strip().split('\t')
                                if len(parts) >= 5:
                                    idx = int(parts[0])
                                    name = parts[4].strip().lower()
                                    class_names[idx] = name
                        print(f"✓ Loaded {len(class_names)} ADE20K classes")
                        return class_names
                    except:
                        continue
            
            print("Using fallback class names...")
            class_names = self.get_fallback_classes()
            
        return class_names
    
    def get_fallback_classes(self):
        """Fallback mapping based on the data"""
        return {
            1: 'wall', 2: 'building, edifice', 3: 'sky', 4: 'floor, flooring',
            5: 'tree', 6: 'ceiling', 7: 'road, route', 8: 'bed',
            9: 'windowpane, window', 10: 'grass', 11: 'cabinet', 12: 'sidewalk, pavement',
            13: 'person', 14: 'earth, ground', 15: 'door, double door', 16: 'table',
            17: 'mountain, mount', 18: 'plant', 19: 'curtain', 20: 'chair',
            21: 'car', 22: 'water', 23: 'painting', 24: 'sofa', 25: 'shelf',
            26: 'house', 27: 'sea', 28: 'mirror', 29: 'rug, carpet', 30: 'field',
            31: 'armchair', 32: 'seat', 33: 'fence', 34: 'desk', 35: 'rock',
            36: 'wardrobe', 37: 'lamp', 38: 'bathtub', 39: 'railing', 40: 'cushion',
            41: 'base', 42: 'box', 43: 'column', 44: 'signboard', 45: 'chest of drawers',
            46: 'counter', 47: 'sand', 48: 'sink', 49: 'skyscraper', 50: 'fireplace',
            51: 'refrigerator', 52: 'grandstand', 53: 'path', 54: 'stairs, steps', 55: 'runway',
            56: 'case', 57: 'pool table', 58: 'pillow', 59: 'screen door', 60: 'stairway',
            61: 'river', 62: 'bridge', 63: 'bookcase', 64: 'blind', 65: 'coffee table',
            66: 'toilet', 67: 'flower', 68: 'book', 69: 'hill', 70: 'bench',
            71: 'countertop', 72: 'stove', 73: 'palm tree', 74: 'kitchen island', 75: 'computer',
            76: 'swivel chair', 77: 'boat', 78: 'bar', 79: 'arcade machine', 80: 'hovel',
            81: 'bus', 82: 'towel', 83: 'light', 84: 'truck', 85: 'tower',
            86: 'chandelier', 87: 'awning', 88: 'streetlight', 89: 'booth', 90: 'television',
            91: 'airplane', 92: 'dirt track', 93: 'apparel', 94: 'pole', 95: 'land',
            96: 'bannister', 97: 'escalator', 98: 'ottoman', 99: 'bottle', 100: 'buffet',
            101: 'poster', 102: 'stage', 103: 'van', 104: 'ship', 105: 'fountain',
            106: 'conveyer belt', 107: 'canopy', 108: 'washer', 109: 'toy', 110: 'swimming pool',
            111: 'stool', 112: 'barrel', 113: 'basket', 114: 'waterfall', 115: 'tent',
            116: 'bag', 117: 'minibike', 118: 'cradle', 119: 'oven', 120: 'ball',
            121: 'food', 122: 'step', 123: 'tank', 124: 'brand', 125: 'microwave',
            126: 'pot', 127: 'animal', 128: 'bicycle', 129: 'lake', 130: 'dishwasher',
            131: 'screen', 132: 'blanket', 133: 'sculpture', 134: 'hood', 135: 'sconce',
            136: 'vase', 137: 'traffic light', 138: 'tray', 139: 'trash can', 140: 'fan',
            141: 'pier', 142: 'crt screen', 143: 'plate', 144: 'monitor', 145: 'bulletin board',
            146: 'shower', 147: 'radiator', 148: 'glass', 149: 'clock', 150: 'flag'
        }
    
    def create_mapping(self):
        """
        Create mapping from ADE class IDs to 4 classes
        Returns: dict {ade_class_id: your_class_id}
        """
        mapping = {}
        
        # Default all classes to obstacle/wall (class 1)
        for ade_id in self.ade_class_names:
            mapping[ade_id] = 1  # Default to obstacle/wall
        
        print("\n" + "="*60)
        print("MAPPING ADE20K CLASSES TO NAVIGATION CLASSES")
        print("="*60)
        
        # Floor classes (map to 0)
        floor_keywords = ['floor', 'flooring', 'carpet', 'rug', 'ground', 'path', 'sidewalk', 'pavement', 'tile', 'earth']
        self._map_keywords(floor_keywords, 0, "FLOOR", mapping)
        
        # Door classes (map to 2)
        door_keywords = ['door', 'doorway', 'doorframe', 'gate', 'screen door', 'entrance', 'exit']
        self._map_keywords(door_keywords, 2, "DOOR", mapping)
        
        # No-go zones (map to 3)
        nogo_keywords = ['stairs', 'stair', 'staircase', 'stairway', 'escalator', 'steps', 
                        'river', 'lake', 'water', 'pool', 'ocean', 'sea', 'cliff', 'mountain',
                        'fireplace', 'waterfall', 'fountain', 'hill']
        self._map_keywords(nogo_keywords, 3, "NO-GO", mapping)
        
        # Count mappings
        counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for nav_class in mapping.values():
            counts[nav_class] += 1
            
        print("\n" + "="*60)
        print("MAPPING SUMMARY")
        print("="*60)
        print(f"Floor (class 0): {counts[0]} classes")
        print(f"Obstacle/Wall (class 1): {counts[1]} classes (default)")
        print(f"Door (class 2): {counts[2]} classes")
        print(f"No-go (class 3): {counts[3]} classes")
        
        return mapping
    
    def _map_keywords(self, keywords, target_class, class_name, mapping):
        """Helper function to map keywords to target class"""
        print(f"\n{class_name} (class {target_class}):")
        found_ids = []
        for ade_id, name in self.ade_class_names.items():
            if any(keyword in name for keyword in keywords):
                mapping[ade_id] = target_class
                print(f"  ID {ade_id:3d}: {name[:60]}")
                found_ids.append(ade_id)
        if not found_ids:
            print(f"  No matches found")
        return found_ids
    
    def save_mapping(self):
        """Save mapping to file"""
        base_path = getattr(Config, 'BASE_PATH', '/content/Indoor-Segmentation-Navigation')
        mapping_path = os.path.join(base_path, 'ade_to_nav_mapping.pkl')
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.mapping, f)
        print(f"\n✓ Mapping saved to '{mapping_path}'")

# Run if script is executed directly
if __name__ == "__main__":
    print("="*60)
    print("RUNNING CLASS MAPPER")
    print("="*60)
    
    mapper = ADEClassMapper()
    mapper.save_mapping()
    
    print("\n" + "="*60)
    print("✅ Class mapping complete!")
    print("   Mapping saved to: ade_to_nav_mapping.pkl")
    print("="*60)
