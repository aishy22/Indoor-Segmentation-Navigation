# src/class_mapper.py
import numpy as np
import os
import pickle
from config import Config

class ADEClassMapper:
    """
    Maps ADE20K's 150 classes to your 4 navigation classes:
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
        object_info_path = os.path.join(Config.DATA_PATH, 'objectInfo150.txt')
        
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
            # Fallback to hardcoded common classes based on your output
            class_names = self.get_fallback_classes()
            
        return class_names
    
    def get_fallback_classes(self):
        """Fallback mapping based on the data you showed"""
        return {
            1: 'wall',
            2: 'building, edifice',
            3: 'sky',
            4: 'floor, flooring',
            5: 'tree',
            6: 'ceiling',
            7: 'road, route',
            8: 'bed',
            9: 'windowpane, window',
            10: 'grass',
            11: 'cabinet',
            12: 'sidewalk, pavement',
            13: 'person, individual, someone, somebody, mortal, soul',
            14: 'earth, ground',
            15: 'door, double door',
            16: 'table',
            17: 'mountain, mount',
            18: 'plant, flora, plant life',
            19: 'curtain, drape, drapery, mantle, pall',
            20: 'chair',
            21: 'car, auto, automobile, machine, motorcar',
            22: 'water',
            23: 'painting, picture',
            24: 'sofa, couch, lounge',
            25: 'shelf',
            26: 'house',
            27: 'sea',
            28: 'mirror',
            29: 'rug, carpet, carpeting',
            30: 'field',
            31: 'armchair',
            32: 'seat',
            33: 'fence, fencing',
            34: 'desk',
            35: 'rock, stone',
            36: 'wardrobe, closet, press',
            37: 'lamp',
            38: 'bathtub, bathing tub, bath, tub',
            39: 'railing, rail',
            40: 'cushion',
            41: 'base, pedestal, stand',
            42: 'box',
            43: 'column, pillar',
            44: 'signboard, sign',
            45: 'chest of drawers, chest, bureau, dresser',
            46: 'counter',
            47: 'sand',
            48: 'sink',
            49: 'skyscraper',
            50: 'fireplace, hearth, open fireplace',
            51: 'refrigerator, icebox',
            52: 'grandstand, covered stand',
            53: 'path',
            54: 'stairs, steps',
            55: 'runway',
            56: 'case, display case, showcase, vitrine',
            57: 'pool table, billiard table, snooker table',
            58: 'pillow',
            59: 'screen door, screen',
            60: 'stairway, staircase',
            61: 'river',
            62: 'bridge, span',
            63: 'bookcase',
            64: 'blind, screen',
            65: 'coffee table, cocktail table',
            66: 'toilet, can, commode, crapper, pot, potty, stool, throne',
            67: 'flower',
            68: 'book',
            69: 'hill',
            70: 'bench',
            71: 'countertop',
            72: 'stove, kitchen stove, range, kitchen range, cooking stove',
            73: 'palm, palm tree',
            74: 'kitchen island',
            75: 'computer, computing machine, computing device, data processor, electronic computer, information processing system',
            76: 'swivel chair',
            77: 'boat',
            78: 'bar',
            79: 'arcade machine',
            80: 'hovel, hut, hutch, shack, shanty',
            81: 'bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle',
            82: 'towel',
            83: 'light, light source',
            84: 'truck, motortruck',
            85: 'tower',
            86: 'chandelier, pendant, pendent',
            87: 'awning, sunshade, sunblind',
            88: 'streetlight, street lamp',
            89: 'booth, cubicle, stall, kiosk',
            90: 'television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box',
            91: 'airplane, aeroplane, plane',
            92: 'dirt track',
            93: 'apparel, wearing apparel, dress, clothes',
            94: 'pole',
            95: 'land, ground, soil',
            96: 'bannister, banister, balustrade, balusters, handrail',
            97: 'escalator, moving staircase, moving stairway',
            98: 'ottoman, pouf, pouffe, puff, hassock',
            99: 'bottle',
            100: 'buffet, counter, sideboard',
            101: 'poster, posting, placard, notice, bill, card',
            102: 'stage',
            103: 'van',
            104: 'ship',
            105: 'fountain',
            106: 'conveyer belt, conveyor belt, conveyer, conveyor, transporter',
            107: 'canopy',
            108: 'washer, automatic washer, washing machine',
            109: 'plaything, toy',
            110: 'swimming pool, swimming bath, natatorium',
            111: 'stool',
            112: 'barrel, cask',
            113: 'basket, handbasket',
            114: 'waterfall, falls',
            115: 'tent, collapsible shelter',
            116: 'bag',
            117: 'minibike, motorbike',
            118: 'cradle',
            119: 'oven',
            120: 'ball',
            121: 'food, solid food',
            122: 'step, stair',
            123: 'tank, storage tank',
            124: 'trade name, brand name, brand, marque',
            125: 'microwave, microwave oven',
            126: 'pot, flowerpot',
            127: 'animal, animate being, beast, brute, creature, fauna',
            128: 'bicycle, bike, wheel, cycle',
            129: 'lake',
            130: 'dishwasher, dish washer, dishwashing machine',
            131: 'screen, silver screen, projection screen',
            132: 'blanket, cover',
            133: 'sculpture',
            134: 'hood, exhaust hood',
            135: 'sconce',
            136: 'vase',
            137: 'traffic light, traffic signal, stoplight',
            138: 'tray',
            139: 'ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin',
            140: 'fan',
            141: 'pier, wharf, wharfage, dock',
            142: 'crt screen',
            143: 'plate',
            144: 'monitor, monitoring device',
            145: 'bulletin board, notice board',
            146: 'shower',
            147: 'radiator',
            148: 'glass, drinking glass',
            149: 'clock',
            150: 'flag'
        }
    
    def create_mapping(self):
        """
        Create mapping from ADE class IDs to your 4 classes
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
        floor_ids = self._map_keywords(floor_keywords, 0, "FLOOR", mapping)
        
        # Door classes (map to 2)
        door_keywords = ['door', 'doorway', 'doorframe', 'gate', 'screen door', 'entrance', 'exit', 'portal']
        door_ids = self._map_keywords(door_keywords, 2, "DOOR", mapping)
        
        # No-go zones (map to 3) - stairs, hazardous areas
        nogo_keywords = ['stairs', 'stair', 'staircase', 'stairway', 'escalator', 'steps', 
                        'fragile', 'restricted', 'danger', 'hole', 'pit', 'elevator shaft',
                        'river', 'lake', 'water', 'pool', 'ocean', 'sea', 'cliff', 'mountain',
                        'fireplace', 'furnace', 'oven', 'stove', 'waterfall', 'fountain',
                        'cliff', 'hill', 'mountain']
        nogo_ids = self._map_keywords(nogo_keywords, 3, "NO-GO", mapping)
        
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
            # Check if any keyword is in the name
            if any(keyword in name for keyword in keywords):
                mapping[ade_id] = target_class
                print(f"  ID {ade_id:3d}: {name[:60]}")
                found_ids.append(ade_id)
        if not found_ids:
            print(f"  No matches found")
        return found_ids
    
    def print_mapping_details(self):
        """Print detailed mapping for verification"""
        print("\n" + "="*60)
        print("DETAILED MAPPING BY CLASS")
        print("="*60)
        
        # Group by navigation class
        nav_classes = {0: [], 1: [], 2: [], 3: []}
        
        for ade_id, nav_class in self.mapping.items():
            nav_classes[nav_class].append((ade_id, self.ade_class_names[ade_id]))
        
        # Print each navigation class
        for nav_class, class_name in enumerate(Config.CLASS_NAMES):
            print(f"\n{class_name.upper()} (class {nav_class}):")
            items = nav_classes[nav_class]
            if items:
                # Sort by ADE ID
                items.sort()
                for ade_id, name in items[:15]:  # Show first 15
                    print(f"  ID {ade_id:3d}: {name[:60]}")
                if len(items) > 15:
                    print(f"  ... and {len(items)-15} more")
            else:
                print("  No classes mapped")
    
    def save_mapping(self):
        """Save mapping to file"""
        mapping_path = os.path.join(Config.BASE_PATH, 'ade_to_nav_mapping.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.mapping, f)
        print(f"\n✓ Mapping saved to '{mapping_path}'")
        
        # Also save as text for readability
        txt_path = os.path.join(Config.OUTPUT_PATH, 'class_mapping.txt')
        with open(txt_path, 'w') as f:
            f.write("ADE20K to Navigation Classes Mapping\n")
            f.write("="*60 + "\n\n")
            for ade_id, nav_class in sorted(self.mapping.items()):
                name = self.ade_class_names.get(ade_id, "unknown")
                f.write(f"ADE ID {ade_id:3d} ({name:40s}) -> {Config.CLASS_NAMES[nav_class]} (class {nav_class})\n")
        print(f"✓ Text mapping saved to '{txt_path}'")

    def create_custom_mapping(self):
        """Create a custom mapping based on manual inspection"""
        # This is a more accurate mapping based on your actual classes
        custom_map = {}
        
        # Set default for all to obstacle/wall
        for i in range(1, 151):
            custom_map[i] = 1
        
        # Floor (class 0)
        floor_ids = [4, 12, 14, 29, 53, 95]  # floor, sidewalk, earth, rug, path, land
        for idx in floor_ids:
            custom_map[idx] = 0
            
        # Door (class 2)
        door_ids = [15, 59]  # door, screen door
        for idx in door_ids:
            custom_map[idx] = 2
            
        # No-go (class 3)
        nogo_ids = [17, 22, 27, 50, 54, 55, 60, 61, 69, 72, 97, 105, 110, 114, 119, 122, 125, 129]  # stairs, water, mountain, etc.
        for idx in nogo_ids:
            custom_map[idx] = 3
            
        return custom_map

# Run if script is executed directly
if __name__ == "__main__":
    mapper = ADEClassMapper()
    
    # Option 1: Use keyword-based mapping
    mapper.print_mapping_details()
    mapper.save_mapping()
    
    # Option 2: Uncomment below to use custom mapping instead
    # mapper.mapping = mapper.create_custom_mapping()
    # mapper.print_mapping_details()
    # mapper.save_mapping()
    
    print("\n" + "="*60)
    print("Next step: Run 'python src/data_prep.py' to test the dataset with this mapping")
    print("="*60)
