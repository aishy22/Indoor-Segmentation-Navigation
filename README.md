# Indoor Navigation Segmentation

Semantic segmentation for indoor mobile robot navigation.

## Navigation Classes
- **0: floor** - Traversable (cost 1.0)
- **1: wall** - Blocked (cost inf)
- **2: door** - Preferred path (cost 0.8)
- **3: no-go** - Blocked (cost inf)

## Installation
```bash
pip install -r requirements.txt
```

## Download ADE20K dataset

```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
unzip ADEChallengeData2016.zip
```
## Run class_mapper
```bash
python src/class_mapper.py
```
