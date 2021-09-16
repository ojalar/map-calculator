# Dependencies
1. numpy

# File Format 
Detections and ground truth labels are given in identically named .txt files in separate folders. Bounding box coordinates are given as pixel coordinates for upper left corner (x1, y1) and lower right corner (x2, y2).

## Detections  
```
<class> <confidence> <x1> <y1> <x2> <y2> 
```
## Ground Truths  
```
<class> <x1> <y1> <x2> <y2> 
```
# Usage 
```
python3 map_calculator.py -d <detection folder> -t <ground truth folder> -i <IoU-threshold> -m <mode>
```
# Modes
Available modes are "normal", "multi" and "cluster". The "normal" and "cluster" modes are utilised in the related paper.
