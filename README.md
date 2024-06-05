# SportCoachAI_fer
> create a `weights` folder to place weights
## 🚀 What's new
### initial commit
- VGG19 model trained with fer2013
- Detecting face with cv2.cascadeclassifier
### 2024.6.5
- deepsort is available for multi-target tracking
- yolo detector is available for better performance
- execute `python yolo-detect.py` to use these features.
- Due to large sizes, weight files are not uploaded.

# Dependency
- torch
- numpy
- opencv
- ultralytics

# Usage
## cv2.cascadeclassifier & fer
use cv2.cascadeclassifier to detect faces. Use VGG19 network to predict facial expression
```plain text
python detect.py
```

## yolov8 & fer
use yolov8 to detect faces. Use VGG19 network to predict facial expression
```plain text
python yolo-detect.py --use_tracker 0
```

## yolov8 & deepsort & fer
add deepsort for multiple target response
```plain text
python yolo-detect.py
```
