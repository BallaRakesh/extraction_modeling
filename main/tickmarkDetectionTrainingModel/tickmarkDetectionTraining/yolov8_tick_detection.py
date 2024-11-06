import json
from typing import List

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Load the model
model = YOLO('/home/nt-user1/vasu/yolov8/yolo_train/yolov8n.pt')
print('Loaded Model Successfully !!!')

# Read the image
img = cv2.imread('/home/nt-user1/vasu/yolov8/yolo_prepare_data/images/train/5.png')

# Run inference
results = model(img)

# Get the first result (assuming single image input)
result: Results = results[0]

# Create a copy of the original image to draw on
output_image = img.copy()

# Get class names
class_names = model.names

detected_boxes: List = []

for box in result.boxes:
    class_id = int(box.cls)
    class_name = class_names[class_id]

    # Get box coordinates
    x1, y1, x2, y2 = box.xyxy[0].tolist()

    x1,y1,x2,y2 = map(int,[x1,y1,x2,y2])

    detected_boxes.append({
        'name': class_name,
        'bbox': [x1,y1,x2,y2]
    })

    # Draw the box
    cv2.rectangle(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Draw the label
    cv2.putText(output_image, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

with open('/home/nt-user1/vasu/yolov8/result.json', 'w') as f:
    json.dump(detected_boxes,f,indent = 2)

cv2.imwrite('/home/nt-user1/vasu/yolov8/image.png', output_image)