import torch
from pathlib import Path
from PIL import Image
import cv2

# Load YOLOv5 (loading the smallest model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load an image
img_path = 'stock_street_image.jpg'
img = Image.open(img_path)

# Run inference
results = model(img)

# Show results
results.print()               # Print label + confidence
results.show()                # Display image with bounding boxes
results.save()                # Save to output folder

# What's Happening:
# torch.hub.load() loads a YOLOv5 model
# results = model(img) runs object detection
# results.print() shows what was detected
# results.show() displays the image
# results.save() saves the result image with boxes