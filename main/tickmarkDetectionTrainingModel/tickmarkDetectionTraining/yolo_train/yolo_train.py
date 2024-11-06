from ultralytics import YOLO

# Load the YOLO model
model = YOLO('yolov8n.pt')  # or any other YOLOv8 model variant

# Train the model
results = model.train(data='config.yaml', epochs=5, imgsz=640)

# Save the trained model
model.save('yolov8n_tick.pt')
