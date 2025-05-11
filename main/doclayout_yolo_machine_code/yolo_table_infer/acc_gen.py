import os
import cv2
import numpy as np
import torch
import glob
from pathlib import Path
import json
from ultralytics import YOLO  # Import YOLO
from torchmetrics.detection.mean_ap import MeanAveragePrecision

# Define Paths
BASE_DIR = "/home/ng6309/datascience/anand/Stamps-Signature.v2i.yolov8/test"
IMAGES_DIR = "non_layout_data/input_data/tf_data/Images"
LABELS_DIR = "/home/data_science/project_files/Genai_test/GenAI-POC/new_training/test_labels"
PREDICTIONS_PATH= "/home/ng6309/datascience/anand/non_layout_data/output_data/tf_data/tf_data_result_json"


OUTPUT_DIR = "image_annotation_gp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load YOLO Model
# model_file = "/home/ng6309/datascience/anand/Invoice_Extraction_Solution/models_infer_code/yolo/runs/detect/train9/weights/best.pt"
# device = "cpu"
# model = YOLO(model_file, task="detect").to(device)

# Initialize TorchMetrics mAP calculator
metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)
def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


# Function to Read Annotations
def read_annotations(label_path, img_w, img_h):
    """ Reads YOLO format annotation files and returns bbox in (class, x_min, y_min, x_max, y_max) """
    gt_boxes = []
    with open(label_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            values = list(map(float, line.strip().split()))
            class_id = int(values[0])  # First value is class
            x_center, y_center, width, height = values[1:5]

            # Convert YOLO format to absolute pixel values
            x_min = (x_center - width / 2) * img_w
            y_min = (y_center - height / 2) * img_h
            x_max = (x_center + width / 2) * img_w
            y_max = (y_center + height / 2) * img_h
            gt_boxes.append([class_id, x_min, y_min, x_max, y_max])
    
    return np.array(gt_boxes)

# Process Images
results_data = []
for image_path in os.listdir(IMAGES_DIR):
    image_name = Path(image_path).stem
    label_path = os.path.join(LABELS_DIR, image_name + ".txt")
    predictions_path = os.path.join(PREDICTIONS_PATH,image_name + ".json")

    if not os.path.exists(label_path):
        print(f"Skipping {image_name}: No annotation file found.")
        continue

    # Read Image
    image = cv2.imread(os.path.join(IMAGES_DIR,image_path))
    img_h, img_w, _ = image.shape

    # Read Ground Truth Annotations
    gt_boxes = read_annotations(label_path, img_w, img_h)
    if len(gt_boxes)<1:
        continue
    
    # Extract ground truth labels and boxes
    gt_labels = gt_boxes[:, 0].astype(int).tolist()
    gt_boxes_list = gt_boxes[:, 1:].tolist()

    with open(predictions_path,"r") as f:
        results = json.load(f)
    pred_boxes = []
    pred_scores = []
    pred_labels = []

    for box in results["prediction"]:  # Convert to numpy
        x_min, y_min, x_max, y_max, conf, class_id = box["bbox"]
        pred_boxes.append([x_min, y_min, x_max, y_max])
        pred_scores.append(conf)
        pred_labels.append(int(class_id))

    pred_boxes = torch.tensor(pred_boxes)
    pred_scores = torch.tensor(pred_scores)
    pred_labels = torch.tensor(pred_labels)

    
    gt_boxes_tensor = torch.tensor(gt_boxes[:, 1:], dtype=torch.float32)
    gt_labels_tensor = torch.tensor(gt_boxes[:, 0], dtype=torch.int64)

    # Convert data to TorchMetrics format
    preds = [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}]
    targets = [{"boxes": gt_boxes_tensor, "labels": gt_labels_tensor}]
    
    print(preds)
    # Add to mAP Calculator
    metric.update(preds, targets)

    # Draw Predictions and Ground Truth
    for box in gt_boxes:
        x1, y1, x2, y2 = map(int, box[1:])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for GT
        cv2.putText(image, f"GT-{int(box[0])}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for i, box in enumerate(pred_boxes):
        x1, y1, x2, y2 = map(int, box.numpy())
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for Prediction
        cv2.putText(image, f"Pred-{pred_labels[i]} ({pred_scores[i]:.2f})", 
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save Annotated Image
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{image_name}.jpg"), image)


    # Calculate per-box correctness and scores
    per_box_results = []
    for i, pred_box in enumerate(pred_boxes):
        max_iou = 0
        matched_label = None
        for j, gt_box in enumerate(gt_boxes_tensor):
            iou = compute_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
                matched_label = gt_labels[j]
        
        # Store prediction details
        per_box_results.append({
            "label": pred_labels[i].item(),
            "score": pred_scores[i].item(),
            "correct": max_iou >= 0.5 and pred_labels[i] == matched_label,
            "iou": max_iou
        })

    # Store results for this image
    results_data.append({
        "image_name": image_name,
        "ground_truth_labels": gt_labels,
        "ground_truth_boxes": gt_boxes_list,
        "predicted_labels": pred_labels.tolist(),
        "predicted_boxes": pred_boxes.tolist(),
        "prediction_results": per_box_results
    })

# Create CSV file
csv_data = []
for result in results_data:
    csv_data.append({
        "Image Name": result["image_name"],
        "Ground Truth Labels": str(result["ground_truth_labels"]),
        "Ground Truth Boxes": str(result["ground_truth_boxes"]),
        "Predicted Labels": str(result["predicted_labels"]),
        "Predicted Boxes": str(result["predicted_boxes"]),
        "Prediction Results": str(result["prediction_results"])
    })
    
import pandas as pd
df = pd.DataFrame(csv_data)
df.to_csv(os.path.join(OUTPUT_DIR, "image_wise_results.csv"), index=False)
print(f"CSV file saved to {os.path.join(OUTPUT_DIR, 'image_wise_results.csv')}")


# Compute and Print Final mAP Results
final_results = metric.compute()
print(f"\n🎯 mAP@0.5:0.95: {final_results['map'].item():.4f}")
print(f"🎯 mAP@0.5: {final_results['map_50'].item():.4f}")