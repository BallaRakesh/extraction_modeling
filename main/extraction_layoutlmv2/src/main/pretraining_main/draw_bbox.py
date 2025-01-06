import cv2
import numpy as np
import json

def draw_annotations(image_path, json_path, output_path):
    # Load the JSON file
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    roi_data = {'roi' :json_data['roi'], 'repeating_symbol':json_data['repeating_symbol'], 'dontcare':json_data['dontcare']}
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found!")
        return

    # Extract image dimensions from JSON metadata
    image_width = json_data["meta"]["image_size"]["width"]
    image_height = json_data["meta"]["image_size"]["height"]

    # Resize image if necessary
    img = cv2.resize(img, (image_width, image_height))

    # Draw ROI
    roi = roi_data["roi"]
    roi_points = np.array([
        [roi["x1"], roi["y1"]], [roi["x2"], roi["y2"]],
        [roi["x3"], roi["y3"]], [roi["x4"], roi["y4"]]
    ], np.int32)
    cv2.polylines(img, [roi_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # Draw repeating symbols
    for symbol_list in roi_data["repeating_symbol"]:
        for symbol in symbol_list:
            quad = symbol["quad"]
            quad_points = np.array([
                [quad["x1"], quad["y1"]], [quad["x2"], quad["y2"]],
                [quad["x3"], quad["y3"]], [quad["x4"], quad["y4"]]
            ], np.int32)
            cv2.polylines(img, [quad_points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw dontcare regions
    for dontcare_list in roi_data["dontcare"]:
        for dontcare in dontcare_list:
            dontcare_points = np.array([
                [dontcare["x1"], dontcare["y1"]], [dontcare["x2"], dontcare["y2"]],
                [dontcare["x3"], dontcare["y3"]], [dontcare["x4"], dontcare["y4"]]
            ], np.int32)
            cv2.polylines(img, [dontcare_points], isClosed=True, color=(255, 0, 0), thickness=2)

    # Parse the bounding boxes from JSON data
    valid_lines = json_data["valid_line"]
    for line in valid_lines:
        words = line["words"]
        category = line.get("category", "unknown")
        for word in words:
            quad = word["quad"]

            # Extract quad coordinates
            x1, y1 = quad["x1"], quad["y1"]
            x2, y2 = quad["x2"], quad["y2"]
            x3, y3 = quad["x3"], quad["y3"]
            x4, y4 = quad["x4"], quad["y4"]

            # Create the bounding box (polygon)
            pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], np.int32)
            pts = pts.reshape((-1, 1, 2))

            # Draw the bounding box
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Add category text near the box
            cv2.putText(img, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save the output image
    cv2.imwrite(output_path, img)
    print(f"Annotated image saved to {output_path}")

input_image_path = "/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset/test/Images/image_0.png"
json_path = "/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset/test/Labels/label_0.json"
output_image_path = "output_image_with_annotations.jpg"

draw_annotations(input_image_path, json_path, output_image_path)
