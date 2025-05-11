import cv2
import pandas as pd

# Function to draw bounding boxes and labels
def draw_bboxes(image_path, tsv_path, output_path):
    # Read the image
    image = cv2.imread(image_path)
    
    # Read the TSV file with error handling for inconsistent lines
    data = pd.read_csv(tsv_path, sep=',', header=None, on_bad_lines='skip')

    # Iterate through each row in the TSV
    for index, row in data.iterrows():
        try:
            # Extract bounding box coordinates and label
            x1, y1, x2, y2, x3, y3, x4, y4, label, tag = int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5]), int(row[6]), int(row[7]), int(row[8]), row[9], row[10]
            
            # Draw the bounding box (use (x1, y1) and (x3, y3) as opposite corners)
            cv2.rectangle(image, (x1, y1), (x3, y3), (0, 255, 0), 2)
            
            # Put the label text near the bounding box
            cv2.putText(image, tag, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error processing row {index}: {e}")

    # Save the output image
    cv2.imwrite(output_path, image)

# Example usage
tsv_path = '/home/ntlpt19/Desktop/TF_release/layoutlmv1/main/PICK-pytorch/data/data_examples_root/boxes_and_transcripts/asdf.tsv'  # Path to your TSV file
image_path = '/home/ntlpt19/Desktop/TF_release/layoutlmv1/main/PICK-pytorch/data/data_examples_root/images/asdf.jpg'  # Path to the image

output_path = 'output_image.jpg'

draw_bboxes(image_path, tsv_path, output_path)
