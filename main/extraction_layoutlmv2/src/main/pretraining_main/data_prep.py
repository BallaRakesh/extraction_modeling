from datasets import load_dataset  
import pyarrow as pa  
import pyarrow.csv as csv  # Optional, only if you need to export to CSV  
import pandas as pd  
import os
import json
from PIL import Image
import io
from datasets import load_from_disk

download_flag = False

if download_flag:
    # Load the dataset  
    ds = load_dataset("naver-clova-ix/cord-v2")  
    # Save the dataset locally  
    ds.save_to_disk("/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset")
    
if __name__ == "__main__":
    folder_name = 'validation'
    ds = load_from_disk("/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset")
    # Convert the dataset to a Pandas DataFrame
    df = ds[folder_name].to_pandas()  # Replace 'test' with the appropriate split (e.g., 'train', 'validation') if needed
    # Display the DataFrame
    print(df)
    # Create directories if they don't exist
    root_path = '/home/data_science/geo_testing/lmv2_code/local_cord_v2_dataset/validation'
    images_dir = os.path.join(root_path, "Images")
    labels_dir = os.path.join(root_path, "Labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Save the image
        image_data = row['image']['bytes']  # Extract image bytes
        image = Image.open(io.BytesIO(image_data))  # Decode the image bytes
        image_path = os.path.join(images_dir, f"image_{index}.png")
        print(image_path)
        image.save(image_path)  # Save the image as a PNG file

        # Save the label
        label_data = row['ground_truth']  # Extract ground truth (JSON string)
        label_path = os.path.join(labels_dir, f"label_{index}.json")
        with open(label_path, 'w') as label_file:
            json.dump(json.loads(label_data), label_file, indent=4)  # Save as a JSON file

    print(f"Images saved to: {images_dir}")
    print(f"Labels saved to: {labels_dir}")

