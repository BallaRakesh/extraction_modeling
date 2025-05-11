import os
import base64
import json
import requests
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    filename='image_processing_client.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
INPUT_FOLDER = "/datadrive/rakesh/data_temp/images"  # Folder containing input images
RESULTS_FOLDER = "/datadrive/rakesh/data_temp/results"     # Folder to save results
API_URL = "http://10.2.3.14:8000/process_image"  # API endpoint URL
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg")    # Supported image extensions

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        logger.info(f"Successfully encoded image to base64: {image_path}")
        return encoded_string
    except Exception as e:
        logger.error(f"Failed to encode image {image_path}: {e}")
        raise

def process_image(image_path: str, output_folder: str) -> None:
    """Process a single image by hitting the API and saving the result."""
    image_name = Path(image_path).stem  # Get filename without extension
    file_extension = Path(image_path).suffix.lower()[1:]  # Get extension without dot (e.g., "png")

    # Prepare payload
    try:
        b64_string = encode_image_to_base64(image_path)
        payload = {
            "image_name": image_name,
            "b64_encoded_image": b64_string,
            "fileType": file_extension
        }
    except Exception as e:
        logger.error(f"Skipping {image_name} due to encoding error: {e}")
        return

    # Send request to API
    logger.info(f"Sending request to API for image: {image_name}")
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes
        result = response.json()
        logger.info(f"Received successful response for {image_name}")
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {image_name}: {e}")
        return

    # Save result to file
    output_file = os.path.join(output_folder, f"{image_name}.json")
    try:
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        logger.info(f"Saved result to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save result for {image_name}: {e}")
    return result

def main():
    """Iterate over images in the input folder and process them."""
    # Ensure folders exist
    if not os.path.exists(INPUT_FOLDER):
        logger.error(f"Input folder '{INPUT_FOLDER}' does not exist")
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist")
        return
    
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    logger.info(f"Results folder ensured: {RESULTS_FOLDER}")

    # Iterate over files in input folder
    for filename in os.listdir(INPUT_FOLDER):
        filename = 'Claim Document_page-0007.jpg'
        image_path = os.path.join(INPUT_FOLDER, filename)
        if not os.path.isfile(image_path):
            continue
        
        if filename.lower().endswith(VALID_EXTENSIONS):
            logger.info(f"Processing image: {filename}")
            res = process_image(image_path, RESULTS_FOLDER)
            print(res)
            exit('DONE')
        else:
            logger.warning(f"Skipping {filename} - not a supported image type")

    logger.info("Processing completed for all images")

if __name__ == "__main__":
    logger.info("Starting image processing script")
    main()
    logger.info("Script execution finished")