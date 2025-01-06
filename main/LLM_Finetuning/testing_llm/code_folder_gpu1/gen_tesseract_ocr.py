import pytesseract
from PIL import Image, ImageDraw
from fastapi import FastAPI
import base64
import requests
from pydantic import BaseModel
from io import BytesIO
import os
from PIL import Image
import base64
from io import BytesIO


def get_ocr_tesseract(image):
    """
    Extracts text and word coordinates from an image using the Tesseract OCR engine.

    Args:
        image_path (str): The file path of the image to be processed.

    Returns:
        tuple: A tuple containing two elements:
            - word_coordinates (list): A list of dictionaries, where each dictionary represents a word in the image and contains the following keys:
                - "word": The text of the word.
                - "confidence": The confidence score of the OCR result for the word.
                - "left": The x-coordinate of the left edge of the word.
                - "top": The y-coordinate of the top edge of the word.
                - "width": The width of the word.
                - "height": The height of the word.
                - "x1": The x-coordinate of the left edge of the word.
                - "y1": The y-coordinate of the top edge of the word.
                - "x2": The x-coordinate of the right edge of the word.
                - "y2": The y-coordinate of the bottom edge of the word.
            - all_text (str): The full text extracted from the image.
    """

    word_coordinates = []
    all_text = ""
    print("called Image OCR...", end="")
    try:
        if image:
            d = pytesseract.image_to_data(image, lang="ind", config="--psm 6", output_type=pytesseract.Output.DICT)
            all_text = pytesseract.image_to_string(image, lang="ind", config="--psm 6")
            word_coordinates = []
            for i in range(len(d['text'])):
                word = d['text'][i]
                conf = float(d['conf'][i])
                if conf > 0:
                    x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
                    word_coordinates.append({
                        "word": word,
                        "confidence": conf,
                        "left": x,
                        "top": y,
                        "width": w,
                        "height": h,
                        "x1": x,
                        "y1": y,
                        "x2": x + w,
                        "y2": y + h
                    })
    except Exception as e:
        print(f"exception : {e}")

    return word_coordinates, all_text


def image_to_base64(image_path):
    # Open the image file and convert it to base64
    with Image.open(image_path) as image:
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return image_base64




def process_images_in_folder(image_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Iterate over all images in the image_folder
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            # Construct full file path
            image_path = os.path.join(image_folder, filename)
            image_path = '/home/gpu1admin/rakesh/temp_data/all_sample/7.png'
            # Convert image to base64 and decode back to image data
            base64_image = image_to_base64(image_path)
            image_data = base64.b64decode(base64_image)
            image = Image.open(BytesIO(image_data))
            
            # Run OCR processing
            word_coordinates, all_text = get_ocr_tesseract(image)
            print(all_text)
            exit('OLLLLLLL')
            # Construct output filenames with prefixes
            base_name = os.path.splitext(filename)[0]
            text_and_coordinates_path = os.path.join(output_folder, f"{base_name}_textAndCoordinates.txt")
            all_text_path = os.path.join(output_folder, f"{base_name}_all_text.txt")
            
            # Save word coordinates and all text to files
            with open(text_and_coordinates_path, 'w') as file:
                file.write(str(word_coordinates))
            
            with open(all_text_path, 'w') as file:
                file.write(all_text)
            
            print(f"Processed and saved: {filename}")


if __name__ == "__main__":
    # Define image and output folders
    image_folder = '/home/gpu1admin/rakesh/temp_data/all_sample'
    output_folder = '/home/gpu1admin/rakesh/temp_data/ocr'
    os.makedirs(output_folder, exist_ok = True)
    # Run the processing
    process_images_in_folder(image_folder, output_folder)

