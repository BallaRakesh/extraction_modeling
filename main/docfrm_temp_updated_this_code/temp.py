from datasets import Dataset
import torch
import torch
import os
import ast
from PIL import Image
import pytesseract


def get_ocr_tesseract(image_):
    """
    Performs OCR (Optical Character Recognition) using Tesseract OCR engine.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing word coordinates (list of dictionaries) and all the extracted text (str).

    """
    # img=None
    word_coordinates, all_text = [],""
    print("called Image OCR...", end="")
    # try:
    # img = Image.open(image_path)
    d = pytesseract.image_to_data(image_, output_type=pytesseract.Output.DICT)
    all_text = pytesseract.image_to_string(image_)
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
    # except Exception as e:
    # 	print(f"exception: {e}")	
    # finally:
    # 	if hasattr(img,"close"):
    # 		img.close()
    return word_coordinates, all_text

def convert_image_mode(img: Image.Image, target_mode: str = 'L') -> Image.Image:
    """
    Convert image to the specified mode, preserving as much original information as possible.
    
    Args:
        img (PIL.Image.Image): Input image
        target_mode (str): Desired image mode (default 'L' for grayscale)
    
    Returns:
        PIL.Image.Image: Converted image
    """
    # Map of conversion strategies
    conversion_strategies = {
        '1': lambda x: x.convert('L'),    # 1-bit pixels (black and white)
        'L': lambda x: x,                 # Grayscale 
        'P': lambda x: x.convert('L'),    # Palette-mapped 
        'RGB': lambda x: x.convert('L'),  # Color to grayscale
        'RGBA': lambda x: x.convert('L'), # Color with alpha to grayscale
    }
    
    # Get the current mode
    current_mode = img.mode
    
    # Choose conversion strategy
    if current_mode in conversion_strategies:
        return conversion_strategies[current_mode](img)
    
    # Fallback to direct conversion
    return img.convert(target_mode)


img_path = '/home/data_science/geo_testing/Classification_root/BILLS/AWB/Bill_Of_Landing_183_page_0.png'
ocr_path = '/home/data_science/geo_testing/Classification_root/OCR_GV/Bill_Of_Landing_562_page_3_text.txt'
with open(ocr_path, 'r') as file:
    content = file.read()
    wc_data = ast.literal_eval(content)
file.close()
word_coordinates = wc_data.get('word_coordinates', [])
print(word_coordinates)
exit('OKOKOK')

img = Image.open(img_path)
img = convert_image_mode(img, target_mode='L')  # Ensure grayscale mode
word_coordinates, all_text = get_ocr_tesseract(img)
print(word_coordinates)
exit('OKO')


# Example dataset that produces scalar values
data = {"value": [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]}
hf_dataset = Dataset.from_dict(data)

# final_encoding = {'label': tensor(0)}
final_encoding = {'label': torch.tensor(0)}

print('label>>>>>>>>>>> before.', final_encoding['label'])

final_encoding['label'] = final_encoding['label'].unsqueeze(0).long()
print('label>>>>>>>>>>> after.', final_encoding['label'])




# This may cause an issue in certain cases
for item in hf_dataset:
    print(item["value"][0])  # IndexError if value is 0-dim tensor
