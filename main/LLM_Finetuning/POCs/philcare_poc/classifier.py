import os
import pdf2image
import pytesseract
from PyPDF2 import PdfReader
from typing import List, Dict, Optional
import os
import pickle
import cv2
import pytesseract
import numpy as np
import pandas as pd
from PIL import Image


# Define keywords for classification
keywords = {
    'invoice': ['invoice', "vat exempt"],
    'claim_form': ['patient', 'claim form'],
    'id_card': ['identification', "identification card", "regulation"],
    "license": ["license", "transportation"],
    'authorization_letter': ["authorization", "letter of authorization"]
}

def get_num_pages(pdf_path: str) -> int:
    """Get the number of pages in the PDF."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return 0

def classify_document(text: Optional[str], keywords: Dict[str, List[str]]) -> str:
    """Classify a document based on keyword presence in the text."""
    if not text:
        return "error"  # Return "error" if text is None
    
    text = text.lower()  # Case-insensitive matching
    scores = {doc_type: 0 for doc_type in keywords}
    
    # Count keyword matches for each document type
    for doc_type, kw_list in keywords.items():
        for kw in kw_list:
            if kw in text:
                scores[doc_type] += 1
    
    # Find the highest score
    max_score = max(scores.values())
    if max_score >= 1:  # Require at least two matches
        for doc_type, score in scores.items():
            if score == max_score:
                return doc_type
    return "unknown"  # Default if no type meets the threshold

def check_negative_value(word_coordinates):
    has_negative_value = any(
        value < 0 if isinstance(value, (int, float)) else False for value in word_coordinates[-1].values())
    if has_negative_value:
        word_coordinates.pop()



def tesseract_generate_ocr_string_and_word_coordinates(image, ocr_confidence_threshold: int = 0, language: str = "english"):
    language_mapper = {"english": "eng"}
    blacklisted_character = ""
    psm = "6"; oem = "3"
    language = language_mapper.get(language, "eng")

    image = image.convert('RGB')
    np_array = np.array(image)
    image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
    
    custom_config = f"--psm {psm} --oem {oem} -c tessedit_char_blacklist='{blacklisted_character}'"
    word_coordinates = []
    ocr_string = pytesseract.image_to_string(image, lang=language, config=custom_config)
    ocr_data = pytesseract.image_to_data(image, lang=language, config=custom_config, output_type=pytesseract.Output.DICT)
    ocr_confidence_based_string = ""
    
    for i, text in enumerate(ocr_data["text"]):
        if text != "":
            word_coordinates.append({
                "word": text,
                "left": ocr_data["left"][i],
                "top": ocr_data["top"][i],
                "width": ocr_data["width"][i],
                "height": ocr_data["height"][i],
                "x1": ocr_data["left"][i],
                "y1": ocr_data["top"][i],
                "x2": ocr_data["left"][i] + ocr_data["width"][i],
                "y2": ocr_data["top"][i] + ocr_data["height"][i],
                "bbox": [ocr_data["left"][i], ocr_data["top"][i], 
                         ocr_data["left"][i] + ocr_data["width"][i], ocr_data["top"][i] + ocr_data["height"][i]],
                "confidence": ocr_data["conf"][i]
            })
            if ocr_data["conf"][i] > ocr_confidence_threshold:
                ocr_confidence_based_string += text
            check_negative_value(word_coordinates)
    
    if ocr_confidence_threshold > 0:
        return ocr_string, word_coordinates, ocr_confidence_based_string
    else:
        return ocr_string, word_coordinates

def process_pdf_with_save(pdf_path: str, output_dir: str, keywords: Dict[str, List[str]]) -> List[Dict[str, Optional[str]]]:
    """
    Process a PDF: extract and save images and text if not already present, then classify each page.
    
    Args:
        pdf_path (str): Path to the input PDF file.
        output_dir (str): Directory to save images and text files.
        keywords (Dict[str, List[str]]): Keywords for document classification.
    
    Returns:
        List of dictionaries with page details and classification results.
    """
    num_pages = get_num_pages(pdf_path)
    if num_pages == 0:
        return []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for i in range(1, num_pages + 1):
        image_path = os.path.join(output_dir, f"page_{i}.png")
        text_path = os.path.join(output_dir, f"page_{i}.txt")
        
        # Check if both image and text files already exist
        if os.path.exists(image_path) and os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print(f"Using existing files for page {i}")
        else:
            # Extract image, save it, and run OCR if files are missing
            try:
                images = pdf2image.convert_from_path(pdf_path, first_page=i, last_page=i)
                image = images[0]
                image.save(image_path, 'PNG')
                # text, _= tesseract_generate_ocr_string_and_word_coordinates(image)
                text = pytesseract.image_to_string(image)
                with open(text_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                print(f"Extracted and saved page {i}")
            except Exception as e:
                print(f"Error processing page {i}: {e}")
                text = None
        
        # Classify the document using the text
        doc_type = classify_document(text, keywords)
        
        # Collect results
        results.append({
            'page': i,
            'image_path': image_path if os.path.exists(image_path) else None,
            'text_path': text_path if os.path.exists(text_path) else None,
            'document_type': doc_type
        })
    
    return results

# Example usage
if __name__ == "__main__":
    pdf_path = '/home/ntlpt-42/Documents/mani_projects/mani_POCS/philcare/data/Claim Document.pdf'  # Replace with your PDF file path
    output_dir = '/home/ntlpt-42/Documents/mani_projects/mani_POCS/philcare/data/results'  # Replace with your desired output directory
    
    results = process_pdf_with_save(pdf_path, output_dir, keywords)
    
    # Print classification results
    for entry in results:
        print(f"Page {entry['page']}: {entry['document_type']}")
        # if entry['image_path']:
        #     # print(f"  Image: {entry['image_path']}")
        # if entry['text_path']:
        #     print(f"  Text: {entry['text_path']}")