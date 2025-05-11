import os
import pickle
import cv2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
# from your_ocr_module import get_ocr_vision_api  # Ensure this function is available
import os 
from datetime import datetime
from google.cloud import vision
from base64 import b64encode
from google.oauth2.service_account import Credentials
import pytesseract
import os
import os
import json
from pdf2image import convert_from_path
from pymupdf import open as open_pdf
import os
import json
import fitz  # PyMuPDF
import os
import json
import fitz  # PyMuPDF
from PIL import Image 
import pdfplumber
import os
import json
from pdf2image import convert_from_path




# CLASS_MAPPING = {"BEA": 0, "CCNA": 1, "MDS": 2}
CLASS_MAPPING= {
    "BEA": 0,
    "CCNA": 1,
    "Cognitive Questionnaire MD": 2,
    "MDS": 3,
    "Medical Records": 4,
    "Medication Administration Records (MAR)": 5,
    "Medication Administration Screen Form": 6,
    "Medication List": 7,
    "Nursing Assessment": 8,
    "Plan of Care (POC)": 9,
    "RFR Facility": 10,
    "RFR General": 11,
    "RFR HHC": 12
}
def get_ocr_tesseract(image_path):
    """
    Performs OCR (Optical Character Recognition) using Tesseract OCR engine.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing word coordinates (list of dictionaries) and all the extracted text (str).

    """
    img=None
    word_coordinates, all_text = [],""
    print("called Image OCR...", end="")
    try:
        img = Image.open(image_path)
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        all_text = pytesseract.image_to_string(img)
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
        print(f"exception: {e}")	
    finally:
        if hasattr(img,"close"):
            img.close()
    return word_coordinates, all_text

def get_ocr_vision_api(ocr_coord_file_name: str = None, ocr_all_text_file_name: str = None, image_path: str = "./", image_name: str = None, ocr_credential_file = "./google_vision_key.json"):
    """
    Function to get word and their co-ordinates information and the complete OCR text from an image using Google Vision OCR.
    
    (Parameters)
    ocr_path: str               = Folder path where the ocr result will be saved
    ocr_coord_file_name: str    = OCR Coordinate response json file name 
    ocr_all_text_file_name: str = OCR Complete text response file name
    image_path: str             = Folder path where the image is saved
    image_name: str             = Image name 
    ocr_credential_file         = OCR Creadential Json file complete path
    
    (Response)
    save_coords: dict           = word and their co-ordinates
    all_text: str               = ocr complete text string
    """
    all_text = ''
    save_coords = ''
    ocr_path = OCR_PATH
    ocr_coord_file_name = image_name.rsplit(".",1)[0] + "_textAndCoordinates.txt"
    ocr_text_file_name = image_name.rsplit(".",1)[0] + "_text.txt"
    if os.path.exists(os.path.join(ocr_path, ocr_text_file_name)):
        with open(os.path.join(ocr_path, ocr_text_file_name), 'r') as f:
            all_text = f.read()
        f.close()
        save_coords = ''
        file_exists = True
    else:
        if not use_gv_flag:
            save_coords, all_text = get_ocr_tesseract(os.path.join(image_path, image_name))
        else:
            image = open(os.path.join(image_path, image_name), 'rb')
            ctxt = b64encode(image.read()).decode()
            ocr_credential_dict = {
                "type": "service_account",
                "project_id": "imagedocumentprocessing",
                "private_key_id": "598eab6cfc358ef3fac5f1a0a1b178dc10706d5b",
                "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDGzlwHe9Qrf89O\nAEcMOKb8SNeEHK0YUES3Yy/5zWBPj+ecR+NKMsP7ay1KMMUjKFVLbixrGr8KJue7\nwx7lFmklYmeIJlaiI/PBgNG3ky8Yth14OHB20bI9E77XRIfHpT9p+WyS4t0WkYxq\nWFLhwKGViPArFVHt2HMvHTHWGM1wS+FOWERZ/1Z01bQstUuZRqKyQCIbukUfEoDM\nKOiaffICsOyzdxhXiqBb+qLUvZKWSAcsUy18OMf+NqcBrt1IN4kID95UVRANKAy7\naX+2mh9RQ9gi2YhMdax7nVZdvWVkrW02q29TMgOuC4EAtGdCutukyqd1DTA+igc2\nDIp+RwJvAgMBAAECggEAINA0rmOI3Hkm/Ufcci7zmNZpA/w7sbSl8uLjK0bzq44j\n+05+PGPupxPEkOdF0oy4r0+K806h04oiW4JUGhm91xbL4dP6Hp7yf7DEbJlVf62n\nZY1jOqlX0u7sY9mC07f5pIMvoXriZPQ3CeJ0I6DIysakZWgKcsh0EoWuERlc5zjE\nf4FeLo5LQNoAMDPBa5PmH4LYB+wej8owkcYPYRaiR+s5ZHjJ7Sz1Pfricg2S6f3p\ntPCrterWVStbmEr5zN9fGAkERLgk44DxvX7gdOp8dOeHGJCs5/0XlzlOD2lZBdkT\nsUYKVfXRNFcxd1zgkA6ugEpVd0boKs9PBXcWtICI0QKBgQDpB01p7vD6f3DMvrMy\nu7BxmY18MWQmEpuiwC3IUePocTD+/mUnyxqordzYVrr9rBVPi/klgHQauJT4DaZf\nqk7AykEG44nH86RHO7V9ltpcTAopwh7Fdn26CBboIw06tRRAi+hsXKK82W0hzLRr\n9bOiYwWIZEMoR1ptwz/1bWCW+QKBgQDaZ2vYgMVIH2Vma0myc/Ss4s9j2BwUZqOu\n4WPexXcxDfkqFm34E0qiuBOy8hHlCTqa+EN8PinSQzOFi5RhgTKlXwJnULFOHok1\ngHIlTBnyIY3vCEsdgotaCCckiePeHs38NqkVF3PR1/I6CJx3Sn14wTGvZI9BvfkN\nXz+Apsw2pwKBgAQe+/B+qEZV0KHeUX75MMKhi44BtZqyw4vaSDT9tcEbl1k7GIDe\np6cKBBjTV9U6oNnaSNqv7d23G/NTEnkoouHn8cR0a7Bcj7AuzyPccholwhxA8Zhe\nYxTSJc2PKSG7qBMIJmEcNkiOs85gN2SdYMLja2qhE0r4EBJW3J+zgnzJAoGAQJq7\nIg3lTZJL8mHmKO14APQOmOBCXh6GaKKvYiRTP6V8gOClou5B86Qz34kPlgqG1XR+\nse/JezL5uJUzTSxkiNpz4Y/TAuKxf6wBF9tNCXy3eW+gmJgXFIi422wWhYvjLqoU\nlfKcBgS02EnXzsmwF9o+Ej3SyoWheD0mka3FVp0CgYEAt5sNlKKDrMnNdjdwzeCv\ntbhOm18fgb2a0ZUGLb8IQrL+o9/w4KYex3yLWkkHoeVoSpjaq2uq4/YS9O07Xm5X\nekKMVir6p0J4w8I1t6uz7OVp1P7mGt7KWFO/KkAPjxJUHWj93cLZIGEINxMtb3MH\nmO2C74/FgxYjk8k2XwKI0J0=\n-----END PRIVATE KEY-----\n",
                "client_email": "document-ai@imagedocumentprocessing.iam.gserviceaccount.com",
                "client_id": "112795268809510843323",
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/document-ai%40imagedocumentprocessing.iam.gserviceaccount.com",
                "universe_domain": "googleapis.com"
            }
            
            if isinstance(ocr_credential_file, dict):
                credentials = Credentials.from_service_account_info(ocr_credential_dict)
                client = vision.ImageAnnotatorClient(credentials=credentials)
            else:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = ocr_credential_file
                client = vision.ImageAnnotatorClient()
            
            image = vision.Image(content=ctxt)

            response = client.text_detection(image=image)

            # for res in response.text_annotations:
            # 	print(res.confidence)

            word_coordinates = []
            for i, text in enumerate(response.text_annotations):
                if i != 0:
                    vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
                    x1 = min([v.x for v in text.bounding_poly.vertices])
                    x2 = max([v.x for v in text.bounding_poly.vertices])
                    y1 = min([v.y for v in text.bounding_poly.vertices])
                    y2 = max([v.y for v in text.bounding_poly.vertices])
                    if x2 - x1 == 0:
                        x2 += 1
                    if y2 - y1 == 0:
                        y2 += 1
                    """"left": x1,
                        "top": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,"""
                    word_coordinates.append({
                        "word": text.description,
                        "left": x1,
                        "top": y1,
                        "width": x2 - x1,
                        "height": y2 - y1,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    })
                else:
                    all_text = text.description

            save_coords = {}
            save_coords_format_2 = []
            for i in range(len(word_coordinates)):
                info = {
                    "text": word_coordinates[i]['word'],
                    "word": word_coordinates[i]['word'],
                    "left": word_coordinates[i]['left'],
                    "top": word_coordinates[i]['top'],
                    "width": word_coordinates[i]['width'],
                    "height": word_coordinates[i]['height'],
                    "x1": word_coordinates[i]['x1'],
                    "y1": word_coordinates[i]['y1'],
                    "x2": word_coordinates[i]['x2'],
                    "y2": word_coordinates[i]['y2'],
                    "bbox": [word_coordinates[i]['x1'], word_coordinates[i]['y1'], 
                            word_coordinates[i]['x2'], word_coordinates[i]['y2']]
                }
                save_coords[i] = info
                save_coords_format_2.append(info)
        file_exists = False
            
    return save_coords, all_text, file_exists


def load_models(tfidf_path, svm_path):
    """Load the trained TF-IDF vectorizer and SVM model from disk."""
    with open(tfidf_path, "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(svm_path, "rb") as f:
        svm = pickle.load(f)
    return tfidf_vectorizer, svm



def save_ocr_results(image_name, ocr_coords, all_text, save_path):
    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    # Generate file names
    ocr_coord_file_name = os.path.join(save_path, image_name.rsplit(".", 1)[0] + "_textAndCoordinates.txt")
    ocr_text_file_name = os.path.join(save_path, image_name.rsplit(".", 1)[0] + "_text.txt")
 
    # Save OCR coordinates
    with open(ocr_coord_file_name, "w") as coord_file:
        coord_file.write(str(ocr_coords))

    # Save OCR extracted text
    with open(ocr_text_file_name, "w") as text_file:
        text_file.write(all_text)

    print(f"OCR results saved at:\n{ocr_coord_file_name}\n{ocr_text_file_name}")


def extract_text_from_image(image_folder_path, image_name):#, ocr_path, ocr_credential_file):
    """Extract text from the given image using OCR Vision API."""
    ocr_coords, all_text, file_exists = get_ocr_vision_api(
        image_path=image_folder_path,
        image_name=image_name,
        # ocr_path=ocr_path,
        ocr_credential_file=ocr_credential_file
    )
    if not file_exists:
        save_ocr_results(image_name, ocr_coords, all_text, OCR_PATH)
    return all_text.strip()

def preprocess_text_for_tfidf(text, tfidf_vectorizer):
    """Transform extracted text into TF-IDF features and convert sparse to dense."""
    print(text)
    return tfidf_vectorizer.transform([text]).toarray()  # Convert to dense array


def classify_image(image_folder_path, image_name, tfidf_vectorizer, svm):
    """Perform OCR on the image, process the text, and classify it using the SVM model."""
    extracted_text = extract_text_from_image(image_folder_path, image_name)#, ocr_path, ocr_credential_file)
    
    if not extracted_text:
        return "Other"  # Return "Other" category for blank images
    
    tfidf_features = preprocess_text_for_tfidf(extracted_text, tfidf_vectorizer)
    prediction = svm.predict(tfidf_features)  # Now input is a dense array
    for category, label in CLASS_MAPPING.items():
        if label == prediction:
            return category
    
    return "Unknown"

    # return prediction[0]


tfidf_path = "/datadrive/rakesh/illumifin_poc/illumifin_poc/models/tf_idf_model.pkl"
svm_path = "/datadrive/rakesh/illumifin_poc/illumifin_poc/models/svm_model.pkl"
tfidf_vectorizer_main, svm_main = load_models(tfidf_path, svm_path)


def main(image_folder_path, image_name):#, ocr_path, ocr_credential_file):
    """Main inference function to classify an image."""

    result = classify_image(image_folder_path, image_name, tfidf_vectorizer_main, svm_main)
    print(f"Predicted category: {result}")
    return result






def process_pdfs_old(pdf_folder, image_folder, res_fol):
    # Ensure image folder exists
    os.makedirs(image_folder, exist_ok=True)
    
    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]
            pdf_image_folder = os.path.join(image_folder, pdf_name)
            
            os.makedirs(pdf_image_folder, exist_ok=True)
            
            # Convert PDF to images using PyMuPDF
            doc = open_pdf(pdf_path)
            for idx, page in enumerate(doc):
                pix = page.get_pixmap()
                image_name = pdf_name + '_' + f"{idx+1:02d}.jpeg"
                image_path = os.path.join(pdf_image_folder, image_name)
                print(image_path)
                pix.save(image_path)
                
                # Run main function on image
                result = main(pdf_image_folder, image_name)
                
                # Save result as JSON per image
                result_json_path = os.path.join(res_fol, f"{image_name}.json")
                with open(result_json_path, "w") as f:
                    json.dump({
                        "pdf_name": pdf_name,
                        "image_name": image_name,
                        "image_path": image_path,
                        "result": result
                    }, f, indent=4)
                
    print(f"Processing completed. Results saved in respective image folders.")



import fitz
from PIL import Image

def get_valid_pixmap(page, dpi):
    """Get a pixmap with the correct resolution dynamically."""
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))
    required_width, required_height = 3000, 4000

    # If resolution is too low, increase DPI
    while pix.width < required_width or pix.height < required_height:
        dpi += 20  # Increase DPI
        print(f"Increasing DPI to {dpi} for better resolution...")
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72))

    return pix



def process_pdfs(pdf_folder, image_folder, res_fol, dpi=40, quality=20):
    """
    Processes PDFs by converting pages into images, running the `main` function on each image,
    and saving the results as individual JSON files.

    Args:
        pdf_folder (str): Path to the folder containing PDF files.
        image_folder (str): Path to the folder where images will be saved.
        res_fol (str): Path to the folder where JSON results will be stored.
        dpi (int, optional): Resolution for image conversion. Default is 150.
        quality (int, optional): JPEG compression quality. Default is 75.
    """
    # Ensure output folders exist
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(res_fol, exist_ok=True)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            pdf_name = os.path.splitext(pdf_file)[0]  # Extract PDF name
            pdf_image_folder = os.path.join(image_folder, pdf_name)

            os.makedirs(pdf_image_folder, exist_ok=True)  # Create subfolder for images

            # Convert PDF to images using PyMuPDF
            doc = fitz.open(pdf_path)
            for idx, page in enumerate(doc):
                image_name = f"{pdf_name}_{idx+1:02d}.jpeg"
                
                # pix = page.get_pixmap(dpi=dpi)  # Lower DPI for smaller images
                # pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))

                result_json_path = os.path.join(res_fol, f"{os.path.splitext(image_name)[0]}.json")
                if not os.path.exists(result_json_path):
                    pix = get_valid_pixmap(page, dpi)
                    image_path = os.path.join(pdf_image_folder, image_name)
                    # print(pix.width, pix.height)
                    print(image_name)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    # Save with compression
                    print('$$$$$$$$$$$$$$$$$$$$$$$$$')
                    print(image_path)
                    print('$$$$$$$$$$$$$$$$$$$$$$$$$')
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    img.save(image_path, "JPEG", quality=quality)
                    # Run main function on the image
                    result = main(pdf_image_folder, image_name)

                    # Save result as JSON per image
                    with open(result_json_path, "w") as f:
                        json.dump({
                            "pdf_name": pdf_name,
                            "image_name": image_name,
                            "image_path": image_path,
                            "result": result
                        }, f, indent=4)

                    print(f"Result saved: {result_json_path}")
                else:
                    print(f"Result already exists: {result_json_path}")
    print("Processing completed. All results are saved in the respective folders.")




if __name__ == "__main__":
    # image_folder_path = "/home/ntlpt19/TF_testing_EXT/dummy_responces/Illumifin_data/All_classes_images"
    # image_name = "/home/ntlpt19/TF_testing_EXT/dummy_responces/Illumifin_data/All_classes_images/Medication Administration Records (MAR)/04_page_2.jpeg"  # Replace with actual image name
    # ocr_path = "/home/ntlpt19/TF_testing_EXT/dummy_responces/Illumifin_data/OCR/All_classes_OCR"  # Replace with actual OCR output path
    # ocr_credential_file = "/home/ntlpt19/TF_testing_EXT/code/miscellaneous_code/src/main/gv_key.json"  # Replace with actual credential file path
    # main(image_folder_path, image_name)#, ocr_path, ocr_credential_file)
    use_gv_flag = True
    OCR_PATH = '/datadrive/rakesh/illumifin_poc/Data/OCR_path'
    ocr_credential_file = "/datadrive/rakesh/illumifin_poc/Data/gv_key.json"
    os.makedirs(OCR_PATH, exist_ok=True)
    # Example usage
    pdf_folder = "/datadrive/rakesh/illumifin_poc/Received_Batch_Documents"
    image_folder = "/datadrive/rakesh/illumifin_poc/Data/root_images"  # Update with actual path
    result_json_path = "/datadrive/rakesh/illumifin_poc/Data/results"  # Update with actual path
    process_pdfs(pdf_folder, image_folder, result_json_path)
