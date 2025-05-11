import re
import json
from google.cloud import vision
import os
import cv2
import numpy as np
import copy


number_mapping = {
    "first":"1",
    "second":"2",
    "third":"3",
    "fourth":"4",
    "fifth":"5",
    "six":"6",
    "seven":"7",
    "nine":"9",
    "ten":"10"
}

token_check = ['Number', 'originals', "ORIGINA"]
padding1 = 150
padding2 = 200

'''def get_bol_total_originals(data):
    result = []
    for item in data:
        text = item[0]
        bbox = item[1]
        confidence = item[2]
        # Use regex to extract the word after the last '/'
        match = re.search(r'\s+(\S+)\s*$', text)
        extracted_word = match.group(1).strip() if match else None
        # Create a new dictionary with the desired key and values
        print(extracted_word)
        if extracted_word:
            result.append([extracted_word, bbox, confidence])

    return result'''


def get_bol_total_originals(text):
    splitting_token = None
    total_original_no = None
    bol_original_number = None
    if '/' in text:
        splitting_token = "/"
    if "of" in text:
        splitting_token = "of"
        
    if splitting_token:
        org_numb_total = text.split(splitting_token)
        print(org_numb_total)
        if len(org_numb_total) > 1:
            total_original_no = org_numb_total[1]
            bol_original_number = org_numb_total[0]
    else:
        total_original_no = text
        
    return total_original_no, bol_original_number

def filter_ocr_data_by_bbox(ocr_data, bbox):
    filtered_ocr = []

    # Extract the coordinates of the bounding box
    bbox_left, bbox_top, bbox_right, bbox_bottom = bbox

    # Filter OCR data based on bbox coordinates
    for item in ocr_data:
        left, top, right, bottom = item['x1'], item['y1'], item['x2'], item['y2']

        # Check if the OCR element is within the specified bbox
        if bbox_left <= left <= bbox_right and bbox_top <= top <= bbox_bottom and bbox_left <= right <= bbox_right and bbox_top <= bottom <= bbox_bottom:
            filtered_ocr.append(item)

    return filtered_ocr



def get_ocr_vision_api_charConfi(image_path):
    # Set your Google Cloud service account credentials
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] =  "/home/ntlpt19/Desktop/TF_release/jan12_TarunG_MVP/TradeFinance/client_code/trade_finance_apis_extraction/src/main/spheric-time-383904-f1b421d86eef.json"

    # Initialize the Vision API client
    client = vision.ImageAnnotatorClient()

    # Load the image
    with open(image_path, 'rb') as image_file:
        image_data = image_file.read()

    # Perform text detection
    image = vision.Image(content=image_data)
    response = client.document_text_detection(image=image)

    # Initialize a list to store the formatted results
    formatted_results = []

    # Initialize a string to store all the extracted text
    all_extracted_text = ""

    # Extract and format the text and bounding box information
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    confidence = word.confidence
                    char_confidences = [symbol.confidence for symbol in word.symbols]

                    vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                    x1 = min([v[0] for v in vertices])
                    x2 = max([v[0] for v in vertices])
                    y1 = min([v[1] for v in vertices])
                    y2 = max([v[1] for v in vertices])

                    formatted_word = {
                        "word": word_text,
                        "confidence": confidence,
                        'char_confi': char_confidences,  # List of character-level confidences
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                    }
                    formatted_results.append(formatted_word)
                    all_extracted_text += word_text + ' '

    return formatted_results, all_extracted_text

import pytesseract
from PIL import Image, ImageSequence
def get_ocr_tesseract(image_path):
	print("called Image OCR...", end="")
	img = Image.open(image_path)
	d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
	all_text = pytesseract.image_to_string(img)
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
	return word_coordinates, all_text

def get_text_from_wc(ocr_wc):
    all_text = ''
    for words in ocr_wc:
        all_text = all_text + words['word'] + " "
    return all_text


def check_all_text(all_text, requited_tokens):
    for token in requited_tokens:
        if token.lower() in all_text.lower():
            return True
def number_from_org_copy(number_mapping, required_text_cpy_org):
    for k, v in number_mapping.items():
        if k in required_text_cpy_org.lower():
            return v
def is_list_of_lists(lst):
    return isinstance(lst, list) and all(isinstance(elem, list) for elem in lst)

    
import logging
# Create a logger
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)  # You can set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Create a file handler and set the logging level
file_handler = logging.FileHandler('my_log_file.log')
file_handler.setLevel(logging.DEBUG)  # You can set the logging level for the file handler

# Create a formatter and attach it to the file handler
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the file handler to the logger
logger.addHandler(file_handler)


img_path = '/home/ntlpt19/Downloads/Evaluation_Data/test_images/BOL/test_images'
result_path = '/home/ntlpt19/Downloads/Evaluation_Data/test_images/BOL/Results_test_images'

for imgs in os.listdir(img_path):
    pred_json = os.path.join(result_path, imgs.split('.')[0]+"1.txt")
    print(pred_json)
    if os.path.exists(pred_json):
        with open(pred_json, "r") as f2:
            predicted = json.load(f2)

        wc, all_text = get_ocr_tesseract(os.path.join(img_path,imgs))

        def segregate_bol_number_total_original(predicted, wc):
            #[['1 / three', [256, 2013, 358, 2035], 77.00571428571429]]
            bol_original_number = None
            #declare a variable for get('bol_original_number') then update the prediction
            pred_org_number = copy.deepcopy(predicted.get('bol_original_number', {}))
            indx_to_delete = []
            bol_originals = []
            for idx, pred in enumerate(pred_org_number):
                print(pred)
                text = pred[0]
                bbox_ = pred[1]
                total_original_no, bol_original_number = get_bol_total_originals(text) #case1
                print(total_original_no)
                print(bol_original_number)
                updated_bbox = [bbox_[0]-padding1, bbox_[1]-padding1,bbox_[2]+padding1, bbox_[3]+padding1]
                required_text = get_text_from_wc(filter_ocr_data_by_bbox(wc, updated_bbox))
                if total_original_no or check_all_text(required_text, token_check): #case2
                    print("...................", idx)
                    # predicted.get('bol_original_number', {}).pop(idx) 
                    indx_to_delete.append(idx)
                    if "total_original_no" in predicted:     
                        predicted["total_original_no"].append([total_original_no, bbox_]+pred[2:])
                    else:
                        predicted["total_original_no"] = [[total_original_no, bbox_]+pred[2:]]
                if bol_original_number:
                    bol_originals.append([bol_original_number, bbox_]+pred[2:])
            if 'bol_original_number' in predicted:
                predicted['bol_original_number'] = np.delete(np.array(predicted['bol_original_number']), indx_to_delete, axis=0).tolist()
                predicted['bol_original_number'].extend(bol_originals)
            if not bol_original_number:
                for pred2 in predicted.get('bol_original_or_copy', {}):
                    text_ = pred2[0]
                    bboxs_ = pred2[1]
                    # confi_ = pred2[2]
                    updated_bbox = [bboxs_[0]-padding2, bboxs_[1]-padding2,bboxs_[2]+padding2, bboxs_[3]+padding2]
                    required_text_cpy_org = get_text_from_wc(filter_ocr_data_by_bbox(wc, updated_bbox))
                    print(required_text_cpy_org) # also add one more check (in the text "no.of originals should not be there")
                    print(updated_bbox)
                    org_number = number_from_org_copy(number_mapping, required_text_cpy_org) #bol_original_number
                    print(org_number)
                    if org_number:
                        if "bol_original_number" in predicted:
                            predicted["bol_original_number"].append([org_number, bboxs_]+pred2[2:])
                        else:
                            predicted["bol_original_number"] = [[org_number, bboxs_]+pred2[2:]]
                        break
            if 'bol_original_number' in predicted and len(predicted['bol_original_number']) == 0:
                del predicted['bol_original_number']
            return predicted

        logger.debug(f"IMAGE NAME: {imgs}")
        logger.debug("*********************************** BEFORE ****************************")
        logger.debug(f"bol_original_number >> {predicted.get('bol_original_number', {})}")
        logger.debug(f"total_original_no >> {predicted.get('total_original_no', {})}")
        logger.debug(f"bol_original_or_copy >> {predicted.get('bol_original_or_copy', {})}")
        predicted = segregate_bol_number_total_original(predicted, wc)
        logger.debug("**************************** AFTER *************************************")
        logger.debug(f"bol_original_number >> {predicted.get('bol_original_number', {})}")
        logger.debug(f"total_original_no >> {predicted.get('total_original_no', {})}")
        logger.debug(f"bol_original_or_copy >> {predicted.get('bol_original_or_copy', {})}")
    # Now you can use the logger to log messages  







'''print(bol_number)
bbox_ = [738, 1973, 751, 1991]
updated_bbox = [bbox_[0]-padding, bbox_[1]-padding,bbox_[2]+padding, bbox_[3]+padding]
required_text = get_text_from_wc(filter_ocr_data_by_bbox(wc, updated_bbox))
if check_all_text(required_text, token_check):
    pass # assign the key to the >>>>>>>>>>>>>>>>    "total_original_no"
'''
'''
total_original_number = get_bol_total_originals(predicted.get('bol_original_number', {}))


print('>>>>>>>>>>>>>>>>>>>>>>')
print(total_original_number)
if total_original_number:
    predicted['total_original_number'] = total_original_number
    '''
    


'''bbox_ = [1134, 492, 1357, 531]                                                   #bol_original_number
updated_bbox = [bbox_[0]-padding, bbox_[1]-padding,bbox_[2]+padding, bbox_[3]+padding]
required_text_cpy_org = get_text_from_wc(filter_ocr_data_by_bbox(wc, updated_bbox))
print(required_text)
'''
'''
image =  cv2.imread(img_path)
cv2.rectangle(image,(updated_bbox[0], updated_bbox[1]), (updated_bbox[2], updated_bbox[3]), (0,255, 0), 2)
cv2.imwrite("testing.png", image)
'''




