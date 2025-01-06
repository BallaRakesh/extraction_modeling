from transformers import LayoutLMForTokenClassification
from transformers import LayoutLMv2Processor
from datetime import datetime
import os, re
from configparser import ConfigParser
import argparse
import glob
import psutil
from datetime import datetime
from fuzzywuzzy import fuzz
from nltk import ngrams
import traceback

import torchvision.transforms as transforms
import torch
from functools import cmp_to_key
from base64 import b64encode
from PIL import Image, ImageDraw, ImageFont
import os, traceback
from datetime import datetime
import cv2, json
import random
import warnings
from PIL import ImageSequence
from sklearn.cluster import DBSCAN
import pytesseract
import gc
from scipy.special import softmax
import psutil
from datetime import datetime

import numpy as np
from scipy import stats
import ast

# Custom encode function
import torch
from transformers import LayoutLMTokenizer

# def custom_processor_no_limit(words, bboxes, pad_token_box=None):
#     if pad_token_box is None:
#         pad_token_box = [0, 0, 0, 0]

#     tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
#     normalized_word_boxes = bboxes

#     # Ensure words and boxes length match
#     assert len(words) == len(normalized_word_boxes), "Mismatch between number of words and bounding boxes."

#     # Initialize lists for tokens and boxes
#     tokenized_words = []
#     token_boxes = []

#     for word, box in zip(words, normalized_word_boxes):
#         # Tokenize the word
#         word_tokens = tokenizer.tokenize(word)

#         # Extend the tokenized words list
#         tokenized_words.extend(word_tokens)

#         # For each subword, append the bounding box of the original word
#         token_boxes.extend([box] * len(word_tokens))

#     print(f"Number of tokens: {len(tokenized_words)}")
#     print(f"Number of bboxes: {len(token_boxes)}")

#     # Add bounding boxes for CLS and SEP tokens
#     token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

#     # Encoding using tokenizer
#     encoding = tokenizer(
#         ' '.join(words), 
#         return_tensors="pt",  # Returns tensors directly
#         add_special_tokens=True,  # Automatically adds CLS and SEP tokens
#         truncation=False,  # Don't truncate (removes length constraint)
#         padding=False  # Don't pad (removes padding)
#     )

#     # Check if the number of tokens matches the bbox list length
#     input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
#     # print(f"input ids shape: {input_ids.shape[0]}")
#     # print(f"attention mask shape: {encoding['attention_mask'].squeeze(0).shape[0]}")
#     # print(f"token type ids shape: {encoding['token_type_ids'].squeeze(0).shape[0]}")
#     # exit('++++++++++++')
#     if len(token_boxes) != input_ids.shape[0]:
#         raise ValueError(f"Number of tokens ({input_ids.shape[0]}) does not match number of bounding boxes ({len(token_boxes)})")
#     print("input id shape: {i}")
#     # exit('====================')
#     encoding['bbox'] = torch.tensor([token_boxes])

#     # Return encoding
#     return encoding
class LayoutLMProcessor:
    def __init__(self, model_name="microsoft/layoutlm-base-uncased"):
        self.tokenizer = LayoutLMTokenizer.from_pretrained(model_name)

    def encode(self, words, bboxes, pad_token_box=None):
        """
        Process words and bounding boxes into tokenized input with LayoutLM tokenization,
        without limiting sequence length or applying padding.
        """
        if pad_token_box is None:
            pad_token_box = [0, 0, 0, 0]

        normalized_word_boxes = bboxes

        # Ensure words and boxes length match
        assert len(words) == len(normalized_word_boxes), "Mismatch between number of words and bounding boxes."

        # Initialize lists for tokens and boxes
        tokenized_words = []
        token_boxes = []

        for word, box in zip(words, normalized_word_boxes):
            # Tokenize the word
            word_tokens = self.tokenizer.tokenize(word)

            # Extend the tokenized words list
            tokenized_words.extend(word_tokens)

            # For each subword, append the bounding box of the original word
            token_boxes.extend([box] * len(word_tokens))

        print(f"Number of tokens: {len(tokenized_words)}")
        print(f"Number of bboxes: {len(token_boxes)}")

        # Add bounding boxes for CLS and SEP tokens
        token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

        # Encoding using tokenizer
        encoding = self.tokenizer(
            ' '.join(words),
            return_tensors="pt",  # Returns tensors directly
            add_special_tokens=True,  # Automatically adds CLS and SEP tokens
            truncation=False,  # Don't truncate (removes length constraint)
            padding=False  # Don't pad (removes padding)
        )

        # Check if the number of tokens matches the bbox list length
        input_ids = encoding['input_ids'].squeeze(0)  # Remove batch dimension
        if len(token_boxes) != input_ids.shape[0]:
            raise ValueError(f"Number of tokens ({input_ids.shape[0]}) does not match number of bounding boxes ({len(token_boxes)})")

        print(f"Input ID shape: {input_ids.shape[0]}")

        encoding['bbox'] = torch.tensor([token_boxes])

        # Return encoding
        return encoding



import torch
from transformers import LayoutLMTokenizer

def custom_processor(words, bboxes, max_seq_length=512, pad_token_box=None):
    if pad_token_box is None:
        pad_token_box = [0, 0, 0, 0]

    tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
    # words = words
    normalized_word_boxes = bboxes

    # Ensure words and boxes length match
    assert len(words) == len(normalized_word_boxes), "Mismatch between number of words and bounding boxes."

    # Initialize lists for tokens and boxes
    tokenized_words = []
    token_boxes = []

    for word, box in zip(words, normalized_word_boxes):
        # Tokenize the word
        word_tokens = tokenizer.tokenize(word)
        # Extend the tokenized words list
        tokenized_words.extend(word_tokens)
        
        # For each subword, append the bounding box of the original word
        token_boxes.extend([box] * len(word_tokens))

    # Truncate token_boxes and tokenized_words if necessary
    special_tokens_count = 2  # CLS and SEP tokens
    total_length = len(token_boxes) + special_tokens_count

    if total_length > max_seq_length:
        excess_length = total_length - max_seq_length
        tokenized_words = tokenized_words[:-excess_length]
        token_boxes = token_boxes[:-excess_length]

    # Add bounding boxes for CLS and SEP tokens
    token_boxes = [[0, 0, 0, 0]] + token_boxes + [[1000, 1000, 1000, 1000]]

    # Encoding input_ids
    encoding = tokenizer(' '.join(tokenized_words), padding='max_length', truncation=True, return_tensors="pt", max_length=max_seq_length)
    
    input_ids = encoding['input_ids'].squeeze(0)  # Squeeze to remove batch dimension

    # Pad token_boxes to max_seq_length
    padding_length = max_seq_length - len(token_boxes)
    token_boxes += [pad_token_box] * padding_length
    token_boxes = token_boxes[:max_seq_length]  # Ensure the size does not exceed max_seq_length

    encoding['bbox'] = torch.tensor(token_boxes)

    # Assertions to verify dimensions
    assert input_ids.shape[0] == max_seq_length, f"input_ids shape {input_ids.shape[0]} does not match max_seq_length {max_seq_length}."
    assert encoding['bbox'].shape[0] == max_seq_length, f"bbox shape {encoding['bbox'].shape[0]} does not match max_seq_length {max_seq_length}."

    return encoding


def get_label_mappings(config_path):
    """
    This function reads the given config.json file and returns the label2id and id2label mappings.
    It ensures that label2id has int values for IDs and string values for labels.
    
    Args:
    config_path (str): Path to the config.json file
    
    Returns:
    dict: label2id mapping
    dict: id2label mapping
    """
    # Load the config.json file
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Extract label2id and id2label
    label2id = {str(key): int(value) for key, value in config.get("label2id", {}).items()}
    id2label = {int(key): str(value) for key, value in config.get("id2label", {}).items()}
    
    return label2id, id2label


# Function to read the classes from the file
def read_classes(file_path):
    with open(file_path, 'r') as file:
        # Read the single line and convert the string representation of the list to an actual list
        classes = ast.literal_eval(file.readline().strip())
    return classes

# Function to create label2id and id2label mappings
def create_label_mappings(labels):
    label2id = {}
    id2label = {}
    
    # Enumerate through the labels to create mappings
    for idx, label in enumerate(labels):
        # Directly use the label as it is, without modifying
        label2id[label] = idx
        id2label[idx] = label
    
    return label2id, id2label

configur = {
    "ALPHA": {
        "Default":1.6,
        "bill_to":1.6,
        "ship_to":1.6,
        "remit_to": 1.6,
        "vendor_name": 1.6
    }
}




process_memory = psutil.Process()
transform2 = transforms.ToPILImage()
transform = transforms.ToTensor()


# will not use DBScan with these tokens
# these token are for merging labels whose values are present in a single line only
original_single_text_labels = ["bill_exchange_no","bill_exchange_date","boe_currency","boe_amount","country_of_origin","invoice_no","invoice_date","invoice_currency","invoice_amount","tenor_type","usance_tenor","tenor_indicator","indicator_type","indicator_date","invoice_due_date","original_or_copy","original_number","lc_ref_no","lc_date","issue_place","transaction_date","awb_bill_no","master_awb_bill_no","house_awb_bill_no","awb_bill_issue_date","flight_no","flight_date","shipper_name","shipper_country","consignee_name","consignee_country","notify_party_name","notify_party_country","carrier_name","agent_name","place_of_receipt","airport_of_departure","airport_of_destination","final_destination","declared_value_of_carriage","amount_insurance","gross_quantity","gross_weight","net_weight","good_marks","invoice_number","invoice_date","lc_no","lc_date","freight_collect_or_prepaid","freight_collected_at","signed_by_carrier","signed_by_agent","awb_original_number","awb_original_or_copy","flight_details","declared_value_of_custom","transaction_date","awb_bill_no","master_awb_bill_no","house_awb_bill_no","awb_bill_issue_date","flight_no","flight_date","shipper_name","consignee_name","notify_party_name","notify_party_country","carrier_name","agent_name","place_of_receipt","airport_of_departure","airport_of_destination","final_destination","declared_value_of_carriage","amount_insurance","gross_quantity","gross_weight","net_weight","good_marks","invoice_number","invoice_date","lc_no","lc_date","freight_collect_or_prepaid","freight_collected_at","awb_original_number","awb_original_or_copy","flight_details","declared_value_of_custom","dimension","at_place","swift_code", "account_number"]
original_vertical_merge_labels = ["shipper_country","shipper_address","carriage_condition","consignee_country","carrier_address","agent_address","goods_description","amount_in_words","drawee_bank_address","drawer_bank_address","drawee_address","nostro_bank_address","drawer_address","consignee_address","consignee_addres","shipper_address","notify_party_address","consignor_address","coo_issuer_address","nostro_bank_address","consignor_address","address_of_assured","drawee_address","insurance_issuer_address","remitter_address","beneficiary_address","coo_issuer_address","notify_party_address","drawer_bank_address","consignee_address","drawer_address","drawee_bank_address","insurance_issuer_address_bottom","drawer_bank_bottom_address","shipper_address","claim_payable_by_address","carrier_country","agent_country"]

single_text_labels = ["delivery_challan_no", "freight_amount", "taxable_amount", "vendor_vat_no", "vendor_name", "vat_code", "tax_amount", "net_amount", "im_vat_no", "charge_type", "cash_discount", "amount", "purchase_order_number", "ship_date", "page_number", "order_date", "customer_no", "invoice_total", "sales_order_number", "order_number", "customer_order_number",      "doc_curr","bill_exchange_no","bill_exchange_date","boe_currency","boe_amount","country_of_origin","invoice_no","invoice_date","invoice_currency","invoice_amount","tenor_type","usance_tenor","tenor_indicator","indicator_type","indicator_date","invoice_due_date", "due_date","original_or_copy","original_number","lc_ref_no","lc_date","issue_place","transaction_date","awb_bill_no","master_awb_bill_no","house_awb_bill_no","awb_bill_issue_date","flight_no","flight_date","shipper_name","shipper_country","consignee_name","consignee_country","notify_party_name","notify_party_country","carrier_name","agent_name","place_of_receipt","airport_of_departure","airport_of_destination","final_destination","declared_value_of_carriage","amount_insurance","gross_quantity","gross_weight","net_weight","good_marks","invoice_number","lc_no","lc_date","freight_collect_or_prepaid","freight_collected_at","signed_by_carrier","signed_by_agent","awb_original_number","awb_original_or_copy","flight_details","declared_value_of_custom","transaction_date","awb_bill_no","master_awb_bill_no","house_awb_bill_no","awb_bill_issue_date","flight_no","flight_date","shipper_name","consignee_name","notify_party_name","notify_party_country","carrier_name","agent_name","place_of_receipt","airport_of_departure","airport_of_destination","final_destination","declared_value_of_carriage","amount_insurance","gross_quantity","gross_weight","net_weight","good_marks","invoice_number","invoice_date","lc_no","lc_date","freight_collect_or_prepaid","freight_collected_at","awb_original_number","awb_original_or_copy","flight_details","declared_value_of_custom","dimension","at_place", "account_number", "swift_code", "tax_percent"]
vertical_merge_labels = ["shipper_country","shipper_address","carriage_condition","consignee_country","carrier_address","agent_address","goods_description","amount_in_words","drawee_bank_address","drawer_bank_address","drawee_address","nostro_bank_address","drawer_address","consignee_address","consignee_addres","shipper_address","notify_party_address","consignor_address","coo_issuer_address","nostro_bank_address","consignor_address","address_of_assured","drawee_address","insurance_issuer_address","remitter_address","beneficiary_address","coo_issuer_address","notify_party_address","drawer_bank_address","consignee_address","drawer_address","drawee_bank_address","insurance_issuer_address_bottom","drawer_bank_bottom_address","shipper_address","claim_payable_by_address","carrier_country","agent_country","ship_to"]        

def most_common(lst):
	return max(set(lst), key=lst.count)

# converts a tuple to string
def tuple_to_string(sen):
	str_test = sen
	word = ""
	for i, w in enumerate(str_test):
		if i == 0:
			word = word + w
		else:
			word = word + ' ' + w
	return word

def area(coordinates):
	l = coordinates[2] - coordinates[0]
	h = coordinates[3] - coordinates[1]
	return l * h

def normalize(points: list, width: int, height: int) -> list:
	x0, y0, x2, y2 = [int(p) for p in points]

	x0 = int(1000 * (x0 / width))
	x2 = int(1000 * (x2 / width))
	y0 = int(1000 * (y0 / height))
	y2 = int(1000 * (y2 / height))

	if x0 > 1000:
		x0 = 1000
	# print(">")
	if x0 < 0:
		x0 = 0
	# print("<")
	if x2 > 1000:
		x2 = 1000
	# print(">")
	if x2 < 0:
		x2 = 0
	# print("<")
	if y0 > 1000:
		y0 = 1000
	# print(">")
	if y0 < 0:
		y0 = 0
	# print("<")
	if y2 > 1000:
		y2 = 1000
	# print(">")
	if y2 < 0:
		y2 = 0
	# print("<")
	return [x0, y0, x2, y2]


def unnormalize_box(bbox, width, height):
	return [
		int(width * (bbox[0] / 1000)),
		int(height * (bbox[1] / 1000)),
		int(width * (bbox[2] / 1000)),
		int(height * (bbox[3] / 1000)),
	]

def contour_sort(a, b):
	if abs(a[1][1] - b[1][1]) <= 15:
		return a[1][0] - b[1][0]
	return a[1][1] - b[1][1]

def minimum_distance_vertical(bb1, bb2):
    # Calculate the minimum vertical distance between two bounding boxes
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    min_distance_y = min(abs(y1_bb2 - y2_bb1), abs(y1_bb1 - y2_bb2))

    return min_distance_y

def vertical_horizontal_values(word_bbox_list):
        # Initialize two lists for separate groups
    group1 = []  # Words with height > width
    group2 = []  # Words with height <= width
    # Iterate through the word_bbox_list and separate words based on their bbox dimensions
    for word, bbox in word_bbox_list:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        if height > width:
            group1.append([word, bbox])
        else:
            group2.append([word, bbox])
        
    return [group1, group2]

def special_chr_check(bb_token, flag):
    if ',' in bb_token:
        flag = False
    if '-' in bb_token:
        flag = False
    return flag

def minimum_distance(bb1, bb2):
	# bb1 points
	min_distance = 9999999999
	p_11 = np.array((bb1[0], bb1[1]))
	p_12 = np.array((bb1[0], bb1[3]))
	p_13 = np.array((bb1[2], bb1[3]))
	p_14 = np.array((bb1[2], bb1[1]))
	all_points_bb1 = [p_11, p_12, p_13, p_14]
	# bb2 points
	p_21 = np.array((bb2[0], bb2[1]))
	p_22 = np.array((bb2[0], bb2[3]))
	p_23 = np.array((bb2[2], bb2[3]))
	p_24 = np.array((bb2[2], bb2[1]))
	all_points_bb2 = [p_21, p_22, p_23, p_24]
	for point1 in all_points_bb1:
		for point2 in all_points_bb2:
			dist = abs(np.linalg.norm(point1 - point2))
			if dist < min_distance:
				min_distance = dist
	return min_distance

def get_iou_horizontal(bb1, bb2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes (horizontal intersection)
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    x_left = max(x1_bb1, x1_bb2)
    y_top = max(y1_bb1, y1_bb2)
    x_right = min(x2_bb1, x2_bb2)
    y_bottom = min(y2_bb1, y2_bb2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    bb1_area = (x2_bb1 - x1_bb1) * (y2_bb1 - y1_bb1)
    bb2_area = (x2_bb2 - x1_bb2) * (y2_bb2 - y1_bb2)
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area

def get_iou_vertical(bb1, bb2):
    # Calculate the Intersection over Union (IoU) of two bounding boxes (vertical intersection)
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    x_left = max(x1_bb1, x1_bb2)
    y_top = max(y1_bb1, y1_bb2)
    x_right = min(x2_bb1, x2_bb2)
    y_bottom = min(y2_bb1, y2_bb2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    bb1_area = (x2_bb1 - x1_bb1) * (y2_bb1 - y1_bb1)
    bb2_area = (x2_bb2 - x1_bb2) * (y2_bb2 - y1_bb2)
    union_area = bb1_area + bb2_area - intersection_area

    return intersection_area / union_area

def get_intersection_percentage(bb1, bb2):
    # Calculate the percentage of vertical intersection between two bounding boxes
    x1_bb1, y1_bb1, x2_bb1, y2_bb1 = bb1
    x1_bb2, y1_bb2, x2_bb2, y2_bb2 = bb2

    x_left = max(x1_bb1, x1_bb2)
    y_top = max(y1_bb1, y1_bb2)
    x_right = min(x2_bb1, x2_bb2)
    y_bottom = min(y2_bb1, y2_bb2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = max(0, y_bottom - y_top)
    bb1_area = y2_bb1 - y1_bb1
    bb2_area = y2_bb2 - y1_bb2

    return intersection_area / min(bb1_area, bb2_area)

def get_iou_new(bb1, bb2):
	"""
	Calculate the Intersection over Union (IoU) of two bounding boxes.

	Parameters
	----------
	bb1 : dict
		Keys: {0, '2', 1, '3'}
		The (x1, 1) position is at the top left corner,
		the (2, 3) position is at the bottom right corner
	bb2 : dict
		Keys: {0, '2', 1, '3'}
		The (x, y) position is at the top left corner,
		the (2, 3) position is at the bottom right corner

	Returns
	-------
	float
		in [0, 1]
	"""
	assert bb1[0] < bb1[2]
	assert bb1[1] < bb1[3]
	assert bb2[0] < bb2[2]
	assert bb2[1] < bb2[3]

	# determine the coordinates of the intersection rectangle
	x_left = max(bb1[0], bb2[0])
	y_top = max(bb1[1], bb2[1])
	x_right = min(bb1[2], bb2[2])
	y_bottom = min(bb1[3], bb2[3])

	if x_right < x_left or y_bottom < y_top:
		return 0.0

	# The intersection of two axis-aligned bounding boxes is always an
	# axis-aligned bounding box
	intersection_area = (x_right - x_left) * (y_bottom - y_top)

	# compute the area of both AABBs
	bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
	bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
	assert iou >= 0.0
	assert iou <= 1.0
	return iou


def model_output_sum(key, box, model_output):
	all_values = model_output[key]
	all_values = sorted(all_values, key=cmp_to_key(contour_sort))
	all_text = ""
	for value in all_values:
		try:
			iou = get_iou_new(value[1], box)
		except:
			continue
		if iou > 0:
			if all_text == "":
				all_text = value[0]
			else:
				all_text = all_text + " " + value[0]
	return all_text


def check_vertical_distribution(bb1, bb2):
    y1 = bb1[1]
    y2 = bb2[1]
    return abs(y1-y2)


def merge_surrounding(data, model_output, w, h):
    new = data.copy()
    print('entered into merging_surroundings ++++++++++++++++++++++++++++++++++++++')
    for key in list(data.keys()):
        print(key)
        if key in vertical_merge_labels:
            all_values = data[key]
            # all_values = vertical_horizontal_values(new_all_values)
            # if key=='drawee_address':
            #     print(all_values)
            #     exit("PPPPPPPPPPPP")
            bboxes = [x[1] for x in data[key]]
            eps_horizontal = 100  # Threshold for horizontal merging
            eps_vertical = 100    # Threshold for vertical merging
            ######################
            # if w>h:
            #     eps_horizontal = round(h*17/100)#100  
            #     eps_vertical = round(w*12/100) #100
            # else:
            #     eps_horizontal = round(h*12/100)#100  
            #     eps_vertical = round(w*17/100) #100
            ############################
            all_values = data[key]
            print(all_values)
            # for all_values in new_all_values:
            length = len(all_values)
            if length > 1:
                i = 0
                # if (bb1_height > bb1_width and bb2_height > bb2_width) or (bb1_height < bb1_width and bb2_height < bb2_width):
                while i in range(length - 1):
                    # print("i is ++++>> ", i)
                    bb1 = all_values[i][1]
                    bb2 = all_values[i + 1][1]
                    bb1_token = all_values[i][0]
                    bb2_token = all_values[i+1][0]
                    confs = [all_values[i][2], all_values[i + 1][2]]
                    min_dist_horizontal = minimum_distance(bb1, bb2)
                    min_dist_vertical = minimum_distance_vertical(bb1, bb2)

                    try:
                        IOU_horizontal = get_iou_horizontal(bb1, bb2)
                        IOU_vertical = get_iou_vertical(bb1, bb2)
                        inter_percentage = get_intersection_percentage(bb1, bb2)
                    except:
                        # print("i is ++++ except ", i)
                        i = i + 1 ; continue
                    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1
                    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2
                    bb1_width = bb1_x2 - bb1_x1
                    bb1_height = bb1_y2 - bb1_y1
                    bb2_width = bb2_x2 - bb2_x1
                    bb2_height = bb2_y2 - bb2_y1
                    # if len(bb1_token)<3:
                    print('beore',bb2_width,bb2_height)
                    print('bb2_token length',len(bb2_token))
                    flag1 = True ; flag2 = True
                    flag1 = special_chr_check(bb1_token, flag1)
                    flag2 = special_chr_check(bb2_token, flag2)
                   
                    if len(bb1_token)==1 or (len(bb1_token)<3 and flag1==False):
                        temp = bb1_width
                        bb1_width = bb1_height
                        bb1_height = temp
                    if len(bb2_token)==1 or (len(bb2_token)<3 and flag2==False):
                        temp = bb2_width
                        bb2_width = bb2_height
                        bb2_height = temp
                    # print('flag2', flag2)
                    # print('bb2_token', bb2_token)
                    # print('bb1_width', bb1_width, 'bb1_height', bb1_height)
                    # print('bb2_width', bb2_width, 'bb2_height', bb2_height)
                    if (bb1_height >= bb1_width and bb2_height >= bb2_width) or (bb1_height <= bb1_width and bb2_height <= bb2_width):
                        # This case is when both the boxes are either horizontal or vertically alligned on a document.
                        print('entered into first if')
                        print(f"min_dist_horizontal : {min_dist_horizontal} h_eps : {h_eps} IOU_horizontal : {IOU_horizontal} inter_percentage : {inter_percentage} || min_dist_vertical : {min_dist_vertical} v_eps : {v_eps} or IOU_vertical : {IOU_vertical} ")
                        if (min_dist_horizontal <= eps_horizontal or IOU_horizontal > 0 or inter_percentage>0) or (min_dist_vertical <= eps_vertical or IOU_vertical > 0 or inter_percentage):
                            print('entered into second if')
                            print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
                            x_left = min(bb1[0], bb2[0])
                            y_top = min(bb1[1], bb2[1])
                            x_right = max(bb1[2], bb2[2])
                            y_bottom = max(bb1[3], bb2[3])
                            box = [x_left, y_top, x_right, y_bottom]
                            text = model_output_sum(key, box, model_output)
                            # print("merged text is ", text)
                            avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
                            new_value = [text, box, [avg_confs]]
                            # print(new_value)
                            all_values.remove(all_values[i])
                            all_values.remove(all_values[i])
                            all_values.insert(i, new_value)
                            print(all_values)
                            length = len(all_values)
                            if length == 1:
                                print("will break")
                                break
                        else:
                            print("distance is very high")
                            i = i + 1
                    else:
                        i = i+1
            else:
                print("will continue")
                continue
            
        else:
            print('Vertical merging not happening ++++++++++++++++++++++++++++++++++++')
            # print("Image dim: ", w, h)
            # print(key)
            bboxes = [x[1] for x in data[key]]
            if w>h:
                v_eps = round(h*1.5/100)#10round(number)
                h_eps = round(w*5/100) #36
            else:
                v_eps = round(h*1.1/100)#10round(number)
                h_eps = round(w*5.8/100) #36
            # print("Epsilons : ", v_eps, h_eps)
            all_values = data[key]
            # print('all_values >>>>>>>>>>>',all_values)
            length = len(all_values)
            if length > 1:
                i = 0
                while i in range(length - 1):
                    # print("i is ++++>> ", i)
                    bb1 = all_values[i][1]
                    bb2 = all_values[i + 1][1]
                    confs = [all_values[i][2], all_values[i + 1][2]]
                    # ocr_confs = [all_values[i][3],all_values[i+1][3]]
                    # min_dist = minimum_distance(bb1, bb2)
                    vertical_distance = check_vertical_distribution(bb1, bb2)
                    # dist_bwt_words = (abs(bb1[2]-bb2[0])/w)*100
                    hori_distance = abs(bb1[2]-bb2[0])
                    # print('min_dist========>', vertical_distance,'bb1',bb1, 'bb2', bb2)
                    # print('hori_distance ==========>', hori_distance)
                    try:
                        IOU = get_iou_new(bb1, bb2)
                        # print('IOU>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', IOU)
                    except:
                        # print("i is except ++++>> ", i)
                        i = i + 1
                        continue
                    
                    print(f"vertical_distance : {vertical_distance} v_eps : {v_eps} IOU : {IOU} hori_distance : {hori_distance} h_eps : {h_eps}")
                    if (vertical_distance <= v_eps and hori_distance<h_eps) or IOU > 0.1:
                        print("merging: " + all_values[i][0] + " and " + all_values[i + 1][0])
                        x_left = min(bb1[0], bb2[0])
                        y_top = min(bb1[1], bb2[1])
                        x_right = max(bb1[2], bb2[2])
                        y_bottom = max(bb1[3], bb2[3])
                        box = [x_left, y_top, x_right, y_bottom]
                        text = model_output_sum(key, box, model_output)
                        print("merged text is ", text)
                        avg_confs = (confs[0] * area(bb1) + confs[1] * area(bb2)) / (area(bb1) + area(bb2))
                        """if "NA" in ocr_confs:
                            avg_ocr_confs = "NA"
                        else:
                            avg_ocr_confs = ( ocr_confs[0]* area(bb1) + ocr_confs[1]*area(bb2) )/(area(bb1) + area(bb2))"""
                        new_value = [text, box, avg_confs]
                        # print(new_value)
                        all_values.remove(all_values[i])
                        all_values.remove(all_values[i])
                        all_values.insert(i, new_value)
                        # print(all_values)
                        length = len(all_values)
                        if length == 1:
                            print("will break")
                            break
                    else:
                        print("distance is very high")
                        i = i + 1
            else:
                print("will continue")
                continue

# lookup function to find contries in 'bill to', 'ship to', 'remit to' fields
def lookup(
		text,
		n_words,
		match_threshold,
		file_path,
		result_set,
		key
):
	try:
		out_dict = {}
		out_list = []
		all_box = []
		message = ""

		read_file = open(file_path, "r")
		lines = read_file.readlines()

		txt_words = []
		for l in lines:
			line = l.split("\n")
			txt_words.append(line[0])

		if n_words > 5:
			status = "N words larger than 5, provide N words less than 5"
			out_dict["status"] = status
			return out_dict

		if match_threshold < 80:
			status = "Matching threshold value less than 80, provide Matching threshold greater than 80"
			out_dict["status"] = status
			return out_dict
		else:
			res = re.sub(r"[^\w\s]", "", text)
			gram_list = []
			for j in range(n_words):
				gram_count = ngrams(res.split(), j + 1)

				for gram in gram_count:
					sen = tuple_to_string(gram)
					gram_list.append(sen)

			for word in gram_list:
				for txt_char in txt_words:

					# print(word[0].lower())

					if fuzz.ratio(word.lower(), txt_char) > match_threshold:
						# if word.lower() in txt_words:
						# print(word, "----", txt_char, fuzz.ratio(word.lower(), txt_char))

						info_dict = {}
						info_dict["searched_string"] = word  # searched string is our data.
						info_dict["found_string"] = txt_char  # found string is present in countries.txt (lookup read_file)
						info_dict["string_match_value"] = fuzz.ratio(
							word.lower(), txt_char
						)
						out_list.append(info_dict)
			if len(out_list) != 0:
				for res in out_list:
					look_up = res['found_string']
					# print(found)
					original = res['searched_string']
					original_words = original.split()
					for val in result_set[key]:
						for word in original_words:
							if fuzz.ratio(word.lower(), val[0]) > match_threshold:
								all_box.append(val[1])
					x1 = min([x[0] for x in all_box])
					x2 = max([x[2] for x in all_box])
					y1 = min([x[1] for x in all_box])
					y2 = max([x[3] for x in all_box])
					box_result = [x1, y1, x2, y2]
					res['bbox'] = box_result

			print(out_list)

			return out_list

	except Exception as e:
		print(traceback.format_exc())


def merge_words_in_bbox(ocr_data, bbox):
    merged_words = []
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    for word_info in ocr_data:
        word_x1, word_y1, word_x2, word_y2 = word_info["x1"], word_info["y1"], word_info["x2"], word_info["y2"]
        if bbox_x1-4 <= word_x1 <= bbox_x2+4 and bbox_x1-4 <= word_x2 <= bbox_x2+4 and \
            bbox_y1-4 <= word_y1 <= bbox_y2+4 and bbox_y1-4 <= word_y2 <= bbox_y2+4:
            merged_words.append(word_info)
    print("merged_words : ", merged_words)
    simple_text = ' '.join([x["word"] for x in merged_words])
    return simple_text

def image_result(file, image_folder_name, model, processor, device, data_folder_path, output_folder_name):
    folder_path = data_folder_path
    result_path = f"{folder_path}/{output_folder_name}/"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    #############################################################
    print(f"file name is      : {file}")
    print(f"image folder path : {image_folder_name}")
    t_total_start = datetime.now()
    print("Calling Image result")
    all_page_result: dict = {}
    print("is all_page_result is instance of dict? %s", isinstance(all_page_result, dict))
    #############################################################

    # initialising basic variables
    count = 0

    if len(image_folder_name) == 0:
        im = Image.open(os.path.join(file))
        file = list(file.split('/'))[-1]
    else:
        im = Image.open(os.path.join(folder_path, image_folder_name, file))   
        

    ############################################################    
    for i, image in enumerate(ImageSequence.Iterator(im)):
        count += 1
        print(f"********** Index : {i} ***************")
        print("******* Page " + str(count) + "********")

        t_page_start = datetime.now()
        w, h = image.size
        temp = image.convert("L")
        image_data = np.asarray(temp)
        image = cv2.cvtColor(image_data, cv2.COLOR_GRAY2RGB)
        arr = transform(image)
        
        ocr_path = os.path.join(folder_path, "OCR")
        # path = os.path.join(folder_path, "image_data")
        # label_path = os.path.join(folder_path, "label_id_mapping.txt")
        # label_path = os.path.join(folder_path, "model_label_mappings.txt")

        print("OCR Path   : ", ocr_path)
        # print("Label Path : ", label_path)

        try:
            print()
            try:
                with open(os.path.join(ocr_path, file[:-4] + "_text.txt"), "r") as f:
                    word_coordinates = json.load(f)['word_coordinates']
            except:
                # ocr_path = os.path.realpath(os.path.join(folder_path, "../ocr_data"))
                ocr_path = os.path.realpath(ocr_data_path)
                print("OCR Path   : ", ocr_path)
                with open(os.path.join(ocr_path, file[:-4] + "_textAndCoordinates.txt"), "r") as f:
                    word_coordinates = eval(f.read())
                # Debug statements
                # print("word_coordinates are:")
                # print(word_coordinates)
            f.close()
        
        except Exception as e:
            print(e)
            traceback.print_exc()
            exit()
        
        
        # check if infence is already present
        if os.path.exists(os.path.join(result_path, file[:-4] + str(count) + "model_output.txt")) and \
            os.path.exists(os.path.join(result_path, file[:-4] + str(count) + "_lookup.txt")) and \
                os.path.exists(os.path.join(result_path, file[:-4] + str(count) + ".txt")) and \
                    os.path.exists(os.path.join(result_path, file[:-4] + str(count) + ".png")) and \
                        os.path.exists(os.path.join(result_path, file[:-4] + "all_page_result.txt")):
                            print(f"Inference for this image {file} already exists")
                            continue
            
        if len(word_coordinates) == 0:
            print("Not enough text")
        
        words: list = []
        bboxes: list = []
        bounding_boxes: list = []
        # print("is words is instance of list? %s", isinstance(words, list))
        # print("is bboxes is instance of list? %s", isinstance(bboxes, list))
        # print("is bounding_boxes is instance of list? %s", isinstance(bounding_boxes, list))
        
        for t in word_coordinates:
            if 'right' in list(t.keys()):
                t['x1'] = t['left']
                t['y1'] = t['top']
                t['x2'] = t['right']
                t['y2'] = t['bottom']
            words.append(t['word'])
            bounding_boxes.append([t['x1'], t['y1'], t['x2'], t['y2']])
            bboxes.append(normalize([t['x1'], t['y1'], t['x2'], t['y2']], w, h))
        
        # Debug statements
        #print(words)
        #print(bboxes)
        #print(bounding_boxes) 
        #exit()
        
        # encoded_inputs = processor(arr, words, boxes=bboxes, return_tensors="pt")
        encoded_inputs= processor.encode(words, bboxes)
        # print('#################**********************************************###########################################################')
        print(encoded_inputs)
        # exit('+++++++++++==')
        ######################################################################
        input_id_chunks = list(encoded_inputs['input_ids'][0].split(510))
        #print(input_id_chunks)
        token_type_id_chunks = list(encoded_inputs['token_type_ids'][0].split(510))
        #print(token_type_id_chunks)
        mask_chunks = list(encoded_inputs['attention_mask'][0].split(510))
        #print(mask_chunks)
        bbox_chunks = list(encoded_inputs['bbox'][0].split(510))
        #print(bbox_chunks)
        # image_chunk = encoded_inputs['image'][0]
        #print(image_chunk)
        # image_chunks = list()
        # print("is input_id_chunks is instance of list? %s", isinstance(input_id_chunks, list))
        # print("is mask_chunks is instance of list? %s", isinstance(mask_chunks, list))
        # print("is bbox_chunks is instance of list? %s", isinstance(bbox_chunks, list))
        # print("is image_chunks is instance of list? %s", isinstance(image_chunks, list))
        # loop through each chunk}
        #exit()
        #######################################################################
        
        
        print(input_id_chunks)
        # exit('++++++++++++++++')
        for i in range(len(input_id_chunks)):
            # image_chunks.append(image_chunk)
            # add CLS (start-of-sequence) and SEP (separator) tokens to input IDs
            input_id_chunks[i] = torch.cat([
                torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
            ])
            token_type_id_chunks[i] = torch.cat([
                torch.tensor([0]), token_type_id_chunks[i], torch.tensor([0])
            ])
            # add attention tokens to attention mask
            mask_chunks[i] = torch.cat([
                torch.tensor([1]), mask_chunks[i], torch.tensor([1])
            ])
            bbox_chunks[i] = torch.cat([
                torch.tensor([[0, 0, 0, 0]]), bbox_chunks[i], torch.tensor([[1000, 1000, 1000, 1000]])
            ])
            
            # get required padding length
            pad_len = 512 - input_id_chunks[i].shape[0]
            # check if tensor length satisfies required chunk size
            if pad_len > 0:
                # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat([
                    input_id_chunks[i], torch.Tensor([0] * pad_len)
                ])
                token_type_id_chunks[i] = torch.cat([
                    token_type_id_chunks[i], torch.Tensor([0] * pad_len)
                ])
                mask_chunks[i] = torch.cat([
                    mask_chunks[i], torch.Tensor([0] * pad_len)
                ])

                bbox_chunks[i] = torch.cat([
                    bbox_chunks[i], torch.Tensor([[0, 0, 0, 0]] * pad_len)
                ])

        ###################################################
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)
        token_type_ids = torch.stack(token_type_id_chunks)
        bbox = torch.stack(bbox_chunks)
        # images = torch.stack(image_chunks)
        ####################################################
        
        ####################################################
        input_dict = {
            'input_ids': input_ids.long().to(device),
            'attention_mask': attention_mask.float().to(device),
            'token_type_ids': token_type_ids.long().to(device),
            'bbox': bbox.long().to(device),
            # 'image': images.float().to(device)
        }
        #####################################################
        
        print("is input_dict is instance of dict? %s", isinstance(input_dict, dict))
        outputs = model(**input_dict)
        print("Model Called")
        print(f'RAM memory % a used for {file}:', psutil.virtual_memory()[2])
        # print(outputs)
        
        #######################################################
        all_predictions: list = []
        all_boxes: list = []
        all_confidences: list = []
        all_text: list = []
        print("is all_predictions is instance of list? %s", isinstance(all_predictions, list))
        print("is all_boxes is instance of list? %s", isinstance(all_boxes, list))
        print("is all_confidences is instance of list? %s", isinstance(all_confidences, list))
        print("is all_text is instance of list? %s", isinstance(all_text, list))
        ########################################################
        
        # this is the list of classes that will be given to us to be extracted.
        # with open(label_path, "r") as f:
        #     print("(((((((((((((((())))))))))))))))")
        #     labels = f.read().split("\n")
        #     print(labels)
        #     print("(((((((((((((((())))))))))))))))")
        # f.close()
        
        # labels = list(model.config.label2id.keys())
        # print("before >> ", labels)
        # # Creating two dictionaries labels2id and id2labels
        # labels = [x.replace("S-", "") for x in labels if x != ""]
        # print("Prediction Labels : ", labels)
        # Read the classes from the file
        labels = read_classes(classes_path)
        label2id, id2label = create_label_mappings(labels)
        # label2id, id2label = model.config.label2id, model.config.id2label
        # print(f"label2id  from classes.txt: {label2id}")
        # print(f"id2label from classes.txt: {id2label}")
        # exit('++++++++++++++++')   
        # label2id= {'S-gstin': 0, 'S-invoice_number': 1, 'S-bill_to': 2, 'S-invoice_date': 3, 'S-remit_to': 4, 'S-invoice_total': 5, 'S-taxable_amount': 6, 'S-swift_code': 7, 'S-ship_to': 8, 'S-page_number': 9, 'S-net_amount': 10, 'S-due_date': 11, 'S-amount': 12, 'S-freight_amount': 13, 'S-pan_number': 14, 'S-purchase_order_number': 15, 'S-order_date': 16, 'S-account_number': 17, 'O': 18, 'S-customer_no': 19, 'S-invoice_total_in_words': 20, 'S-sales_order_number': 21, 'S-order_number': 22, 'S-doc_curr': 23, 'S-tax_percent': 24, 'S-tax_amount': 25, 'S-vendor_name': 26, 'S-delivery_challan_no': 27}
        # id2label= {0: 'S-gstin', 1: 'S-invoice_number', 2: 'S-bill_to', 3: 'S-invoice_date', 4: 'S-remit_to', 5: 'S-invoice_total', 6: 'S-taxable_amount', 7: 'S-swift_code', 8: 'S-ship_to', 9: 'S-page_number', 10: 'S-net_amount', 11: 'S-due_date', 12: 'S-amount', 13: 'S-freight_amount', 14: 'S-pan_number', 15: 'S-purchase_order_number', 16: 'S-order_date', 17: 'S-account_number', 18: 'O', 19: 'S-customer_no', 20: 'S-invoice_total_in_words', 21: 'S-sales_order_number', 22: 'S-order_number', 23: 'S-doc_curr', 24: 'S-tax_percent', 25: 'S-tax_amount', 26: 'S-vendor_name', 27: 'S-delivery_challan_no'}
        labels= list(label2id.keys())
        # label2id = {label: idx for idx, label in enumerate(labels)}
        # id2label = {idx: label for idx, label in enumerate(labels)}
        print("after >> ", label2id)
        # print("is label2id is instance of dict? %s", isinstance(label2id, dict))
        # print("is id2label is instance of dict? %s", isinstance(id2label, dict))
        
        
        # fixed 80 unique hexcodes has been created if more than 80 classes will be there 
        # this needs to change
        ############################################################################## 
        number_of_colors: int = 80
        # creating random hexcodes
        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) 
                    for _ in range(number_of_colors)]
        color = ["#1656AD" for _ in color] # Dark Blue
        ##############################################################################
        
        # print(color)
        # color for each label
        label2color = {}
        # print("is label2color is instance of dict? %s", isinstance(label2color, dict))
        for i, l in enumerate(labels):
            label2color[l] = color[i]
        
        # print(f"label2color : {label2color}")
        ########################################################################3                


        for i, output in enumerate(outputs.logits):
            #print(i, output)
            # converting back into PIL image
            new_img = transform2(arr)
            # loading the image font
            font = ImageFont.truetype(font = "./arial.ttf", size = 20)
            
            # print("output is", output.cpu().detach().numpy())
            predictions = output.argmax(-1).squeeze().tolist()
            # print('the predictions', predictions)
            
            confidences = softmax(output.cpu().detach().numpy(), axis=1)
            #print(confidences)
            
            max_confidences = np.max(confidences, axis=1).reshape(confidences.shape[0], -1)
            #print(max_confidences)
            
            all_confidences += [x[0] for x in max_confidences]
            # print("all_confidences :",all_confidences)
            
            token_boxes = bbox_chunks[i].squeeze().tolist()
            width, height = new_img.size
            true_predictions = [id2label[prediction] for prediction in predictions]
            all_predictions += true_predictions
            #print('all_predictions',all_predictions)
            
            true_boxes = [unnormalize_box(box, width, height) for box in token_boxes]
            all_boxes += true_boxes
            #print(all_boxes)
            
            for id in input_dict['input_ids'][i]:
                all_text.append(processor.tokenizer.decode(id))

        #print(all_text)
        #exit()
        
        del outputs

        for p,t,c,b in zip(all_predictions, all_text, all_confidences, all_boxes):
            if p in ["O", "o", '[CLS]', '[SEP]', '[PAD]']: continue
            print(f">> {p} | {t} | {c} | {b}")


        new_img = transform2(arr)
        draw = ImageDraw.Draw(new_img)
        # print("%20s - %30s - %12s - %30s" % ("Text", "Prediction", "Confidence", "Bounding Box"))

        print("confidences :=: ", confidences)
        print("all_predictions :=: ", all_predictions)
        print("all_text :=: ", all_text)
        

        curr_box: list = []
        results_pred: list = []
        results_conf: list = []
        results_bbox: list = []
        results_text: list = []
        temp_preds: list = [] 
        temp_confs: list = []
        temp_text: list = []
        sep_index: list = all_text.index('[ S E P ]')
        
        if len(all_text) > 512:
            if '[ P A D ]' in all_text:
                sep_index = all_text.index('[ P A D ]') - 2
            else:
                sep_index = len(all_text) - 3
        
        # print(sep_index)
        for i in range(len(all_text)):
            if all_text[i] not in ['[ C L S ]', '[ S E P ]', '[ P A D ]']:  # and all_predictions[i] != 'O':
                # print(i)
                if (curr_box != all_boxes[i] and \
                    len(temp_text) > 0) or \
                        (i == sep_index - 1 and len(temp_text) > 0):
                    
                    # print("1: ", all_text[i])
                    if i == sep_index - 1:
                        temp_text.append(all_text[i])
                        temp_confs.append(all_confidences[i])
                        temp_preds.append(all_predictions[i])
                    text = ""
                    pred = "O"
                    preds = [x for x in temp_preds if x != 'O']
                    conf = 0
                    # print(f">>|> {all_predictions[i]} | {all_text[i]}")
                    # print("pred ::", pred)
                    # print("ALL CONF: ", all_confidences[i])
                    if len(preds) > 0:
                        pred = most_common(preds)
                    # print("temp_text:", temp_text, temp_preds, temp_confs)    
                    for j in range(len(temp_text)):
                        text += temp_text[j].replace("##", "")
                        if temp_preds[j] == pred:
                            conf += temp_confs[j]
                    # print("temp_text-after:", temp_text)
                    conf = float(np.round(conf * 100 / len(temp_text), 2))
                    # print("conf: ", conf)
                    results_text.append(text)
                    results_conf.append(conf)
                    results_pred.append(pred)
                    results_bbox.append(curr_box)

                    temp_text = []
                    temp_confs = []
                    temp_preds = []
                    if i != sep_index - 1:
                        temp_text.append(all_text[i])
                        temp_confs.append(all_confidences[i])
                        temp_preds.append(all_predictions[i])
                        curr_box = all_boxes[i]
                elif curr_box == all_boxes[i]:
                    temp_text.append(all_text[i])
                    temp_confs.append(all_confidences[i])
                    temp_preds.append(all_predictions[i])
                elif len(temp_text) == 0:
                    temp_text.append(all_text[i])
                    temp_confs.append(all_confidences[i])
                    temp_preds.append(all_predictions[i])
                    curr_box = all_boxes[i]
        # print("results_text !!", results_text)
        # print("results_pred !!", results_pred)
        # print("result_conf  !!", results_conf)
        #print('#',results_text)
        #print('##',results_conf)
        #print(results_pred)
        #print(results_bbox)
        #exit()
        result_set = {}
        # print("is result_set is instance of dict? %s", isinstance(result_set, dict))

        for i in range(len(results_pred)):
            if results_pred[i] != 'O':
                if results_pred[i] not in list(result_set.keys()):
                    result_set[results_pred[i]] = []
                result_set[results_pred[i]].append([results_text[i], 
                                                    results_bbox[i], results_conf[i]])
        # print(result_set)
        model_output = result_set.copy()
        print(f'model output++++++++++++++++++++++++++++=={model_output}')
        # exit()
        # print("+++++++++++++++++++reached here+++++++++++++++++")
        # exit("+++++++++")
        with open(os.path.join(result_path, file[:-4] + str(count) + "model_output.txt"), "w") as f:
            json.dump(result_set, f)
        final_result_set = {}
        # print("is final_result_set is instance of dict? %s", isinstance(final_result_set, dict))
        f.close()
        
        ####################### 
        for k in list(result_set.keys()):
            if k not in single_text_labels:
                alpha_data = configur["ALPHA"]
                if k in list(alpha_data.keys()):
                    alpha = float(alpha_data[k])
                else:
                    alpha = float(alpha_data['Default'])
                
                if len(result_set[k]) > 1:
                    # print("++++++++++++++entry in this block+++++++++++")
                    texts = [x[0] for x in result_set[k]]
                    bboxes = [x[1] for x in result_set[k]]
                    confs = [x[2] for x in result_set[k]]
                    avg_w = np.mean([abs(x[0] - x[2]) for x in bboxes])
                    avg_h = np.mean([abs(x[1] - x[3]) for x in bboxes])
                    eps = np.sqrt(avg_w ** 2 + avg_h ** 2) * alpha
                    # if eps<=0.0:
                    # 	eps=0.1
                    clustering = DBSCAN(eps=eps, min_samples=1).fit(bboxes)
                    label_set = set(clustering.labels_)
                    for l in label_set:
                        selected = list(np.where(clustering.labels_ == l)[0])
                        selected_texts = [x for i, x in enumerate(texts) if i in selected]
                        selected_boxes = [x for i, x in enumerate(bboxes) if i in selected]
                        selected_confs = [x for i, x in enumerate(confs) if i in selected]
                        text_boxes = [[x, y] for x, y in zip(selected_texts, selected_boxes)]
                        text_boxes = sorted(text_boxes, key=cmp_to_key(contour_sort))
                        text_result = ""
                        # print(k)
                        # print(text_boxes)
                        for tb in text_boxes:
                            if text_result == "":
                                text_result += tb[0]
                            else:
                                text_result += " " + tb[0]
                        # print(text_result)
                        x1 = min([x[0] for x in selected_boxes])
                        x2 = max([x[2] for x in selected_boxes])
                        y1 = min([x[1] for x in selected_boxes])
                        y2 = max([x[3] for x in selected_boxes])
                        box_result = [x1, y1, x2, y2]
                        conf_result = float(np.round(np.mean(selected_confs), 2))
                        # print(box_result)
                        if k not in list(final_result_set.keys()):
                            final_result_set[k] = []
                        final_result_set[k].append([text_result, box_result, conf_result])
                else:
                    if k not in list(final_result_set.keys()):
                        final_result_set[k] = []
                    final_result_set[k].append([result_set[k][0][0],result_set[k][0][1],result_set[k][0][2]])
            else:
                    if len(result_set[k]) > 1:
                        # print("++++++++++++++entry in this block+++++++++++")
                        texts = [x[0] for x in result_set[k]]
                        bboxes = [x[1] for x in result_set[k]]
                        confs = [x[2] for x in result_set[k]]
                        for i, value in enumerate(zip(texts, bboxes, confs)):
                            # print(list(value))
                            if k not in list(final_result_set.keys()):
                                final_result_set[k] = []
                            final_result_set[k].append(list(value))
                    else:
                        if k not in list(final_result_set.keys()):
                            final_result_set[k] = []
                        final_result_set[k].append([result_set[k][0][0],result_set[k][0][1], result_set[k][0][2]])  

        # print(final_result_set)
        # exit('++++++++++++++++++++')
        merge_surrounding(final_result_set, model_output, w, h)
        # print(final_result_set)
        # exit('final_result_set>>>>>>>>')
        # print("+++++++++++reached here after merge surrounding++++++++++")
        for k in list(final_result_set.keys()):
            all_values = final_result_set[k]
            # print(all_values)
            # exit('________________')
            # for value in all_values:
            #     draw.rectangle(value[1], outline=label2color[k], width=2)
            #     draw.text((value[1][0] + 5, value[1][1] - 20),
            #                 text=k , fill=label2color[k], font=font)
            for value in all_values:
                draw.rectangle(value[1], outline=label2color[k], width=2)
                draw.text((value[1][0] + 5, value[1][1] - 20),
                            text=k + " - " + str(value[2]), fill=label2color[k], font=font)
        print("\n\nFinal Result Set : ")
        print(final_result_set)
        # exit('++++++++++++======')
        lookup_result = {}
        t_page_end = datetime.now()
        print("\nTime taken for page " + str(count)+' of ' + str(file) + ":" + str(t_page_end - t_page_start))
        all_page_result["Page Number " + str(count)] = final_result_set
        for k in list(final_result_set.keys()):
            if k in ["applicant_country", "beneficiary_country"]:
                for val in final_result_set[k]:
                    result_country = lookup(val[0], 4, 90, "countries.txt", result_set, k)
                    result_company = lookup(val[0], 4, 90, "organization.txt", result_set, k)
                    for res in result_company:
                        found = res['found_string']
                        searched = res['searched_string']
                        new_val = val[0].replace(searched, found)
                        val[0] = new_val
                    # replacing ocr result with correct result
                    for res in result_country:
                        found = res['found_string']
                        searched = res['searched_string']
                        new_val = val[0].replace(searched, found)
                        val[0] = new_val
                    for res in result_country:
                        if (str(k) + "-country") not in lookup_result:
                            lookup_result[('LUT_' + str(k) + "-country")] = []
                        lookup_result[('LUT_' + str(k) + "-country")].append(
                            (res['found_string'], res['string_match_value'], res["bbox"]))
                    for res in result_company:
                        if (str(k) + "-organization") not in lookup_result:
                            lookup_result[('LUT_' + str(k) + "-organization")] = []
                        lookup_result[('LUT_' + str(k) + "-organization")].append(
                            (res['found_string'], res['string_match_value'], res["bbox"]))
        # print(lookup_result)
        print("Result File Path : ", os.path.join(result_path, file[:-4] + str(count) + "_lookup.txt"))
        with open(os.path.join(result_path, file[:-4] + str(count) + "_lookup.txt"), "w") as f:
            json.dump(lookup_result, f)
        f.close()
        print("\n\nFinal Result Set : ")
        print(final_result_set)

        with open(os.path.join(result_path, file[:-4] + str(count) + ".txt"), "w") as f:
            json.dump(final_result_set, f)
        
        ext_result = {} ; b_box_result = {} ; conf_result = {}
        for key, value in final_result_set.items():
            if key != 'O':
                text = []
                b_box = []
                confs = []
                for tests in value:
                    print("Label : ", key)
                    text_info = merge_words_in_bbox(word_coordinates, tests[1].copy())
                    print("prev :", tests[0])
                    print("curr :", text_info)
                    box = tests[1]
                    b_box = b_box + [box]
                    value = text_info if len(text_info) > 0 else tests[0]
                    text = text + [value]
                    if not isinstance(tests[2], list):
                        confs = confs + [tests[2]]
                    else:
                        confs = confs + tests[2]

                ext_result.update({key: text})
                b_box_result.update({key: b_box})
                conf_result.update({key: confs})

        print("ext_result :", ext_result)
        idp_structure_result = {file: {model_type: {"keys_extraction": ext_result, "keys_bboxes": b_box_result, "keys_confidence": conf_result}}}
        with open(os.path.join(result_path, file[:-4] + ".json"), "w") as f:
            json.dump(idp_structure_result, f, indent=4)
        
        # del image
        # del new_img
        # del encoded_inputs       
        
        f.close()
        # new_img.save(os.path.join(result_path, file[:-4] + str(count) + ".png"))
        new_img.save(os.path.join(result_path, file[:-4] + ".png"))
    with open(os.path.join(result_path, file[:-4] + "all_page_result.txt"), "w") as f:
        json.dump(all_page_result, f)
    f.close()
    im.close()
    # print(f"*****Generated All Pages****** for {file}" )
    print(f'RAM memory % a used for {file} Generating All Pages:', psutil.virtual_memory()[2])
    t_total_end = datetime.now()
    print(f"Processing time taken for all pages of {file}" + str(count) + ":" + str(t_total_end - t_total_start))

    gc.collect()
    return all_page_result	



if __name__ == "__main__":

    # Setting configuration for logging purposes   
    #####################################################################
    process_memory = psutil.Process()
    start_time = datetime.now()
    cpu_utilization_start = psutil.cpu_percent()
    before_memory = process_memory.memory_info().rss
    # print("checkpoint 1 => setting basis cofiguration for logging purposes")
    # exit("+++++++++++++++++++")
    #####################################################################  


    ######################################################################
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--path', type =str, required = False, 
    #                     default='Images', help = "provide the folder name")
    # parser.add_argument('-p', '--image', type =str, required = False, 
    #                     help = "provide the image path")
    #######################################################################

    #######################################################################
    # args = parser.parse_args()
    # product config
    # product_config = ConfigParser()
    # product_config.read("src/main/extraction/config/config.ini")

    # data folder path
    # product_wise_folder = ConfigParser()
    # product_wise_folder.read("src/main/extraction/config/prod.ini")
    
    # data_folder_path = "/home/khushal/Desktop/Projects/POC_CODES/lmv2_training_code/ingram_invoice_keys_extraction_training_data_corrected" 
    # model_path = "/home/khushal/Downloads/Model_Jun22_40_epochs"
    # model = LayoutLMv2ForTokenClassification.from_pretrained(pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'), config=os.path.join(model_path, 'config.json'), from_tf=True)
    
    # data_folder_path = "/home/khushal/Desktop/data_n_models/data/Invoice_Non_Tabular_extraction_data_7.2k_images/Invoice-Data/"
    
    ### EDIT THESE VARIABLES ###
    root_data_folder_path= "/media/gpuadmin/New Volume/mani/grasim_poc/autolabelling/ingram_data/ingram_layoutlm_data_without_processor/ingram_test_data"
    image_folder_name = "invoice_ingram_apollo_test_image_data"
    # ocr_data_path = "/media/gpuadmin/New Volume/mani/grasim_poc/grasim_67_vendors_ocr"
    ocr_data_path = "/media/gpuadmin/New Volume/mani/grasim_poc/autolabelling/ingram_data/ingram_layoutlm_data_without_processor/ingram_test_data/ingram_test_data_ocr"
    # model_path = '/media/gpuadmin/New Volume/mani/grasim_poc/autolabelling/data/Model_40_epochs' #  ritwik, model path for idx2label
    # model_path = "/media/gpuadmin/New Volume/mani/grasim_poc/autolabelling/data/data_iter_finetune/Model_40_epochs"
    model_path = "/media/gpuadmin/New Volume/mani/grasim_poc/autolabelling/ingram_data/ingram_layoutlm_data/Model_40_epochs"
    classes_path= "/media/gpuadmin/New Volume/mani/grasim_poc/autolabelling/ingram_data/ingram_layoutlm_data/classes.txt"
    # config_path= "/media/gpuadmin/New Volume/mani/grasim_poc/autolabelling/layoutlm_v1_training_test_cases/layoutlm_test_case_1/Model_40_epochs/config.json"
    output_folder_name = "test_inference"
    ############################

    model_type = "layoutLMV2ForTokenClassification"
    is_image = False
    image_file = "IM-000000011379345-AP_page_1.png" 
    # output_folder_name = "Results_"+str(datetime.now())

    
    
    print("==================Extraction Inference===================")
    # print(f"folder_path: {folder_path}")
    #######################################################################

    # setting up some initial variables
    count = 0

    print("Loading Model...")
    t_start = datetime.now()
    print(f"start time: {t_start}")
    cpu_utilization_start = psutil.cpu_percent()
    before_memory = process_memory.memory_info().rss

    # model_path = glob.glob(f"{folder_path}/Best_Mode*")
    # print(f"Model path: {model_path}") ; assert len(model_path) == 1

    # model_path = model_path[0]
    # model_path= os.path.join(folder_path, 'Best_Mode')#"/home/ntlpt19/Downloads/MERGED_DATA/AIR_WAY/AIRWAY_v2/Best_Mode_Airway"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # model_path = '/home/khushal/Desktop/data_n_models/Models/invoice_extraction/lmv2_aug_16/layoutLMV2ForTokenClassification_b4_best_e7.pth'
    # model_path = '/home/khushal/Desktop/data_n_models/Models/invoice_extraction/lmv2_aug_16/layoutLMV2ForTokenClassification_b4_e18.pth'
    # model_path = '/home/khushal/Downloads/Best_Model/layoutLMV2ForTokenClassification_b4_outer_model.pth'
    # labels= read_classes(classes_path)
    # label2id, id2label = create_label_mappings(labels)
    # print(f"label2id  from classes.txt: {label2id}")
    # print(f"id2label from classes.txt: {id2label}")
    # Label2id, id2Label= get_label_mappings(config_path)
    # print(f"label2id  from config.json: {label2id}")
    # print(f"id2label from config.json: {id2label}")
    
    # model = torch.load(model_path, map_location=device)
    # model = LayoutLMv2ForTokenClassification.from_pretrained(model_path, use_safetensors=True)
    model_id= "microsoft/layoutlm-base-uncased"
    # model = LayoutLMForTokenClassification.from_pretrained(model_id, num_labels=len(labels), label2id=label2id, id2label=id2label)
    model = LayoutLMForTokenClassification.from_pretrained(model_path, use_safetensors=True)
    # model = LayoutLMv2ForTokenClassification.from_pretrained("/home/khushal/Downloads/pretrained_models/huggingface/hub/models--microsoft--layoutlmv2-base-uncased/snapshots/ae6f4350c668f88ec580046e35c670df6ec616c1") #, config="/home/khushal/Downloads/pretrained_models/huggingface/hub/models--microsoft--layoutlmv2-base-uncased/snapshots/ae6f4350c668f88ec580046e35c670df6ec616c1/config.json")
    # model = LayoutLMv2ForTokenClassification.from_pretrained(model_path,ignore_mismatched_sizes=True)
    # label_to_index = {'S-taxable_amount': 0, 'S-vendor_name': 1, 'S-invoice_date': 2, 'S-delivery_challan_no': 3, 'S-page_number': 4, 'S-customer_no': 5, 'S-swift_code': 6, 'O': 7, 'S-order_number': 8, 'S-charge_type': 9, 'S-cash_discount': 10, 'S-net_amount': 11, 'S-tax_percent': 12, 'S-order_date': 13, 'S-account_number': 14, 'S-vendor_vat_no': 15, 'S-doc_curr': 16, 'S-purchase_order_number': 17, 'S-customer_order_number': 18, 'S-invoice_total': 19, 'S-bill_to': 20, 'S-sales_order_number': 21, 'S-remit_to': 22, 'S-due_date': 23, 'S-freight_amount': 24, 'S-vat_code': 25, 'S-ship_to': 26, 'S-amount': 27, 'S-invoice_number': 28, 'S-tax_amount': 29, 'S-ship_date': 30}
    # index_to_label = {0: 'S-taxable_amount', 1: 'S-vendor_name', 2: 'S-invoice_date', 3: 'S-delivery_challan_no', 4: 'S-page_number', 5: 'S-customer_no', 6: 'S-swift_code', 7: 'O', 8: 'S-order_number', 9: 'S-charge_type', 10: 'S-cash_discount', 11: 'S-net_amount', 12: 'S-tax_percent', 13: 'S-order_date', 14: 'S-account_number', 15: 'S-vendor_vat_no', 16: 'S-doc_curr', 17: 'S-purchase_order_number', 18: 'S-customer_order_number', 19: 'S-invoice_total', 20: 'S-bill_to', 21: 'S-sales_order_number', 22: 'S-remit_to', 23: 'S-due_date', 24: 'S-freight_amount', 25: 'S-vat_code', 26: 'S-ship_to', 27: 'S-amount', 28: 'S-invoice_number', 29: 'S-tax_amount', 30: 'S-ship_date'}
    # model.config.id2label = index_to_label
    # model.config.label2id = label_to_index
    
    model.to(device)

    print("Model loaded successfully !!!")
    print("Model Label ID mapping : ",model.config.label2id)
    # exit('+++++++++++++')
    # processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", apply_ocr=False)
    # processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", revision="no_ocr")
    # processor= custom_processor
    processor= LayoutLMProcessor()
    device = 'cpu'
    model.to(device)

    t_end = datetime.now()
    cpu_utilization_end = psutil.cpu_percent()
    after_memory = process_memory.memory_info().rss
    cpu_utt = cpu_utilization_end - cpu_utilization_start
    memory_consumption = after_memory - before_memory
    print("Time Taken for loading Model:", str(t_end - t_start))
    print('RAM memory for Model loading used: % a', psutil.virtual_memory()[2])
    print(f"cpu_utilization % for Model Loading:{str(cpu_utt)}")
    print(
        f"memory_consumption in bytes for Loading Model:{str(memory_consumption)}"
    )
    print("Loaded Model successfully")
    # if args.image:
    
    
    print("Time Taken:", t_end - t_start)

    if is_image:
        img_name = list(image_file.split('/'))[-1]
        file_extension = os.path.splitext(img_name)[1]
        print(f"file_extension: {file_extension}")
        if file_extension == '.pdf':
            print('It is a pdf !!!')
            print("Skipping the document !!")
            # ans = pdf_result(image_file, image_folder_name, model, processor, device)
        else:
            ans = image_result(image_file, image_folder_name, model, processor, device, root_data_folder_path, output_folder_name)
        count += 1
    else:
        print("Time Taken :", t_end - t_start)

        for img_file in os.listdir(os.path.join(root_data_folder_path, image_folder_name)):
            # if "IM-000000011379458-AP_page_5" not in img_file:continue
            # if "IM-000000011379458-AP_page_5" not in img_file:continue
            file_extension = os.path.splitext(img_file)[1]
            print(f"file_extension: {file_extension}")

            if file_extension == '.pdf':
                print('It is a pdf !!!')
                # ans = pdf_result(img_file, image_folder_name, model, processor, device)
            else:
                ans = image_result(img_file, image_folder_name, model, processor, device, root_data_folder_path, output_folder_name)
            count += 1
            # exit()


    print("Total files processed:", count)
    print("***************Processed all the files!****************")