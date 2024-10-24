"""
Description:

This program takes the OCR data generated by the AWS textract, and also the 
already annotated data (in YOLO format). 

Primary goal here is to clean out wrong key-value pairs from textract OCR data, 
and also including the key-value paris that the textract failed to detect. 

Finally, the output is generated in FUNSD-like format and resultant data 
is dumped in this path -> preprocess/custom/custom_data/data_in_funsd_format 
"""

from copy import copy
import os
import json
from functools import cmp_to_key
from tqdm import tqdm
from PIL import Image, ImageDraw
import shutil
from typing import Dict, List

from glob import glob

import imagesize
from transformers import BertTokenizer
from typing import Dict


def denormalize(h, w, bbox, denom=2):
    """
    Get entire label coordinate region
    
    Parameters
    ----------
    h: int
       heigth of the image
    w: int
       width of the image
    bbox: list
        word coordinates obtained from ocr
    denom: int
        Denominator for denormalize operation

    Returns:
    x0, y0, x1, y1: tuple
        tuple of denormalized word coordinates
    """

    x_center = float(bbox[0]) * w
    y_center = float(bbox[1]) * h
    width = float(bbox[2]) * w
    height = int(float(bbox[3]) * h)
    x0 = int(x_center - (width / denom))
    x1 = int(x_center + (width / denom))
    y0 = int(y_center - (height / denom))
    y1 = int(y_center + (height / denom))

    return x0, y0, x1, y1

def calculate_iou(bbox1, bbox2):
    """
    Finds the percentage of intersection  with a smaller box. (what percernt of smaller box is in larger box)
    """
    # assert bbox1['x1'] < bbox1['x2']
    # assert bbox1['y1'] < bbox1['y2']
    # assert bbox2['x1'] < bbox2['x2']
    # assert bbox2['y1'] < bbox2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    # min_area = min(bbox1_area,bbox2_area)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    intersection_percent = intersection_area / bbox2_area

    return intersection_percent

def contour_sort(a, b):
	if abs(a['y1'] - b['y1']) <= 15:
		return a['x1'] - b['x1']

	return a['y1'] - b['y1']

def get_area(bbox):
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def get_other_text(words):
    """
    1. Track the already covered words here
    2. Difference of all words and Covered words = Others 
    """
    pass

def get_text(ocr_region, labelled_region, words_coords=None, words=None, all_words=None):
    if get_area(labelled_region) > get_area(ocr_region):
        print(f'area of labelled region: {get_area(labelled_region)}')
        print(f'area of ocr region : {get_area(ocr_region)}')
        # exit('+++++++++++')
        coords = []
        for idx, item in all_words.items():
            #if abs(item['bbox'][0] - labelled_region[0]) < 250:
                iou = calculate_iou(item['bbox'], labelled_region)
                if iou > 0.0:
                    coords.append({'bbox' : item['bbox'],
                                   'word': item['text']})
        # coords = sorted(coords, key=cmp_to_key(contour_sort))  
        # print(f"coords: {coords}")
        # exit('+++++++++++++++++')
        words = [item['word'] for item in coords]
        cords = [item['bbox'] for item in coords]
        assert len(words) == len(cords)
        return words, cords
    else:
        assert len(words_coords) == len(words)
        print('Else condition ++++++++++++++++++')
        print(f'area of labelled region: {get_area(labelled_region)}')
        print(f'area of ocr region : {get_area(ocr_region)}')
        res = []
        coords = []
        print(f'word corred: {words_coords}')
        print(f'words : {words}')
        # exit('+++++++++++++')
        for i in range(len(words_coords)):
            iou = calculate_iou(labelled_region, words_coords[i])
            #print(iou)
            if iou > 0.4:
                res.append(words[i])
                coords.append(words_coords[i])
        return res, coords


def preprocessData(json_file:Dict, dataset_split:str, input_path, file:str, data_folder=None, MAX_SEQ_LENGTH = 512, MODEL_TYPE = "bert", 
                    VOCA = "bert-base-uncased", anno_dir="annotations"):
    tokenizer = BertTokenizer.from_pretrained(VOCA, do_lower_case=True)
    OUTPUT_PATH = os.path.join(input_path, "dataset/custom_geo")
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "preprocessed"), exist_ok=True)
    if dataset_split == "train":
        dataset_root_path = os.path.join(input_path, "training_data")
    elif dataset_split == "test":
        dataset_root_path = os.path.join(input_path, "testing_data")
    elif dataset_split == "val":
        dataset_root_path = os.path.join(input_path, "validation_set")
        if not os.path.exists(dataset_root_path):
            os.makedirs(dataset_root_path)
        try:
            if not os.path.exists(os.path.join(dataset_root_path,'images')):
                shutil.copytree(os.path.join(input_path,'images'), os.path.join(dataset_root_path,'images'))
                shutil.copytree(os.path.join(input_path,'annotations'), os.path.join(dataset_root_path,'annotations'))
        except Exception as e:
            print(e)

    else:
        raise ValueError(f"Invalid dataset_split={dataset_split}")

    # json_files = glob(os.path.join(dataset_root_path, anno_dir, "*.json"))
    preprocessed_fnames = []
    # for json_file in tqdm(json_files):
    in_json_obj = json_file
    print('&*10')
    # print(in_json_obj)
    # exit('++++++++++++++++')

    out_json_obj = {}
    out_json_obj['blocks'] = {'first_token_idx_list': [], 'boxes': []}
    out_json_obj["words"] = []
    form_id_to_word_idx = {} # record the word index of the first word of each block, starting from 0
    other_seq_list = {}
    num_tokens = 0

    # words
    for form_idx, form in enumerate(in_json_obj):
        print(form)
        # exit('+++++++++++++++')
        form_id = form["id"]
        # form_text = form["text"].strip()
        # form_label = form["label"]
        # if form_label.startswith('O') or form_label.startswith('o'):
        #     form_label = "O"
        # form_linking = form["linking"]
        form_box = form["box"]

        # if len(form_text) == 0:
        #     continue # filter text blocks with empty text

        word_cnt = 0
        class_seq = []
        real_word_idx = 0
        for word_idx, word in enumerate(form["words"]):
            word_text = word["text"]

            print(f'word text: {word_text}')
            # exit('+++++++++++++++')
            bb = word["box"]
            # form_box=bb
            print(bb)
            # exit('++++++++++++++')
            bb = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]
            tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word_text))
            print(f'input id token:{tokens}')
            # exit('+++++++++++++')

            word_obj = {"text": word_text, "tokens": tokens, "boundingBox": bb}
            if len(word_text) != 0: # filter empty words
                out_json_obj["words"].append(word_obj)
                if real_word_idx == 0:
                    out_json_obj['blocks']['first_token_idx_list'].append(num_tokens + 1)
                num_tokens += len(tokens)

                word_cnt += 1
                class_seq.append(len(out_json_obj["words"]) - 1) # word index
                real_word_idx += 1
        if real_word_idx > 0:
            out_json_obj['blocks']['boxes'].append(form_box)
    # meta
    out_json_obj["meta"] = {}
    image_file = (
        os.path.join(dataset_root_path,'images',file+ ".png")
    )
    print(image_file)
    if not os.path.exists(image_file):       # if image has no ocr files currently skipping
        return True

    # exit('+++++++++++++++')
    if dataset_split == "train":
        out_json_obj["meta"]["image_path"] = image_file[
            image_file.find("training_data/") :
        ]
    elif dataset_split == "test":
        out_json_obj["meta"]["image_path"] = image_file[
            image_file.find("testing_data/") :
        ]
    elif dataset_split == "val":
        out_json_obj["meta"]["image_path"] = image_file[
            image_file.find("validation_set/") :
        ]
    
    width, height = imagesize.get(image_file)
    out_json_obj["meta"]["imageSize"] = {"width": width, "height": height}
    out_json_obj["meta"]["voca"] = VOCA

    this_file_name = file+'.json'

    # # Save file name to list
    preprocessed_fnames= os.path.join("preprocessed", this_file_name)
    print(preprocessed_fnames)
    # exit('+++++++++++++++++++')

    # Save to file
    data_obj_file = os.path.join(OUTPUT_PATH, "preprocessed", this_file_name)
    with open(data_obj_file, "w", encoding="utf-8") as fp:
        json.dump(out_json_obj, fp, ensure_ascii=False)

    # Save file name list file
    preprocessed_filelist_file = os.path.join(
        OUTPUT_PATH, f"preprocessed_files_{dataset_split}.txt"
    )
    with open(preprocessed_filelist_file, "a", encoding="utf-8") as fp:
        fp.write(preprocessed_fnames+"\n")

def __preprocess__(data:dict, file:str,data_folder:str, train_images_path:str, thresh: int):
        print('Enter into preprocess stage+++++++++++++++++++++++++++++++++')
        final_data=[]
        for i, item in enumerate(data):
            final_data.append(data[item])
            if  (i+1)%thresh==0:
                file_name= file + "_s_" + str(int((i + 1) / thresh))
                shutil.copy2(os.path.join(train_images_path,file+'.png'),os.path.join(f'{data_folder}/images',file_name+'.png'))
                print(f"final_data :{final_data}")
                with open(os.path.join(os.path.join(f'{data_folder}/annotations',file_name+'.json')), 'w') as f:
                    json.dump({"form" : final_data}, f, indent=4)
                final_data.clear()
            if len(final_data)!=0:
                file_name= file + "_s_" + str(int(len(data) /thresh) + 1)
                shutil.copy2(os.path.join(train_images_path,file+'.png'),os.path.join(f'{data_folder}/images',file_name+'.png')) 
                with open(os.path.join(os.path.join(f'{data_folder}/annotations',file_name+'.json')), 'w') as f:
                    json.dump({"form" : final_data}, f, indent=4)
def findOtherCategory(word_box: List, key_box:List, value_box: List):
    print(f'word box: {word_box}')
    print(f'key box: {key_box}')
    print(f'value box: {value_box}')
    # exit('++++++++++++++++')
    for i in range(len(key_box)):
        iou1 = calculate_iou(value_box[i], word_box)
        iou2 = calculate_iou(key_box[i], word_box)
        if iou1 and iou2> 0.4:
            return False
        else:
            return True
def main():
    root_path = "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/test_code/Validation_data/test"
    ocr_path = os.path.join(root_path, "custom_data/key_val_sets")
    all_words_path = os.path.join(root_path, "custom_data/all_words")
    save_to = os.path.join(root_path, "complete_inference_files/validation_set")
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    os.makedirs(os.path.join(save_to, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_to, 'annotations'), exist_ok=True)
    images_path = os.path.join(root_path, 'Images')
    images_list = os.listdir(images_path)
    print(images_list)
    images_list= [images.split('.png')[0] for images in images_list]
    for file in images_list:
        # ocr_path= os.path.join(ocr_path, file+'.json')
        with open(os.path.join(all_words_path, file+'.json'), 'r')   as f:
            all_words = json.load(f)
        with open(os.path.join(ocr_path, file+'.json') , 'r') as f:
            ocr_labels = json.load(f)
        key_dict= {}
        val_dict={}
        key_box=[]
        val_box= []
        value_cntr= len(ocr_labels)
        key_cntr=0
        for i, ocr_coord in enumerate(ocr_labels):
            print(f'ocr coordinate: {ocr_coord}')
            print(len(ocr_labels))
            # exit('++++++++++++++++')
            val_bbox = [int(item) for item in ocr_coord['value_bbox']]
            key_bbox=  [int(item) for item in ocr_coord['key_bbox']]
            print(val_bbox)
            print(key_bbox)
            # exit('+++++++++++++')
            if not len(val_bbox) == len(key_bbox) == 0:
                print('Enters++++++++++++++++++++++++++=')
                key_dict.update({key_cntr : { 
                                    'id' : key_cntr ,
                                    'box': ocr_coord['key_bbox'],
                                    'text': ' '.join(ocr_coord['key_text']),
                                    'words' : [{'text': ocr_coord['key_text'][i], 
                                                'box':ocr_coord['key_text_bbox'][i] }
                                                for i in range(len(ocr_coord['key_text']))],
                                    }})

                val_dict.update({
                            value_cntr : {
                                'id' : value_cntr,
                                'box': ocr_coord['value_bbox'],
                                'text': ocr_coord['value_text'],
                                'words' : [{'text': ocr_coord['value_text'][i], 
                                            'box':ocr_coord['value_bbox'] }
                                            for i in range(len((ocr_coord['value_text'])))]
                        }})
                print(f"the key box is appending here: {ocr_coord['key_bbox']}")
                print(f"the key box is appending here: {ocr_coord['value_bbox']}")
                key_box.append(ocr_coord['key_bbox'])
                val_box.append(ocr_coord['value_bbox'])
                # exit('+++++++++++++=')

                key_cntr += 1
                value_cntr += 1
        # merging two dicts
        key_dict.update(val_dict)
        other_contr= len(ocr_labels)*2
        print(f'number of keys: {len(key_box)}')
        print(f'number of  values: {len(val_box)}')
        # exit('++++++++++====')
        for i, ocr_coord in enumerate(all_words):
            print(ocr_coord)
            print(all_words[ocr_coord])
            text = all_words[ocr_coord]['text']
            bbox = all_words[ocr_coord]['bbox']
            if (findOtherCategory(word_box=bbox, key_box=key_box, value_box= val_box)):
                print(f'text: {text}, box: {bbox}')
                key_dict.update({other_contr : { 
                                        'id' : other_contr,
                                        'box': all_words[ocr_coord]['bbox'],
                                        'words' : [{'text': all_words[ocr_coord]['text'], 
                                                    'box':all_words[ocr_coord]['bbox']}]
                                        }})
                other_contr+=1
        # print(key_dict)
        print(f'the number of elements: {len(key_dict)}')
        if len(key_dict)<=150:
            print(f'the elements less than 150+++++++++++++++++++++++++++++')
            shutil.copy2(os.path.join(images_path, file+'.png'), os.path.join(save_to, 'images'))
            final_data = [key_dict[item] for item in key_dict]
            with open(os.path.join(save_to,'annotations',file+'.json'), 'w') as f:
                json.dump({"form" : final_data}, f, indent=4)
        else:
            __preprocess__(key_dict, file, save_to, images_path, 150)
if __name__=="__main__":
    main()
    preproces_root_path= "/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/test_code/Validation_data/test/complete_inference_files"
    preprocess_input_path= os.path.join(preproces_root_path,'validation_set')
    images_path= os.path.join(preprocess_input_path,'images')
    annotation_path= os.path.join(preprocess_input_path,'annotations')
    images_list = os.listdir(images_path)
    print(images_list)
    images_list= [images.split('.png')[0] for images in images_list]
    for file in images_list:
        with open(os.path.join(annotation_path, file+'.json'), 'r')   as f:
            data = json.load(f)['form']
            preprocessData(data,'val',preproces_root_path,file)
    print('########## Done ################')
    
    


