# -*- coding: utf-8 -*-
import os
import pickle
from functools import lru_cache
import pytesseract
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import re
import ast
import cv2
import json

PAD_TOKEN_BOX = [0, 0, 0, 0]
GRID_SIZE = 1000


def normalize_box(box, width, height, size=1000):
    """
    Takes a bounding box and normalizes it to a thousand pixels. If you notice it is
    just like calculating percentage except takes 1000 instead of 100.
    """
    return [
        int(size * (box[0] / width)),
        int(size * (box[1] / height)),
        int(size * (box[2] / width)),
        int(size * (box[3] / height)),
    ]


@lru_cache(maxsize=10)
def resize_align_bbox(bbox, orig_w, orig_h, target_w, target_h):
    x_scale = target_w / orig_w
    y_scale = target_h / orig_h
    orig_left, orig_top, orig_right, orig_bottom = bbox
    x = int(np.round(orig_left * x_scale))
    y = int(np.round(orig_top * y_scale))
    xmax = int(np.round(orig_right * x_scale))
    ymax = int(np.round(orig_bottom * y_scale))
    return [x, y, xmax, ymax]


def get_topleft_bottomright_coordinates(df_row):
    left, top, width, height = df_row["left"], df_row["top"], df_row["width"], df_row["height"]
    return [left, top, left + width, top + height]


def apply_ocr(image_fp):
    """
    Returns words and its bounding boxes from an image
    """
    image = Image.open(image_fp)
    width, height = image.size

    ocr_df = pytesseract.image_to_data(image, output_type="data.frame")
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    float_cols = ocr_df.select_dtypes("float").columns
    ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
    ocr_df = ocr_df.replace(r"^\s*$", np.nan, regex=True)
    ocr_df = ocr_df.dropna().reset_index(drop=True)
    words = list(ocr_df.text.apply(lambda x: str(x).strip()))
    actual_bboxes = ocr_df.apply(get_topleft_bottomright_coordinates, axis=1).values.tolist()
    # draw_word_coordinates(image_fp, actual_bboxes, out_fol, words_given = words)
    # add as extra columns
    assert len(words) == len(actual_bboxes)
    return {"words": words, "bbox": actual_bboxes}

# ocr_gv = ''
out_fol = '/home/ntlpt19/Downloads/Classification_final_training/debug'

ocr_gv = '/home/ntlpt19/Downloads/Classification_final_training/OCR_GV'
split_ocr_folder = '/home/ntlpt19/Downloads/Classification_final_training/V4_ROOT/LC/ocr_chunks'
ocr_pytess = '/home/ntlpt19/Downloads/Classification_final_training/V4_ROOT/LC/ocr_pytess'
ocr_gv_json = ''

def get_ocr_tesseract(image_, file_name):
    """
    Performs OCR (Optical Character Recognition) using Tesseract OCR engine.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing word coordinates (list of dictionaries) and all the extracted text (str).

    """
    # img=None
    word_coordinates, all_text = [],""
    # print("called Image OCR...", end="")
    # try:
    # img = Image.open(image_path)
    d = pytesseract.image_to_data(image_, output_type=pytesseract.Output.DICT)
    all_text = pytesseract.image_to_string(image_)
    for i in range(len(d['text'])):
        word = d['text'][i]
        # word = d['text'][i].strip()
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
    # print('pytesseract function', word_coordinates)

    with open(os.path.join(ocr_pytess, file_name), 'w') as file:
        file.write(str({'word_coordinates':word_coordinates, 'all_text':all_text}))
    file.close()
    return word_coordinates, all_text




# Function to draw word coordinates on an image
def draw_word_coordinates(image_path, word_coordinates, output_folder, default_cords_flag = True, words_given = None):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    # Loop through word coordinates and draw rectangles
    i = 0
    debug_word_dict = []
    for coord in word_coordinates:
        
        if default_cords_flag:
            x1, y1, x2, y2 = coord[0], coord[1], coord[2], coord[3]
            word = words_given[i]
            i = i + 1
            debug_word_dict.append({'word': word,'coordinates': coord})
        else:
            x1, y1, x2, y2 = coord["x1"], coord["y1"], coord["x2"], coord["y2"]
            word = coord["word"]
        # confidence = coord.get("confidence", 90.8)#"confidence"]
        
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box with thickness 2
        
        # Put the word and confidence near the box
        text = f"{word}"# ({confidence:.2f})"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Prepare output path
    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    output_path_wc_cord = os.path.join(output_folder, image_name[:-4]+'_wc.txt')
    
    with open(output_path_wc_cord, 'w') as file1:  
        file1.write(str(debug_word_dict))  # write each coordinate on a new line
    file1.close()
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the image
    cv2.imwrite(output_path, image)
    print(f"Saved image with drawn coordinates to: {output_path}")







def apply_ocr_gv(image_fp):
    # print(image_fp)
    file_name = os.path.basename(image_fp)
    print('####################>', 'START')
    print('####################>', 'START')
    print('####################>', 'START')
    print(file_name)
    image = Image.open(image_fp)
    width, height = image.size
    # if custom_pytesseract:
    # print(img_name_)
    file_path_gv = os.path.join(ocr_gv, file_name[:-4]+'_text.txt')
    # file_path_pytes = os.path.join(ocr_pytess, file_name[:-4]+'_text.txt')
    file_path_json = os.path.join(ocr_gv_json, file_name[:-4]+'.json')
    if os.path.exists(file_path_json):
        # Open and read the JSON file  
        with open(file_path_json, 'r') as file:  
            content = json.load(file) 
        file.close()
        # wc_data = ast.literal_eval(content)
        word_coordinates = content.get('word_coordinates', [])
        # print('came to if block !!!!')
    
    if os.path.exists(file_path_gv):
        with open(file_path_gv, 'r') as file:
            content = file.read()
            wc_data = ast.literal_eval(content)
        file.close()
        if isinstance(wc_data, dict):
            word_coordinates = wc_data.get('word_coordinates', [])
            print('came to if block !!!!')
    
    # elif os.path.exists(file_path_pytes):
    #     with open(file_path_pytes, 'r') as file:
    #         content = file.read()
    #         wc_data = ast.literal_eval(content)
    #     file.close()
    #     if isinstance(wc_data, dict):
    #         word_coordinates = wc_data.get('word_coordinates', [])
    #         print('came to if block !!!!')
    
    else:
        print('came to else block')
        word_coordinates, _ = get_ocr_tesseract(image, file_name[:-4]+'_text.txt')

    words = []
    actual_bboxes = []


    for wrd_ in word_coordinates:
        word_ = wrd_['word'].strip()
        if word_:
            words.append(wrd_['word'])
            # actual_bboxes.append([wrd_['x1'], wrd_['y1'], wrd_['x2'], wrd_['y2']])
            # def get_topleft_bottomright_coordinates(df_row):
            left, top, width, height = wrd_["left"], wrd_["top"], wrd_["width"], wrd_["height"]
            actual_bboxes.append([left, top, left + width, top + height])
            
    # draw_word_coordinates(image_fp, actual_bboxes, out_fol, words_given = words)
    assert len(words) == len(actual_bboxes)
    # assert len(words) != 0
    return {"words": words, "bbox": actual_bboxes}


def get_ocr_split_files(img_files):
    ocr_file_name = os.path.join(split_ocr_folder, f"{os.path.basename(img_files)[0:-4]}.json")
    with open(ocr_file_name, 'r') as file:
        data = json.load(file)
    file.close()
    return data
    
def get_chunk_words_recursive(chunk_number, chunk_size, words, bounding_box):
    start_index = (chunk_number - 1) * chunk_size
    end_index = start_index + chunk_size
    chunk_words = words[start_index:end_index]
    chunk_bounding_box = bounding_box[start_index:end_index]
    if not len(chunk_words) and chunk_number > 0:
        return get_chunk_words_recursive(chunk_number - 1, chunk_size, words, bounding_box)
    return chunk_words, chunk_bounding_box

def get_tokens_with_boxes(unnormalized_word_boxes, pad_token_box, word_ids,max_seq_len = 512):
    
    # assert len(unnormalized_word_boxes) == len(word_ids), this should not be applied, since word_ids may have higher 
    # length and the bbox corresponding to them may not exist
    
    unnormalized_token_boxes = []
    
    for i, word_idx in enumerate(word_ids):
        if word_idx is None:
            break
        unnormalized_token_boxes.append(unnormalized_word_boxes[word_idx])

    # all remaining are padding tokens so why add them in a loop one by one
    num_pad_tokens = len(word_ids) - i - 1
    if num_pad_tokens > 0:
        unnormalized_token_boxes.extend([pad_token_box] * num_pad_tokens)
        
        
    if len(unnormalized_token_boxes)<max_seq_len:
        unnormalized_token_boxes.extend([pad_token_box] * (max_seq_len-len(unnormalized_token_boxes)))
        
    return unnormalized_token_boxes[:max_seq_len] ## maybe in case the length is higher than max_seq_len


def get_centroid(actual_bbox):
    centroid = []
    for i in actual_bbox:
        width = i[2] - i[0]
        height = i[3] - i[1]
        centroid.append([i[0] + width / 2, i[1] + height / 2])
    return centroid


def get_pad_token_id_start_index(words, encoding, tokenizer): 
#     assert len(words) < len(encoding["input_ids"])  This condition, was creating errors on some sample images
    for idx in range(len(encoding["input_ids"])):
        if encoding["input_ids"][idx] == tokenizer.pad_token_id:
            break
    return idx


def get_relative_distance(bboxes, centroids, pad_tokens_start_idx):

    a_rel_x = []
    a_rel_y = []

    for i in range(0, len(bboxes)-1):
        if i >= pad_tokens_start_idx:
            a_rel_x.append([0] * 8)
            a_rel_y.append([0] * 8)
            continue

        curr = bboxes[i]
        next = bboxes[i+1]

        a_rel_x.append(
            [
                curr[0],  # top left x
                curr[2],  # bottom right x
                curr[2] - curr[0],  # width
                next[0] - curr[0],  # diff top left x
                next[0] - curr[0],  # diff bottom left x
                next[2] - curr[2],  # diff top right x
                next[2] - curr[2],  # diff bottom right x
                centroids[i+1][0] - centroids[i][0],
            ]
        )

        a_rel_y.append(
            [
                curr[1],  # top left y
                curr[3],  # bottom right y
                curr[3] - curr[1],  # height
                next[1] - curr[1],  # diff top left y
                next[3] - curr[3],  # diff bottom left y
                next[1] - curr[1],  # diff top right y
                next[3] - curr[3],  # diff bottom right y
                centroids[i+1][1] - centroids[i][1],
            ]
        )

    # For the last word
    
    a_rel_x.append([0]*8)  
    a_rel_y.append([0]*8)


    return a_rel_x, a_rel_y
     


def apply_mask(inputs, tokenizer):
    inputs = torch.as_tensor(inputs)
    rand = torch.rand(inputs.shape)
    # where the random array is less than 0.15, we set true
    mask_arr = (rand < 0.15) * (inputs != tokenizer.cls_token_id) * (inputs != tokenizer.pad_token_id)
    # create selection from mask_arr
    selection = torch.flatten(mask_arr.nonzero()).tolist()
    # apply selection pad_tokens_start_idx to inputs.input_ids, adding MASK tokens
    inputs[selection] = 103
    return inputs


def read_image_and_extract_text(image):
    original_image = Image.open(image).convert("RGB")
    return apply_ocr(image)


def create_features_org(
        image,
        tokenizer,
        add_batch_dim=False,
        target_size=(500,384),  # This was the resolution used by the authors
        max_seq_length=512,
        path_to_save=None,
        save_to_disk=False,
        apply_mask_for_mlm=False,
        extras_for_debugging=False,
        use_ocr = True,
        bounding_box = None,
        words = None
):

    # step 1: read original image and extract OCR entries
    try:   
        original_image = Image.open(image).convert("RGB")
    except:
        original_image = Image.new(mode = "RGB", size = ((500, 500)), color = (255, 255, 255))
        
    if (use_ocr == False) and (bounding_box == None or words == None):
        raise Exception('Please provide the bounding box and words or pass the argument "use_ocr" = True')

    if use_ocr == True:
    #   entries = apply_ocr(image)
      entries = apply_ocr_gv(image)
      bounding_box = entries["bbox"]
      words = entries["words"]

    CLS_TOKEN_BOX = [0, 0, *original_image.size]    # Can be variable, but as per the paper, they have mentioned that it covers the whole image
    # step 2: resize image
    resized_image = original_image.resize(target_size)

    # step 3: normalize image to a grid of 1000 x 1000 (to avoid the problem of differently sized images)
    width, height = original_image.size
    normalized_word_boxes = [
        normalize_box(bbox, width, height, GRID_SIZE) for bbox in bounding_box
    ]
    assert len(words) == len(normalized_word_boxes), "Length of words != Length of normalized words"

    # step 4: tokenize words and get their bounding boxes (one word may split into multiple tokens)
    encoding = tokenizer(words,
                         padding="max_length",
                         max_length=max_seq_length,
                         is_split_into_words=True,
                         truncation=True,
                         add_special_tokens=False)
    
    unnormalized_token_boxes = get_tokens_with_boxes(bounding_box,
                                                                  PAD_TOKEN_BOX,
                                                                  encoding.word_ids())

    # step 5: add special tokens and truncate seq. to maximum length
    unnormalized_token_boxes = [CLS_TOKEN_BOX] + unnormalized_token_boxes[:-1]
    # add CLS token manually to avoid autom. addition of SEP too (as in the paper)
    encoding["input_ids"] = [tokenizer.cls_token_id] + encoding["input_ids"][:-1]

    # step 6: Add bounding boxes to the encoding dict
    encoding["unnormalized_token_boxes"] = unnormalized_token_boxes
   
    # step 7: apply mask for the sake of pre-training
    if apply_mask_for_mlm:
        encoding["mlm_labels"] = encoding["input_ids"]
        encoding["input_ids"] = apply_mask(encoding["input_ids"], tokenizer)
        assert len(encoding["mlm_labels"]) == max_seq_length, "Length of mlm_labels != Length of max_seq_length"
       
    assert len(encoding["input_ids"]) == max_seq_length, "Length of input_ids != Length of max_seq_length"
    assert len(encoding["attention_mask"]) == max_seq_length, "Length of attention mask != Length of max_seq_length"
    assert len(encoding["token_type_ids"]) == max_seq_length, "Length of token type ids != Length of max_seq_length"

    # step 8: normalize the image
    encoding["resized_scaled_img"] = ToTensor()(resized_image)

    # step 9: apply mask for the sake of pre-training
    if apply_mask_for_mlm:
        encoding["mlm_labels"] = encoding["input_ids"]
        encoding["input_ids"] = apply_mask(encoding["input_ids"], tokenizer)

    # step 10: rescale and align the bounding boxes to match the resized image size (typically 224x224)
    resized_and_aligned_bboxes = []

    for bbox in unnormalized_token_boxes:
        # performing the normalization of the bounding box
        resized_and_aligned_bboxes.append(resize_align_bbox(tuple(bbox), *original_image.size, *target_size))

    encoding["resized_and_aligned_bounding_boxes"] = resized_and_aligned_bboxes
    
    # step 11: add the relative distances in the normalized grid
    bboxes_centroids = get_centroid(resized_and_aligned_bboxes)
    pad_token_start_index = get_pad_token_id_start_index(words, encoding, tokenizer)
    a_rel_x, a_rel_y = get_relative_distance(resized_and_aligned_bboxes, bboxes_centroids, pad_token_start_index)

    # step 12: convert all to tensors
    for k, v in encoding.items():
        encoding[k] = torch.as_tensor(encoding[k])

    encoding.update({
        "x_features": torch.as_tensor(a_rel_x, dtype=torch.int32),
        "y_features": torch.as_tensor(a_rel_y, dtype=torch.int32),
        })

    # step 13: add tokens for debugging
    if extras_for_debugging:
        input_ids = encoding["mlm_labels"] if apply_mask_for_mlm else encoding["input_ids"]
        encoding["tokens_without_padding"] = tokenizer.convert_ids_to_tokens(input_ids)
        encoding["words"] = words


    # step 14: add extra dim for batch
    if add_batch_dim:
        encoding["x_features"].unsqueeze_(0)
        encoding["y_features"].unsqueeze_(0)
        encoding["input_ids"].unsqueeze_(0)
        encoding["resized_scaled_img"].unsqueeze_(0)

    # step 15: save to disk
    if save_to_disk:
        os.makedirs(path_to_save, exist_ok=True)
        image_name = os.path.basename(image)
        with open(f"{path_to_save}{image_name}.pickle", "wb") as f:
            pickle.dump(encoding, f)

    # step 16: keys to keep, resized_and_aligned_bounding_boxes have been added for the purpose to test if the bounding boxes are drawn correctly or not, it maybe removed
    
    keys = ['resized_scaled_img', 'x_features','y_features','input_ids','resized_and_aligned_bounding_boxes']
    
    if apply_mask_for_mlm:
        keys.append('mlm_labels')
    
    final_encoding = {k:encoding[k] for k in keys}
    
    del encoding
    return final_encoding



def create_features(
        image,
        tokenizer,
        add_batch_dim=False,
        target_size=(500, 384),  # This was the resolution used by the authors
        max_seq_length=512,
        path_to_save=None,
        save_to_disk=False,
        apply_mask_for_mlm=False,
        extras_for_debugging=False,
        use_ocr=True,
        bounding_box=None,
        words=None
):
    def process_features(image_pth, tokenizer, apply_ocr_func, using_function = 1, verify_feature= False, percentage_val = 0.05,  words = [], bounding_box=[]):
        image = re.sub(r'_S_\d+$', '', os.path.splitext(image_pth)[0])
        image = f"{image}.png"
        original_image = Image.open(image).convert("RGB")
        print('Current image :', image)
        if not verify_feature:
            # Step 1: Read the original image and extract OCR entries
            match = re.search(r'S_(\d+)\.png', image_pth)
            chunk_size = 250
            chunk_number = ''
            if match:
                chunk_number = match.group(1)
                print(chunk_number)

            if not use_ocr and (bounding_box is None or words is None):
                raise Exception('Please provide the bounding box and words or pass the argument "use_ocr" = True')
            # try:
                # original_image = Image.open(image).convert("RGB")
            
            # except:
                # original_image = Image.new(mode="RGB", size=(500, 500), color=(255, 255, 255))

            if use_ocr:
                entries = apply_ocr_func(image_pth)
                bounding_box = entries["bbox"]
                words = entries["words"]
                if using_function == 1 and chunk_number:
                    words, bounding_box = get_chunk_words_recursive(chunk_number, chunk_size, words, bounding_box)
                    
        else:
            num_words = len(words)
            print('@@@@@@@@@@@@ reduced words by 5%', num_words)
            num_words_to_keep = int(num_words * (1 - percentage_val))
            words, bounding_box = words[:num_words_to_keep], bounding_box[:num_words_to_keep]

            
        # Step 2: Normalize the image
        PAD_TOKEN_BOX = [0, 0, 0, 0]
        CLS_TOKEN_BOX = [0, 0, *original_image.size]  # Covers the whole image as per the paper
        resized_image = original_image.resize(target_size)

        width, height = original_image.size
        normalized_word_boxes = [
            normalize_box(bbox, width, height, GRID_SIZE) for bbox in bounding_box
        ]
        assert len(words) == len(normalized_word_boxes), "Length of words != Length of normalized words"

        # Tokenize words and get bounding boxes
        encoding = tokenizer(words,
                             padding="max_length",
                             max_length=max_seq_length,
                             is_split_into_words=True,
                             truncation=True,
                             add_special_tokens=False)

        unnormalized_token_boxes = get_tokens_with_boxes(
            bounding_box, PAD_TOKEN_BOX, encoding.word_ids()
        )
        unnormalized_token_boxes = [CLS_TOKEN_BOX] + unnormalized_token_boxes[:-1]
        encoding["input_ids"] = [tokenizer.cls_token_id] + encoding["input_ids"][:-1]

        # Add bounding boxes to the encoding dict
        encoding["unnormalized_token_boxes"] = unnormalized_token_boxes

        if apply_mask_for_mlm:
            encoding["mlm_labels"] = encoding["input_ids"]
            encoding["input_ids"] = apply_mask(encoding["input_ids"], tokenizer)
            assert len(encoding["mlm_labels"]) == max_seq_length, "Length mismatch for mlm_labels"

        assert len(encoding["input_ids"]) == max_seq_length, "Length mismatch for input_ids"
        assert len(encoding["attention_mask"]) == max_seq_length, "Length mismatch for attention mask"
        assert len(encoding["token_type_ids"]) == max_seq_length, "Length mismatch for token_type_ids"

        encoding["resized_scaled_img"] = ToTensor()(resized_image)

        resized_and_aligned_bboxes = [
            resize_align_bbox(tuple(bbox), *original_image.size, *target_size)
            for bbox in unnormalized_token_boxes
        ]
        encoding["resized_and_aligned_bounding_boxes"] = resized_and_aligned_bboxes

        bboxes_centroids = get_centroid(resized_and_aligned_bboxes)
        pad_token_start_index = get_pad_token_id_start_index(words, encoding, tokenizer)
        a_rel_x, a_rel_y = get_relative_distance(resized_and_aligned_bboxes, bboxes_centroids, pad_token_start_index)

        for k, v in encoding.items():
            encoding[k] = torch.as_tensor(encoding[k])

        encoding.update({
            "x_features": torch.as_tensor(a_rel_x, dtype=torch.int32),
            "y_features": torch.as_tensor(a_rel_y, dtype=torch.int32),
        })

        if extras_for_debugging:
            input_ids = encoding["mlm_labels"] if apply_mask_for_mlm else encoding["input_ids"]
            encoding["tokens_without_padding"] = tokenizer.convert_ids_to_tokens(input_ids)
            encoding["words"] = words

        if add_batch_dim:
            encoding["x_features"].unsqueeze_(0)
            encoding["y_features"].unsqueeze_(0)
            encoding["input_ids"].unsqueeze_(0)
            encoding["resized_scaled_img"].unsqueeze_(0)

        if save_to_disk:
            os.makedirs(path_to_save, exist_ok=True)
            image_name = os.path.basename(image)
            with open(f"{path_to_save}{image_name}.pickle", "wb") as f:
                pickle.dump(encoding, f)

        keys = ['resized_scaled_img', 'x_features', 'y_features', 'input_ids', 'resized_and_aligned_bounding_boxes']
        if apply_mask_for_mlm:
            keys.append('mlm_labels')
        final_encoding = {k: encoding[k] for k in keys}

        print('y_features MAX $$$$$$$$$', final_encoding["y_features"][:, 0].max().item())
        # if final_encoding["y_features"][:, 0].max().item() > 1023:
        #     final_encoding = process_features(image, tokenizer, apply_ocr, using_function = 1, verify_feature = True, words = words, bounding_box=bounding_box)
        return final_encoding



    # First pass: Process features using `apply_ocr_gv`
    # final_encoding = process_features(image, tokenizer, apply_ocr_gv)
    final_encoding = process_features(image, tokenizer, get_ocr_split_files, using_function = 0)
    # final_encoding = process_features(image, tokenizer, get_ocr_split_files, using_function = 0)

    if final_encoding["x_features"][:, 0].min().item() < 0:#x_feature[:,:,0].min().item()
        # Regenerate features using `apply_ocr`
        final_encoding = process_features(image, tokenizer, apply_ocr, using_function = 1)
        print('@@@@@@@@@@@@@@@@@@ came to the second Itteration')
    return final_encoding
