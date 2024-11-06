"""
* *********************************************************************************
* Number Theory S/W Pvt. Ltd CONFIDENTIAL                                      *
* *
* [2016] - [2023] Number Theory S/W Pvt. Ltd Incorporated                       *
* All Rights Reserved.                                                          *
* *
* NOTICE:  All information contained herein is, and remains                     *
* the property of Number Theory S/W Pvt. Ltd Incorporated and its suppliers,    *
* if any.  The intellectual and technical concepts contained                    *
* herein are proprietary to Number Theory S/W Pvt. Ltd Incorporated             *
* and its suppliers and may be covered by India. and Foreign Patents,           *
* patents in process, and are protected by trade secret or copyright law.       *
* Dissemination of this information or reproduction of this material            *
* is strictly forbidden unless prior written permission is obtained             *
* from Number Theory S/W Pvt. Ltd Incorporated.                                 *
* *
* *********************************************************************************
"""
import json
import numpy as np
import pytesseract
from PIL import Image


class ApplyOcr:
    def __init__(self):
        self.apply_ocr()

    def apply_ocr(self):
        def normalize_box(box, width, height):
            return [
                int(1000 * (box[0] / width)),
                int(1000 * (box[1] / height)),
                int(1000 * (box[2] / width)),
                int(1000 * (box[3] / height)),
            ]

        # get the image
        c = 0
        image = Image.open(self['image_path'])
        name = self['image_path'].split("/")[-1].split(".")[0]
        width, height = image.size

        # apply ocr to the image
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)

        # get the words and actual (unnormalized) bounding boxes
        # words = [word for word in ocr_df.text if str(word) != 'nan'])
        words = list(ocr_df.text)
        words = [str(w) for w in words]
        coordinates = ocr_df[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row)  # the row comes in (left, top, width, height) format
            actual_box = [x, y, x + w,
                          y + h]  # we turn it into (left, top, left+width, top+height) to get the actual box
            actual_boxes.append(actual_box)

        # normalize the bounding boxes
        boxes = []
        for box in actual_boxes:
            boxes.append(normalize_box(box, width, height))

        ocr_data = {}
        # Iterate over the OCR words and boxes
        for word, box in zip(words, boxes):
            box_list = [int(val) for val in box]  # Convert tuple to list
            ocr_data[word] = box_list
        # Convert dictionary to JSON
        json_data = json.dumps(ocr_data)
        return json_data
