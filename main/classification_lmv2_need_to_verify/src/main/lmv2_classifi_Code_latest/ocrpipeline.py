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

# "______________________________________________Inference___________________________________________________"
# document = "/home/ntlpt60/work/just/resume/doc_000441.png"
# encoded_inputs = tokenizer(document, truncation=True, padding="max_length", max_length=512, return_tensors="pt")
# # Forward pass through the model
# outputs = model(encoded_inputs)
#
# # Get the predicted label
# predicted_label = outputs.logits.argmax(dim=1)
#
# # Convert the predicted label to a readable format
# predicted_label = predicted_label.item()
#
# # Map the predicted label to the corresponding document type
# document_type = None
# if predicted_label == 0:
#     document_type = "Scientific_publication"
# elif predicted_label == 1:
#     document_type = "Email"
# elif predicted_label == 2:
#     document_type = "Resume"
#
# print("Document Type:", document_type)

# print(" CODE EXECUTED SUCCESSFULLY___________________________!!!!!!!!!!!!!!!!!!!!")
