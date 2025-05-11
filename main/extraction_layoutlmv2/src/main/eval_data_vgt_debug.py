import numpy as np

def merge_bboxes_and_text(words, bboxes):
    """
    Groups words and their bounding boxes line by line, merging those in the same line.
    Returns a list of merged words and merged bounding boxes.
    """
    if not words or not bboxes:
        return []

    # Sorting bounding boxes by their y-coordinate to group words in the same line
    sorted_indices = sorted(range(len(bboxes)), key=lambda i: bboxes[i][1])
    
    merged_lines = []
    current_line = {"words": [], "bboxes": []}
    last_y = None

    for i in sorted_indices:
        word, bbox = words[i], bboxes[i]
        y_center = (bbox[1] + bbox[3]) / 2  # Calculate the middle Y value of the box

        if last_y is None or abs(y_center - last_y) <= 10:  # Threshold to determine same line
            current_line["words"].append(word)
            current_line["bboxes"].append(bbox)
        else:
            merged_lines.append({
                "words": " ".join(current_line["words"]),
                "bbox": [
                    min(b[0] for b in current_line["bboxes"]),  # x_min
                    min(b[1] for b in current_line["bboxes"]),  # y_min
                    max(b[2] for b in current_line["bboxes"]),  # x_max
                    max(b[3] for b in current_line["bboxes"]),  # y_max
                ]
            })
            current_line = {"words": [word], "bboxes": [bbox]}  # Start a new line group

        last_y = y_center

    # Append the last merged line
    if current_line["words"]:
        merged_lines.append({
            "words": " ".join(current_line["words"]),
            "bbox": [
                min(b[0] for b in current_line["bboxes"]),  
                min(b[1] for b in current_line["bboxes"]),  
                max(b[2] for b in current_line["bboxes"]),  
                max(b[3] for b in current_line["bboxes"]),  
            ]
        })

    return merged_lines

# Sample JSON data
data = {
    "shipper_name": [
        [
            "GOLDLIINE WORLDWIDE INC",
            [121, 152, 472, 173],
            {
                "ind_words": ["GOLDLIINE", "WORLDWIDE", "INC"],
                "ind_bbox": [
                    [125, 148, 260, 171],
                    [274, 152, 407, 174],
                    [427, 156, 467, 176]
                ]
            }
        ]
    ],
    "shipper_address": [
        [
            "P.O.BOX 8443 , SAIF ZONE SHARJAH U.A.E.",
            [122, 177, 458, 223],
            {
                "ind_words": [
                    "P.O.BOX", "8443", ",", "SAIF", "ZONE", "SHARJAH", "U.A.E."
                ],
                "ind_bbox": [
                    [124, 175, 226, 194],
                    [246, 177, 299, 195],
                    [308, 178, 312, 194],
                    [320, 178, 376, 196],
                    [395, 179, 452, 197],
                    [124, 200, 226, 217],
                    [273, 203, 357, 219]
                ]
            }
        ]
    ]
}

# Process each key in the dictionary
for key, values in data.items():
    for item in values:
        text, bbox, details = item
        words, bboxes = details["ind_words"], details["ind_bbox"]
        
        # Merge words and bounding boxes into lines
        line_info = merge_bboxes_and_text(words, bboxes)
        
        # Add to the existing dictionary
        details["line_info"] = line_info
        print('line_info: ', line_info, '\n\n')
print(details)
# # Print updated JSON structure
# import json
# print(json.dumps(data, indent=2))
