import json

# Function to calculate the intersection percentage between two bounding boxes
def get_intersection_percentage(bb1, bb2):
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']
    
    # Determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate the intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # Determine which box is smaller
    if bb1_area > bb2_area:
        intersection_percent = intersection_area / bb2_area
    else:
        intersection_percent = intersection_area / bb1_area
        if intersection_percent < 0.5:
            intersection_percent = 1.0  # Treat as full if < 0.5 overlap for larger box
    
    assert 0.0 <= intersection_percent <= 1.0
    return intersection_percent

# Main function to process data and generate output
def parse_data_and_generate_output(data1_file, word_coordinates_file, output_file):
    # Open the file in read mode ('r')
    with open(data1_file, 'r') as file:
        # Read the contents of the file
        data1 = eval(file.read())
        
    # Print the contents
    # Load data from the provided JSON files
    # with open(data1_file, 'r') as f:
    #     data1 = json.load(f)
        
    # with open(word_coordinates_file, 'r') as f:
    #     word_coordinates = json.load(f)
    
    with open(word_coordinates_file, 'r') as file:
        # Read the contents of the file
        word_coordinates = eval(file.read())
    print(word_coordinates['word_coordinates'])
    output_lines = []
    index = 1

    # Define helper function to format data into the required output format
    def format_output(index, left, top, right, bottom, text, label):
        return f"{index},{left},{top},{right},{top},{right},{bottom},{left},{bottom},{text},{label}"
    
    # Store all bounding boxes from data1 in a list for easier comparison
    data1_bounding_boxes = []
    for key, values in data1.items():
        for value in values:
            text = value[0]
            coords = value[1]
            data1_bounding_boxes.append({
                'text': text,
                'label': key,
                'x1': coords[0],
                'y1': coords[1],
                'x2': coords[2],
                'y2': coords[3]
            })
    
    # Process word coordinates to generate output lines
    for word_data in word_coordinates['word_coordinates']:
        text = word_data['word']
        left = word_data['left']
        top = word_data['top']
        right = word_data['x2']
        bottom = word_data['y2']
        
        # Create a bounding box for the word from word_coordinates
        word_bb = {
            'x1': left,
            'y1': top,
            'x2': right,
            'y2': bottom
        }
        
        # Determine if this word is in a data1 bounding box
        is_covered = False
        for data1_bb in data1_bounding_boxes:
            # Calculate intersection percentage
            iou = get_intersection_percentage(data1_bb, word_bb)
            if iou > 0.5:
                # If the intersection is above the threshold, use the label from data1
                output_lines.append(format_output(index, left, top, right, bottom, text, data1_bb['label']))
                is_covered = True
                index += 1
                break
        
        if not is_covered:
            # If the word does not match any bounding box from data1, label as "other"
            output_lines.append(format_output(index, left, top, right, bottom, text, "other"))
            index += 1
    
    # Write the output to the file
    with open(output_file, 'w') as f:
        f.write("\n".join(output_lines))

# Specify input and output file paths
data1_file = '/home/ntlpt19/Downloads/pick_model_train/BOE/Master_Data/Accepted_Bill_of_Exchange(2012_08_21_15_47_02_7475)_368_5_labels.txt'
word_coordinates_file = '/home/ntlpt19/Downloads/pick_model_train/BOE/Master_Data/Accepted_Bill_of_Exchange(2012_08_21_15_47_02_7475)_368_5_text.txt'
output_file = 'output.txt'
################################################################
################################################################
#need to merge the tokens if they are nearer , use the logic from the LMV2 data perparation 
###############################################################
###############################################################
parse_data_and_generate_output(data1_file, word_coordinates_file, output_file)
