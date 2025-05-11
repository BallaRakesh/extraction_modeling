import os

pi_mapping_old = {
    0: 22,
    1: 23,
    2: 49,
    3: 75,
    4: 10,
    5: 12,
    6: 1,
    7: 3,
    8: 38,
    9: 39,
    10: 76,
    11: 40,
    12: 17,
    13: 18,
    14: 14,
    15: 15,
    16: 7,
    17: 8,
    18: 4,
    19: 5,
    20: 77,
    21: 78,
    22: 31,
    23: 32,
    24: 79,
    25: 33,
    26: 25,
    27: 24,
    28: 30,
    29: 29,
    30: 80,
    31: 81,
    32: 82,
    33: 11,
    34: 51,
    35: 54,
    36: 55,
    37: 52,
    38: 53,
    39: 83,
    40: 84,
    41: 85,
    42: 69,
    43: 34,
    44: 35,
    45: 61,
    46: 86,
    47: 68,
    48: 69,
    49: 87,
    50: 43,
    51: 88,
    52: 89,
    53: 90,
    54: 91,
    55: 92,
    56: 93,
    57: 94,
    58: 95,
    59: 96,
    60: 92,
    61: 97,
    62: 50,
    63: 62,
    64: 64,
    65: 98
}


pi_mapping = {
    4:2,
    5:3,
    2:4,
    3:5,
    8:7,
    7:8
}

#old_key : new_key
awb_mapping = {
    2:1,
    3:1,
    5:41,
}

ic_mapping = {
  52: 48,
  53: 49,
  54: 50,
  55: 51,
  56: 52,
  57: 53,
  58: 54,
  59: 55,
  60: 56,
  61: 57,
  62: 58,
  63: 59,
  64: 60,
  65: 61,
  66: 62,
  67: 63,
  68: 64
}


bol_mapping = {
    39: 42,
    41: 39
}

def label_mapping(labels_folder, actual_mapping):
    all_labels = os.listdir(labels_folder)

    for lbs_ in all_labels:
        print('processing file:', lbs_)
        file_path = os.path.join(labels_folder, lbs_)

        with open(file_path, 'r') as file_:
            label_ = file_.read()

        label_lines = label_.split("\n") 

        modified_lines = []
        for line in label_lines:
            parts = line.split()
            if len(parts) > 0:
                l_class = int(parts[0])
                if l_class in actual_mapping:
                    l_class = actual_mapping[l_class]
                    modified_line = f"{l_class} {' '.join(parts[1:])}"
                    modified_lines.append(modified_line)
                else:
                    modified_lines.append(line)
        # Write the modified content back to the file
        with open(file_path, 'w') as file_:
            file_.write("\n".join(modified_lines))
            

if __name__ == '__main__':              
    labels_folder = '/home/ntlpt19/Downloads/Evaluation_Data/finalEvaluationEvalData_itter3/BEFORE/BOL/Labels'
    label_mapping(labels_folder, bol_mapping)
    