import os



labels_folder = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/verified_tarun/po_org_472/Labels'
all_labels = os.listdir(labels_folder)
pi_mapping = {}


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
            if l_class == 42:
                exit('>>>>>>>>>>>>>>>>>>>>')
            if l_class in pi_mapping:
                l_class = pi_mapping[l_class]
                modified_line = f"{l_class} {' '.join(parts[1:])}"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)
    # Write the modified content back to the file
    with open(file_path, 'w') as file_:
        file_.write("\n".join(modified_lines))
        
    