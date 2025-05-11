

import pandas as pd
import os
import json
import csv

root_directory = '/home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData_itter2/AWB'
original_labels = os.path.join(root_directory, 'New_Master_Data_Merged')
predicted_labels = os.path.join(root_directory, 'Results_Images')
labels_path = os.path.join(root_directory, 'label.txt')
excel_path = '/home/ntlpt19/Downloads/Evaluation_Data/FinalEvaluationEvalData_itter2/AWB/result_path/label_wise/awb_2024-03-14_1/final_report_awb_pre_2024-03-14 01:46:59.080171_after_fuzzy_match_change.csv'

org_info = {}
pred_info = {}

for org_lb in os.listdir(original_labels):
    print(org_lb)
    with open(os.path.join(original_labels, org_lb), "r") as f:
        original_ = json.load(f)
    for keys, val in original_.items():
        if keys not in org_info:
            org_info[keys] = 1
        else:
            org_info[keys] += 1
print(org_info)

exit('......................................')
for org_lb in os.listdir(original_labels):
    print(org_lb)
    with open(os.path.join(original_labels, org_lb), "r") as f:
        original_ = json.load(f)
    file_name_ = org_lb.split('.')[0]
    file_name_ = file_name_.replace('_labels', '')
    file_name_ = file_name_+'1.txt'
    if os.path.exists(os.path.join(predicted_labels, file_name_)):
        with open(os.path.join(predicted_labels, file_name_), "r") as f:
            predictions = json.load(f)
            
        for keys, val in predictions.items():
            if keys not in pred_info:
                pred_info[keys] = 1
            else:
                pred_info[keys] += 1
        
        for keys, val in original_.items():
            if keys not in org_info:
                org_info[keys] = 1
            else:
                org_info[keys] += 1

print(org_info)
print(pred_info)

combined_data = [['Field', 'Original Count', 'Predicted Count', 'verify_from_report_original', 'verify_from_report_predicted']]
master_keys = []

# Iterate over keys and add data to the combined_data list
for key in org_info.keys():
    master_keys.append(key)
    combined_data.append([key, org_info[key], pred_info.get(key, 0)])

for key in pred_info.keys():
    if key not in master_keys:
        master_keys.append(key)
        combined_data.append([key, 0, pred_info[key]])

# Load data from the Excel sheet
excel_data = pd.read_csv(excel_path)
print(excel_data)

# Iterate over rows in the Excel sheet and update combined_data
for index, row in excel_data.iterrows():
    label_name = row['Label_Name']
    label_count = row['Label_Count']
    labels_detected = row['Labels_Detected']

    # Find the corresponding entry in combined_data and update it
    for entry in combined_data:
        if entry[0] == label_name:
            entry.extend([label_count, labels_detected])


with open(labels_path, 'r') as labels_file:
    labels = [line.strip() for line in labels_file]

# Check if each label is in combined_data, and if not, add it with counts set to 0
for label in labels:
    if label not in [entry[0] for entry in combined_data]:
        combined_data.append([label, 0, 0, 0, 0])

# Write the updated data to a CSV file
csv_file_path = os.path.join(os.path.dirname(predicted_labels), 'combined_counts.csv')
with open(csv_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerows(combined_data)

print(f"CSV file '{csv_file_path}' created successfully.")


