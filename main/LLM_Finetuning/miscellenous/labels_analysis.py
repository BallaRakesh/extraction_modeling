import os
import ast
import csv

# Step 1: List all files in the current directory ending with _labels

master_data_path = '/home/ntlpt19/LLM_training/TRAIN/PL/Master_Data'
save_path = os.path.dirname(master_data_path)
files = [f for f in os.listdir(master_data_path) if f.endswith('_labels.txt')]

print(files)

# Initialize a dictionary to count the keys
key_counts = {}

# Step 2: Process each file
for file in files:
    with open(os.path.join(master_data_path, file), 'r') as f:
        content = f.read()
        # Convert the text content into a dictionary
        data = ast.literal_eval(content)
        print(data)
        for key in data.keys():
            if key in key_counts:
                key_counts[key] += 1
            else:
                key_counts[key] = 1

# Step 3: Write the key counts to a CSV file
csv_file = f'{save_path}/key_counts.csv'
with open(csv_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Key', 'Count'])
    for key, count in key_counts.items():
        writer.writerow([key, count])

print(f"Key counts written to {csv_file}")
