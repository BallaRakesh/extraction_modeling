import os

def process_txt_files(directory_path):
    key_file_mapping = {}  # Dictionary to store keys and their corresponding files

    # Iterate through all files in the directory
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(directory_path, file_name)
            
            with open(file_path, "r") as file:
                lines = file.readlines()
            
            for line in lines:
                parts = line.split()
                if parts:  # Ensure the line is not empty
                    key = parts[0]  # First element is the key
                    if key not in key_file_mapping:
                        key_file_mapping[key] = []  # Initialize list for this key
                    if file_name not in key_file_mapping[key]:
                        key_file_mapping[key].append(file_name)

    return key_file_mapping

# Example usage
directory_path = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/IC_816/IC/IC_816/Labels"
result = process_txt_files(directory_path)
print(result)
print(result.keys())
print(result['26'])
