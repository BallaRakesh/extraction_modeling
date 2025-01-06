import json


import ast
ignore_classifier = True
source_num_labels = 50
if not ignore_classifier and source_num_labels is not None:

    print('YES')
exit("OK")
# Specify the path to your text file
file_path = '/home/data_science/geo_testing/COO_V3/CORD_DATA/Eval_data/OCR/image_81_text.txt'

# Open and read the file
with open(file_path, 'r') as file:
    data = file.read()
print(data)
# Parse the content into a Python object (list of dictionaries)
try:
    parsed_data = ast.literal_eval(data)  # Safely evaluate the string as a Python literal
    print("Parsed Data:", parsed_data)
except Exception as e:
    print("Error parsing file:", e)



with open('/home/data_science/geo_testing/COO_V3/CORD_DATA/Eval_data/OCR/image_81_text.txt', "r") as f:
    word_coordinates = json.load(f)#['word_coordinates']