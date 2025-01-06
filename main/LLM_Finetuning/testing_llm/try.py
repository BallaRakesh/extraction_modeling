
import os
import json
import csv
# from rapidfuzz import fuzz
from fuzzywuzzy import fuzz


# field = 'Status Perkawinan'
field_value = 'status_perkawinan'

with open('/home/ntlpt19/Desktop/TF_release/extraction_modeling/main/LLM_Finetuning/testing_llm/pred_data/4.json', 'r') as pred_file:
    predicted_data = json.load(pred_file)

for field in predicted_data:
    if field.lower() == field_value.lower():
        print(predicted_data[field])
    # if field.lower() == field_value.lower().replace('_', ' '):
    if field.lower().replace('_', '').replace(' ', '') == field_value.lower().replace('_', '').replace(' ', ''):
        print(predicted_data[field])
    else:
        print(' ')
