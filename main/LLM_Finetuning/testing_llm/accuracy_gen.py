import os
import json
import csv
# from rapidfuzz import fuzz
from fuzzywuzzy import fuzz


def get_predicted_value(predicted_data, field_value, file_name):
    field_value = str(field_value)
    for field in predicted_data:
        if field.lower() == field_value.lower():
            return predicted_data[field]
        if field.lower().replace('_', '').replace(' ', '') == field_value.lower().replace('_', '').replace(' ', ''):
            return predicted_data[field]
    return ""
    


def calculate_fuzzy_scores(gt_folder, predicted_folder, output_csv):
    results = []
    
    # Get list of all JSON files in the ground truth folder
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.json')]

    for file_name in gt_files:
        print('file_name:', file_name)
        gt_path = os.path.join(gt_folder, file_name)
        pred_path = os.path.join(predicted_folder, file_name)
        
        # Check if the corresponding predicted JSON file exists
        if not os.path.exists(pred_path):
            print(f"Warning: Predicted file {file_name} not found.")
            continue
        
        # Load JSON data
        with open(gt_path, 'r') as gt_file:
            gt_data = json.load(gt_file)
        with open(pred_path, 'r') as pred_file:
            pred_data = json.load(pred_file)
        
        # Iterate over each field in the ground truth JSON
        for field_name, gt_value in gt_data.items():
            # Get the predicted value for the same field (if exists)
            
            # predicted_value = pred_data.get(field_name, "")
            predicted_value = get_predicted_value(pred_data, field_name, file_name)
            # Calculate fuzzy score between gt_value and predicted_value
            fuzzy_score = fuzz.ratio(str(gt_value).lower(), str(predicted_value).lower())
            print('field_name', field_name, '>>>>>>>>>>', predicted_value)
            # Append result to the results list
            results.append({
                "image_name": file_name,
                "field_name": field_name,
                "gt_value": gt_value,
                "predicted_value": predicted_value,
                "fuzzy_score": fuzzy_score
            })
    
    # Save results to CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ["image_name", "field_name", "gt_value", "predicted_value", "fuzzy_score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_csv}")

# Example usage
gt_folder = "/home/ntlpt19/Desktop/TF_release/extraction_modeling/main/LLM_Finetuning/testing_llm/gt_data"
predicted_folder = "/home/ntlpt19/Desktop/TF_release/extraction_modeling/main/LLM_Finetuning/testing_llm/pred_data"
output_csv = "fuzzy_score_results.csv"
calculate_fuzzy_scores(gt_folder, predicted_folder, output_csv)
