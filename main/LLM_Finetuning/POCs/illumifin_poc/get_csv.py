import os
import json
import pandas as pd

def json_to_csv(results_folder, output_csv):
    data_list = []
    
    # Iterate through all JSON files in the folder
    for file_name in os.listdir(results_folder):
        if file_name.endswith(".json"):
            file_path = os.path.join(results_folder, file_name)
            
            with open(file_path, "r") as f:
                data = json.load(f)
                print(data)
                data['pdf_name'] = data['pdf_name']+'.pdf'
                data_list.append(data)
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data_list)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved: {output_csv}")

# Example usage
results_folder = "/datadrive/rakesh/illumifin_poc/Data/results"  # Replace with the actual folder path
output_csv = "output.csv"
json_to_csv(results_folder, output_csv)
