import re
import json
import os
import pandas as pd
import os


def extract_key_value_pairs(input_text):
    pairs = {}
    lines = input_text.strip().split('\n')
    current_key = None
    current_value = ""

    for line in lines:
        line = line.strip().strip('"')
        match = re.match(r'(.+?):\s*(.+?)$', line)

        if match:
            if current_key:
                pairs[current_key] = current_value.strip()
            current_key, current_value = match.groups()
            current_key = current_key.strip('"')
            current_value = current_value.strip('"').rstrip(',')
        elif current_key:
            current_value += " " + line.strip('"').rstrip(',')

    if current_key:
        pairs[current_key] = current_value.strip()

    return pairs


# Function to extract JSON-like content from a text using regex
def extract_json_from_text(text):
    # Regex pattern to find content within {}
    # pattern = r'\{(?:[^{}]|(?R))*\}'
    pattern = r'\{(?:[^{}]|(?:(?R)))*\}'
    # Find all matches
    matches = re.findall(pattern, text)
    # Assuming there's exactly one JSON-like object in the file
    if matches:
        # Extract the first match (assuming it's the JSON portion)
        json_content = matches[0]
        # Return the extracted JSON string
        return json_content
    else:
        return None


def extract_json_only_keyvalue(content):
    # Read the file content

    pattern = r'"([\w_]+)":\s*"([^"]*)"'
    
    # Find all matches
    matches = re.findall(pattern, content, re.MULTILINE)
    
    # Create a dictionary from the matches
    result = {key: value for key, value in matches}
    
    return result



def extract_json_content(content):
    # Read the file content

    # Extract all JSON-like content
    json_pattern = r'\{([^{}]+)\}'
    matches = re.findall(json_pattern, content, re.DOTALL)
    
    results = []
    for match in matches:
        # Extract key-value pairs
        pair_pattern = r'"([\w_]+)":\s*"?([^",\n]+)"?'
        pairs = re.findall(pair_pattern, match)
        
        # Create a dictionary from the pairs
        result = {key: value.strip() for key, value in pairs}
        if result:  # Only add non-empty dictionaries
            results.append(result)
    
    return results




# Function to parse JSON from string
def parse_json_string(json_str):
    try:
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# Read text file
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

def correct_response_creation(query):
        '''
        First we will extract the required key value pairs from the given query and then, send in a perfect JSON format query.
        '''
        pair_pattern = r'"([\w_]+)":\s*"?([^",\n]+)"?'
        pairs = re.findall(pair_pattern, query)
        # Create a dictionary from the pairs
        
        # result = ["{"]
        # for key,value in pairs:
        #     result.append(f'{key}:{value.strip()}')    
        # result.append("}")
        # res = "\n".join(result)
        # print("The final response generated is:",res)
        
        
        result_dict = {}
        for key, value in pairs:
            result_dict[key] = value#.strip()

        # Convert the dictionary to a JSON string
        json_str = json.dumps(result_dict, indent=4)

        print("The final response generated is:", json_str)
        return json_str
        


# Example usage
if __name__ == "__main__":
    root_folder = '/home/ntlpt19/LLM_training/EVAL/COO'
    results_folder = os.path.join(root_folder, 'results/text_files')
    save_folder = os.path.join(root_folder, 'Results_Images')
    csv_file_path = os.path.join(root_folder, 'results.csv')
    os.makedirs(save_folder, exist_ok=True)
    df = pd.read_csv(csv_file_path)
    for index, row in df.iterrows():
        file_name = row['file name'] 
        print(file_name)
        if pd.notna(file_name):
            file_path = os.path.join(results_folder, file_name+'.txt')
            if not pd.isnull(row['gt_len']) and not pd.isnull(row['prediction_count']):
                if int(row['gt_len'])*0.8 < int(row['prediction_count']):
                    file_content = read_file(file_path)
                    print(file_content)
                    # Extract JSON-like content from text
                    json_str = extract_key_value_pairs(file_content)
                    print(json_str)
                    # input_dict = json.loads(json_str)
                    output = {}

                    # Define dummy bounding box and confidence score
                    dummy_bbox = [0, 0, 0, 0]
                    dummy_confidence = 90.0

                    # Populate the output dictionary with details
                    for key, value in json_str.items():
                        output[key] = [[value, dummy_bbox, dummy_confidence]]


                    # Open the file in write mode
                    # with open(os.path.join(save_folder, file_name+'.txt'), 'w') as file:
                    #     # Write the content to the file
                    #     file.write(str(json_str))
                        
                    with open(os.path.join(save_folder, file_name+'1.txt'), "w") as f:
                        json.dump(output, f)
                    f.close()
            
                    # with open(os.path.join(save_folder, file_name+'.json'), 'w') as json_file:
                    #     json.dump(output, json_file, indent=4)

                    # if json_str:
                    #     # Parse JSON string to JSON object
                    #     json_data = parse_json_string(json_str)
                    #     if json_data:
                    #         print("Parsed JSON data:")
                    #         print(json_data)
                    #     else:
                    #         print("Failed to parse JSON.")
                    # else:
                    #     print("No JSON-like content found in the file.")
