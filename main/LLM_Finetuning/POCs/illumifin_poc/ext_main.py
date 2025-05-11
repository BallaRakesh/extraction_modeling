"""
source ~/Desktop/envs/gen_ai/bin/activate
python -m gen_ai_code.azure_cloud_services.azure_openai.gpt_4o_infer
"""

import re
import json
import openai
import os, re, base64

# from gen_ai_code.azure_cloud_services.azure_openai.data.illumifin_sample_inputs import  \
#     system_prompt, mds_1_Discharge_page_1_text, bea_1_page_1_text, bea_1_page_4_text, ccna_1_page_3_text


api_version = "2024-08-01-preview"
azure_endpoint = "https://ai-akshayamalik3400ai955768269900.openai.azure.com/openai/deployments/gpt-4o/chat/completions?2024-08-01-preview"
api_key = "7jlPMIpHvaEdpt53TE7dsXUbyrJEiVRR9uj1UXysqKhw3HKcSKzwJQQJ99BBACHYHv6XJ3w3AAAAACOGFJYF"

client = openai.AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)



system_prompt = """
You are an AI that extracts key-value pairs from text.

Note:
- If the field is not present or illegible, return "Not Found".
- Preserve the exact format and capitalization of the text within the values.
"""


def get_prompt(ocr_all_text):
    
    prompt_text = f"""
        You are an advanced language model with image analysis capabilities. I will provide you with an image containing text, and your task is to extract specific fields from it. The fields will appear as single values (standalone key-value pairs). Follow these rules:

            1. **Single Values**: If a field is present as a standalone key-value pair, extract its value(s). If a single value is found, return it as a string. If multiple values are found for the same key, return them as a list of strings (e.g., ["value1", "value2"]). If the key or its value is not found, return ["not found"].
            2. **Names**: For the 'names' field, extract all person names present in the image as a list of dictionaries. Each dictionary should have the keys 'first_name', 'middle_name', and 'last_name' (lowercase with underscores). If a single name is provided (e.g., "John Doe"), split it into components where possible: 'first_name' for the first part, 'middle_name' for any middle part (if present), and 'last_name' for the last part. If only one name is given (e.g., "John"), place it under 'first_name' and set 'middle_name' and 'last_name' to "not found". If multiple distinct person names are present, create separate dictionaries for each. Exclude non-person names (e.g., application names, company names) by identifying context or labels indicating a person (e.g., "Name", "Applicant", "Insured").
            3. **Output Format**: Return the result as a JSON object. For single-value fields, use the field name with underscores (e.g., 'policy_number') as the key and the value(s) as a string or list of strings. For the 'names' field, use 'names' as the key and provide a list of dictionaries with 'first_name', 'middle_name', and 'last_name' as keys.

        Here are the fields to extract:  
            1. Carrier  
            2. Policy Number  
            3. Names (containing first_name, middle_name, last_name)  
            4. DoB  
            5. Social Security Number  
            6. SubFolder /Department  
            
        Here is the OCR TEXT of the Image: {ocr_all_text}

        Tips to follow for extracting:  
            1. **Carrier**: Look for the exact phrase "Carrier" in the document. If not found, identify any insurance company name or any company name present in the image (e.g., "ABC Insurance", "XYZ Corp"). Extract the corresponding value(s) as a string or list. If no company name is found, set its value to ["not found"].  
            2. **Policy Number**: Locate the term "Policy Number" in the document. Extract the associated value(s) as a string or list if present; otherwise, return ["not found"].  
            3. **Names**: Identify text indicating a specific person name where the name is either provided alone (e.g., "John Smith") or prefixed with a label such as "Name", "First Name", "Last Name", "Applicant", or "Insured". Extract only the explicitly indicated name (not all person names in the image) into a list of dictionaries. For each identified name:
                - If labeled separately (e.g., "First Name: John", "Last Name: Smith"), use those values to populate 'first_name', 'middle_name' (if provided), and 'last_name'.
                - If a full name is given with a prefix (e.g., "Name: John A Smith"), split it into 'first_name': "John", 'middle_name': "A", 'last_name': "Smith".
                - If only one name is given with or without a prefix (e.g., "John" or "Name: John"), set 'first_name': "John", 'middle_name': "not found", 'last_name': "not found".
                - If the name is not explicitly labeled or isolated (e.g., part of a sentence or another context like a signature or unrelated text), ignore it.
                - Exclude non-person names like application names or company names by context (e.g., ignore "PolicyApp" or "Carrier: ABC Insurance").
                - Only include one name entry unless multiple names are explicitly labeled with the specified prefixes (e.g., "Applicant: John Smith" and "Insured: Jane Doe"). In such cases, create separate dictionaries: [{{"first_name": "John", "middle_name": "not found", "last_name": "Smith"}}, {{"first_name": "Jane", "middle_name": "not found", "last_name": "Doe"}}].
                - If no explicitly indicated person name is found, set 'names' to an empty list: [].
            4. **DoB**: Identify the text "DoB" and extract its corresponding value(s) as a string or list. If missing, set it to ["not found"].  
            5. **Social Security Number**: Identify the text "Social Security Number" and extract its corresponding value(s) as a string or list. If missing, set it to ["not found"].  
            6. **SubFolder /Department**: Identify the text "SubFolder /Department" and extract its corresponding value(s) as a string or list. If missing, set it to ["not found"].  

        Steps to follow:  
            1. Analyze the text content of the image.  
            2. For each field (except 'names'), extract the value(s) as a string or list, or assign ["not found"] if missing.  
            3. For 'names', identify all person names, split them into components where possible, and construct a list of dictionaries with 'first_name', 'middle_name', and 'last_name'. Exclude non-person names.  
            4. Return the result as a JSON object with keys using underscores.
        

        Example output with single values and one name:  
        ```json
        {{
        "carrier": ["ABC Insurance"],
        "policy_number": ["P123456"],
        "names": [
            {{"first_name": "John", "middle_name": "A", "last_name": "Smith"}}
        ],
        "dob": ["1985-05-15"],
        "social_security_number": ["123-45-6789"],
        "subfolder_department": ["Claims"]
        }}
        ```

        Example output with multiple names and missing fields:  
        ```json
        {{
        "carrier": ["not found"],
        "policy_number": ["P987654"],
        "names": [
            {{"first_name": "Jane", "middle_name": "not found", "last_name": "Doe"}},
            {{"first_name": "Robert", "middle_name": "B", "last_name": "Johnson"}}
        ],
        "dob": ["1990-10-20"],
        "social_security_number": ["not found"],
        "subfolder_department": ["not found"]
        }}
        ```

        Example output with a single name and no split:  
        ```json
        {{
        "carrier": ["XYZ Corp"],
        "policy_number": ["not found"],
        "names": [
            {{"first_name": "Alice", "middle_name": "not found", "last_name": "not found"}}
        ],
        "dob": ["not found"],
        "social_security_number": ["not found"],
        "subfolder_department": ["not found"]
        }}
        ```

        Now, please process the image I will provide and extract the fields accordingly, following the rules above.
    """
    return prompt_text
##### Setup Azure OpenAI in Python #####
# Use the API Key and Endpoint to configure the OpenAI client.


messages = []
def encode_image(image_path:str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")



def extract_and_save_json(output_folder, response_text, output_filename):
    """
    Extracts JSON from the response text using regex patterns and saves it as a JSON file.

    Parameters:
        output_folder (str): Path where the JSON file should be saved.
        response_text (str): The text response containing the JSON data.
        output_filename (str): Name of the output JSON file (without extension).
    """
    # Define regex patterns for extracting JSON content
    patterns = [
        r"'''<json response>\s*(\{.*?\})\s*'''",
        r"```<json response>\s*(\{.*?\})\s*```",
        r"```json\s*(\{.*?\})\s*```",
        r"'''json\s*(\{.*?\})\s*'''",
        r"\*\*<json response>\*\*\s*(\{.*?\})\s*\*\*<json response>\*\*",
        r"<json response>\s*(\{.*?\})\s*</json response>"
    ]

    extracted_data = None
    # Try to extract JSON content using regex patterns
    for pattern in patterns:
        print(f"Trying pattern: {pattern}")
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1)  # Extract the JSON string
            try:
                extracted_data = json.loads(json_str)  # Convert to JSON object
                break  # Stop if valid JSON is found
            except json.JSONDecodeError:
                print("Invalid JSON format, trying next pattern...")

    # If no JSON found, set a default error response
    if extracted_data is None:
        extracted_data = {"error": "No valid JSON found in response"}

    # Define the output file path
    output_file = os.path.join(output_folder, f"{output_filename}.json")

    # Save the results to a JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)

    print(f"Saved results for {output_filename} to {output_file}")
import os
import traceback

if __name__ == "__main__":
    image_path = '/home/ntlpt19/TF_testing_EXT/dummy_responces/Illumifin_data/All_classes_images'
    image_results = '/home/ntlpt19/TF_testing_EXT/dummy_responces/Illumifin_data/results'
    ocr_folder = '/home/ntlpt19/TF_testing_EXT/dummy_responces/Illumifin_data/OCR/All_classes_OCR'
    error_log_path = 'error_log.txt'  # Log file to store errors
    max_samples = 50

    os.makedirs(image_results, exist_ok=True)

    for doc_class in os.listdir(image_path):
        os.makedirs(os.path.join(image_results, doc_class), exist_ok=True)
        
        for imgs_ in os.listdir(os.path.join(image_path, doc_class)):
            output_filename = os.path.splitext(imgs_)[0]  # Get filename without extension
            image_file_path = os.path.join(image_path, doc_class, imgs_)
            
            try:
                image_base64 = encode_image(image_file_path)
                print("\n# Case 2: Only image + prompt")
                
                if not os.path.exists(os.path.join(image_results, doc_class, output_filename + ".json")):
                    txt_file = output_filename + "_text.txt"
                    txt_file_path = os.path.join(ocr_folder, doc_class, txt_file)
                    
                    # Open the text file
                    with open(txt_file_path, "r", encoding="utf-8") as f:
                        ocr_content = f.read()
                    
                    prompt_ext = get_prompt(ocr_content)
                    messages.append({"role": "user", "content": prompt_ext})
                    messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]})

                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=0
                    )

                    # Extract response content
                    output_text = response.choices[0].message.content
                    print(output_text)
                    extract_and_save_json(os.path.join(image_results, doc_class), output_text, output_filename)
                else:
                    print(f"Results for {output_filename} already exist. Skipping...")
            
            except Exception as e:
                # Log error details into the text file
                with open(error_log_path, "a", encoding="utf-8") as error_log:
                    error_log.write(f"Error processing {image_file_path}: {str(e)}\n")
                    error_log.write(traceback.format_exc() + "\n\n")
                print(f"Error processing {image_file_path}. Logged error.")
            
            max_samples -= 1
            if max_samples == 0:
                break