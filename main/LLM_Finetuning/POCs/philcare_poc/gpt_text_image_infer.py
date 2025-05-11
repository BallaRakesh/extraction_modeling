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

system_prompt = """
You are an AI that extracts key-value pairs from text.

Note:
- If the field is not present or illegible, return "Not Found".
- Preserve the exact format and capitalization of the text within the values.
"""

prompt_text = """"
The invoice number is MDGJLSHFDJH. The due date is March 15, 2024. The total amount is $1,250.
"""
##### Setup Azure OpenAI in Python #####
# Use the API Key and Endpoint to configure the OpenAI client.

api_version = "2024-08-01-preview"
azure_endpoint = "https://ai-akshayamalik3400ai955768269900.openai.azure.com/openai/deployments/gpt-4o/chat/completions?2024-08-01-preview"
api_key = "7jlPMIpHvaEdpt53TE7dsXUbyrJEiVRR9uj1UXysqKhw3HKcSKzwJQQJ99BBACHYHv6XJ3w3AAAAACOGFJYF"

client = openai.AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

messages = []
def encode_image(image_path:str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Prompt for case 1                                                                                                                                                                                          
image_path = '/home/ntlpt19/Pictures/Screenshot from 2025-02-05 11-11-57.png'
image_base64 = encode_image(image_path)
print("\n# Case 2: Only image")
messages.append({"role": "user", "content": "discribe this image"})
# messages.append({"role": "user", "content": gpt_4o_general_table_extraction_with_csv_response_user_prompt})
messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]})

response = client.chat.completions.create(
    model="gpt-4o",  # Change this if your deployment name is different
    messages=messages,
    temperature=0
)

# Extract response content
output_text = response.choices[0].message.content
print(output_text)
exit('OKKKKKKKKKKKKKKKKKKKKK')
# Prompt for case 1


exit('OKKKKKKKKKKKKKKKKKKKKK')
# Prompt for case 1

# Make the API call
response = client.chat.completions.create(
    model="gpt-4o",  # Change this if your deployment name is different
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ],
    temperature=0
)

# Extract response content
output_text = response.choices[0].message.content

print(output_text)
exit('OKKKKKKKKKKKKKKKKKKKKK')


# Make the API call
response = client.chat.completions.create(
    model="gpt-4o",  # Change this if your deployment name is different
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text},
    ],
    temperature=0
)

# Extract response content
output_text = response.choices[0].message.content

print(output_text)
exit('OKKKKKKKKKKKKKKKKKKKKK')


def extract_and_validate_json(response_text, possible_labels):
    """
    Extracts JSON between specific delimiters in the given response text and validates it against a list of labels.
    Args:
        response_text (str): The input text containing JSON.
        possible_labels (list): A list of keys to validate in the extracted JSON.
    Returns:
        dict: Validated JSON data as a Python dictionary or an error message.
    """
    patterns = [
        r"'''<json response>\s*(\{.*?\})\s*'''",
        r"```<json response>\s*(\{.*?\})\s*```",
        r"```json\s*(\{.*?\})\s*```",
        r"'''json\s*(\{.*?\})\s*'''",
        r"\*\*<json response>\*\*\s*(\{.*?\})\s*\*\*<json response>\*\*",
        r"<json response>\s*(\{.*?\})\s*</json response>"
    ]
    
    try:
        # Iterate through the patterns
        for pattern in patterns:
            print(f"\nTrying pattern: {pattern}")
            match = re.search(pattern, response_text, re.DOTALL)
            
            if match:
                json_str = match.group(1)  # Capture the JSON part
                try:
                    extracted_data = json.loads(json_str)
                    validated_data = {
                        k: v for k, v in extracted_data.items()
                        if k in possible_labels
                    }
                    
                    print("Pattern Found Successfully !!!\n")
                    return {"status": "SUCCESS", "result": validated_data, "raw_output": response_text}
                
                except json.JSONDecodeError as e:
                    return {"status": f"ERROR: Failed to parse JSON response\nDETAILS: {e}", "result":{}, "raw_output": response_text}
            else:
                print("Pattern Not Found !!!")
                
        return {"status": f"ERROR: Text is not present in the JSON", "result":{}, "raw_output": response_text}

    except Exception as e:
        return {"status": f"Extraction failed: {str(e)}", "result":{}, "raw_output": response_text}


# Function that takes any text and extracts key-value pairs.

def extract_key_value_pairs(text="The invoice number is INV-2024-001. The due date is March 15, 2024. The total amount is $1,250.",
                            key_names_list = ["Invoice Number", "Due Date", "Total Amount"]):
    expected_output_json =  json.dumps(str({k:f"{k.upper()} VALUE" for k in key_names_list}), indent=4)
    
    # Define the Key-Value Extraction Prompt    
    # We need to structure the prompt to instruct GPT-4o to extract key-value pairs in our current scenario.
    prompt_text = f"""
                Extract the key-value pairs from the given text. 

                Text: "{text}"

                Expected Output:

                ```json
                {expected_output_json}
                ```

                """
    
    print("\n", "$"*40, sep="")
    print("\nUser Prompt Text :\n", prompt_text)
    print("$"*40)
    
    # Set up Azure OpenAI client (New API format)
    client = openai.AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )
    
    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",  # Change this if your deployment name is different
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        temperature=0
    )

    # Extract response content
    output_text = response.choices[0].message.content
    
    # output_text = """{
    #     'Invoice Number': 'INV-2024-001',
    #     'Due Date': 'March 15, 2024',
    #     'Total Amount': '$1,250'
    # }"""
    
    print("\n", "$"*40, sep="")
    print("\nOutput Response Text :\n", output_text, sep="")
    print("$"*40)

    return extract_and_validate_json(output_text, key_names_list)

    # try:
    #     return {"status": "SUCCESS", "result": json.loads(output_text), "raw_output": output_text}
    # except json.JSONDecodeError:
    #     try:
    #         return {"status": "SUCCESS", "result": eval(output_text), "raw_output": output_text}
    #     except Exception as e:
    #         return {"status": f"ERROR: Failed to parse JSON response\nDETAILS: {e}", "result":{}, "raw_output": output_text}


text1 = mds_1_Discharge_page_1_text
key_names_list1 = ["NationalProviderIdentifier", "CmsCertificationNumber"]

text2 = bea_1_page_1_text
key_names_list2 = ["CompanyName", "PhoneNumber"]

text3 = bea_1_page_4_text
key_names_list3 = ["ClinicianName"]

text4 = ccna_1_page_3_text
key_names_list4 = ["FacilityName", "FacilityAddress", "ResidentRoomNumber", "AreResidentMedicationsAdministeredByStaff", "TypeOfRoom"]

text = text1 + "\n\n" + text2 + "\n\n" + text3 + "\n\n" + text4
key_names_list = key_names_list1 + key_names_list2 + key_names_list3 + key_names_list4

extracted_data = extract_key_value_pairs(text, key_names_list)
print("\n\nExtracted Key-Value Pairs :\n", json.dumps(extracted_data, indent=4), "\n")
