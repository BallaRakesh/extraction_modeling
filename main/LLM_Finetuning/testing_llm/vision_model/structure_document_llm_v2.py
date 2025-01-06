import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig
import os
import json
import re
from dotenv import load_dotenv
from extraction import processor

# Load .env file
load_dotenv()

# Retrieve Hugging Face token
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError(
        "Hugging Face token not found in the .env file. Please add 'HF_TOKEN=your_token' to the .env file.")


def read_labels_from_file(labels_file_path):
    """
    Read labels from a text file.

    Args:
        labels_file_path (str): Path to the text file containing labels

    Returns:
        list: List of labels extracted from the file
    """
    try:
        with open(labels_file_path, 'r', encoding='utf-8') as file:
            # Read lines, strip whitespace, and remove empty lines
            labels = [line.strip() for line in file if line.strip()]
        return labels
    except FileNotFoundError:
        print(f"Error: Labels file not found at {labels_file_path}")
        return []
    except IOError:
        print(f"Error: Unable to read labels file at {labels_file_path}")
        return []


def labels_to_json(labels):
    """
    Converts a list of labels into a JSON format with each label as a key and an empty string as its value.

    Args:
        labels (list): List of label strings.

    Returns:
        str: JSON string with each label as a key and an empty string as the value.
    """
    json_data = {label: "" for label in labels}
    return json.dumps(json_data, indent=4)


def load_model(model_id):
    """
    Load the Mllama model and processor with the specified configuration.

    Args:
        model_id (str): The ID of the pretrained model to load.

    Returns:
        tuple: A tuple containing the loaded model and processor.
    """
    # Create deterministic generation configuration
    generation_config = GenerationConfig(
        do_sample=False,  # Disable sampling
        temperature=0.0,  # Set temperature to 0 for deterministic output
        max_new_tokens=1024  # Maximum tokens to generate
    )

    # Load model with configuration
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        generation_config=generation_config  # Apply deterministic config
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


def extract_invoice_details(image_path, labels_file_path, model, processor):
    """
    Flexibly extract key-value pairs from an invoice image using labels from a file.

    Args:
        image_path (str): Path to the local invoice image file
        labels_file_path (str): Path to the text file containing potential labels
        model:  llama vision model
        processor:  llama vision processor

    Returns:
        dict: Extracted invoice details as key-value pairs
    """
    # Read labels from file
    possible_labels = read_labels_from_file(labels_file_path)
    print(possible_labels)

    # Check if labels were successfully read
    if not possible_labels:
        return {"error": "No labels found to extract"}

    # Model initialization
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    # # Create deterministic generation config
    # generation_config = GenerationConfig(
    #     do_sample=False,  # Disable sampling
    #     temperature=0.0,  # Set temperature to 0 for deterministic output
    #     max_new_tokens= 1024
    # )

    # model = MllamaForConditionalGeneration.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    #     generation_config=generation_config  # Apply deterministic config
    # )
    # processor = AutoProcessor.from_pretrained(model_id)

    # Load local image
    try:
        image = Image.open(image_path)
    except FileNotFoundError:
        return {"error": "Image file not found"}
    except IOError:
        return {"error": "Unable to open image file"}

    # Construct prompt with flexible extraction approach
    # labels_str = ", ".join(possible_labels)
    labels = labels_to_json(possible_labels)
    # print(labels)
    # exit('+++++++++++++++=')
    prompt = f"""
<|im_start|>system
DIRECTIVE: EXTRACT INVOICE DETAILS INTO STRICT JSON FORMAT
The task is to extract information from a Tax Invoice document. This document is issued by the vendor to the buyer and contains critical transaction details.

INPUT REQUIREMENTS:
- Keys to extract: {labels}

CRITICAL EXTRACTION GUIDELINES:
- Response format MUST be a single JSON object. No additional text, explanations, or annotations allowed.
- All keys MUST be present in the JSON output.
- If a key has multiple possible values, provide the most confident one first, followed by alternates if available.
- Maintain maximum accuracy in extraction. 
- Extracting every value check the upper or left of the value clearly.

CONTEXTUAL ASSUMPTIONS:
- "shipper" and "vendor" are considered the same entity if "vendor" is not explicitly mentioned.
- "pan" and "company pan" are equivalent unless otherwise specified.

OUTPUT FORMAT:
Respond ONLY with the extracted JSON data structure. Ensure it is syntactically and semantically correct. 
No additional formatting, explanations, or comments.
Duplicate keys not allowed in reponse

<|im_end|>

<|im_start|>user
You are an expert in generating JSON outputs and have 20 years of experience in producing structured JSON. Ensure the response is strictly encapsulated within a single JSON object and does not include any non-JSON content, commentary, or formatting artifacts.
the output should ```<json response> ```

<|im_end|>

"""
    # Prepare messages for the model
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]

    # Process input
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # Generate response with deterministic settings
    output = model.generate(**inputs,
                            do_sample=False,  # No sampling
                            temperature=0.0,  # Ensure deterministic output
                            max_new_tokens=1024
                            # Penalize repetitive outputs
                            )

    # Decode response
    response = processor.decode(output[0])
    print(response)
    print("### Response")
    try:
        parsed_json = extract_and_validate_json(response, possible_labels)
        return parsed_json
    except Exception as e:
        print(f"unable to parse the response as json: {e}")
        return response


import re
import json


# def extract_and_validate_json(response_text, possible_labels):
#     """
#     Extracts JSON between specific delimiters in the given response text and removes keys not in the possible labels.

#     Args:
#         response_text (str): The input text containing JSON.
#         possible_labels (list): A list of valid keys for the extracted JSON.

#     Returns:
#         dict: A dictionary containing only keys that match the possible labels or an error message.
#     """
#     try:
#         # Define the regex to extract JSON between specific markers
#         pattern = r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\s*<json response>\s*(\{.*?\})\s*</json response><\|eot_id\|>"

#         # Search for the JSON using the pattern
#         match = re.search(pattern, response_text, re.DOTALL)

#         if match:
#             json_str = match.group(1)  # Capture the JSON part
#             try:
#                 # Parse JSON into a Python dictionary
#                 extracted_data = json.loads(json_str)

#                 # Remove keys not in the possible labels
#                 validated_data = {k: v for k, v in extracted_data.items() if k in possible_labels}

#                 return validated_data
#             except json.JSONDecodeError as e:
#                 return {"error": f"Invalid JSON format: {str(e)}"}

#         # If no match found
#         return {"error": "No JSON found in the text"}

#     except Exception as e:
#         return {"error": f"Extraction failed: {str(e)}"}

def extract_and_validate_json(response_text, possible_labels):
    """
    Extracts JSON between specific delimiters in the given response text and validates it against a list of labels.

    Args:
        response_text (str): The input text containing JSON.
        possible_labels (list): A list of keys to validate in the extracted JSON.

    Returns:
        dict: Validated JSON data as a Python dictionary or an error message.
    """
    # print("Possible labels:", possible_labels)

    # List of regex patterns to try
    patterns = [
        r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\s*<json response>\s*(\{.*?\})\s*</json response><\|eot_id\|>",
        # First pattern
        r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\s*<json response>\s*(\{.*?\})\s*<\|eot_id\|>",
        # Second pattern
        r"<\|eot_id\|><\|start_header_id\|>assistant<\|end_header_id\|>\s*<json response>\s*(\{.*?\})\s*"
        # Third pattern
    ]

    try:
        # Iterate through the patterns
        for pattern in patterns:
            print(f"Trying pattern: {pattern}")
            # Search for the JSON using the current pattern
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                json_str = match.group(1)  # Capture the JSON part
                try:
                    # Parse JSON into a Python dictionary
                    extracted_data = json.loads(json_str)
                    print("Extracted Data:", extracted_data)

                    # Validate the extracted data against the possible labels
                    validated_data = {
                        k: v for k, v in extracted_data.items()
                        if k in possible_labels
                    }

                    return validated_data
                except json.JSONDecodeError as e:
                    return {"error": f"Invalid JSON format: {str(e)}"}

        # If no pattern matched
        return {"error": "Text is not present in the JSON"}

    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}"}


# Example usage
if __name__ == "__main__":
    # Example invocation
    local_image_path = "/home/ntlpt58/myworkspace/helping_code/input_template_automation/scblcapplication_name_address_updated_1.png"
    # local_image_path = "/home/data_science/mani/table_extraction/Non_tabular_data/data/test_samples/362118_Invoice_page_0.png"
    labels_file_path = "/home/ntlpt58/myworkspace/helping_code/input_template_automation/label.txt"
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model, processor = load_model(model_id)

    invoice_details = extract_invoice_details(local_image_path, labels_file_path,model, processor)
    print(json.dumps(invoice_details, indent=2))

    # model_id= "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # model, processor= load_model(model_id)
    # print("Model loaded !")
    # # Path to the folder containing images
    # image_folder_path = "/home/data_science/mani/table_extraction/Non_tabular_data/data/100_test_samples"
    # labels_file_path = "/home/data_science/mani/table_extraction/Non_tabular_data/grasim_labels/labels.txt"
    # output_path= "/home/data_science/mani/table_extraction/Non_tabular_data/data/100_test_samples_results"
    # os.makedirs(output_path, exist_ok= True)
    # # Function to process all images in the folder
    # def process_images_in_folder(image_folder, labels_file, output_folder):
    #     # List all files in the directory
    #     image_files = [
    #         os.path.join(image_folder, f) for f in os.listdir(image_folder)
    #         if f.endswith((".png", ".jpg", ".jpeg"))  # Filter image files
    #     ]

    #     # Iterate through each image and extract details
    #     for image_path in image_files:
    #         print(f"Processing: {image_path}")
    #         try:
    #             invoice_details = extract_invoice_details(image_path, labels_file, model, processor)
    #             print(json.dumps(invoice_details, indent=2))  # Print the extracted details
    #             # Define the output JSON file name based on the image file name (e.g., image1.png -> image1.json)
    #             file_name = os.path.splitext(os.path.basename(image_path))[0] + ".json"
    #             output_file_path = os.path.join(output_folder, file_name)
    #             # Save the extracted details to a JSON file
    #             with open(output_file_path, 'w') as json_file:
    #                 json.dump(invoice_details, json_file, indent=2)

    #             print(f"Saved: {output_file_path}")  # Print the saved file path

    #         except Exception as e:
    #             print(f"Error processing {image_path}: {str(e)}")

    # # Invoke the function
    # process_images_in_folder(image_folder_path, labels_file_path, output_path)