import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor, GenerationConfig
import os
import json
import re
from dotenv import load_dotenv
import re
import json


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
        # torch_dtype=torch.bfloat16,
        torch_dtype=torch.float16,  # Changed from bfloat16
        device_map="auto",
        generation_config=generation_config  # Apply deterministic config
    )

    # Load processor
    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor


data_46A = """
drafts in duplicate at sight / usance
drawn on chase bank limited in duplicate ,
mentioning lc reference number and the lc
<sep-token>
issuance date
+ certificate of origin issued by xxxxxxxx
chamber of commerce inxx original ( s )
+ original phytosanitary certificate copies
+ in case of sea shipmentfull set of original
signed clean on board ocean / charter party /
multi modal bills of lading made out to
order and blank endorsed marked freight
collect / freight prepaid / freight payable
as per charter party evidencing shipment
of merchandise described above . bill of
lading must state full name and address of
i ) applicants full name and address
ii ) abc bank ltd full name and address as
parties to be notified .
short form , third party , freight forwarders
and lash bills of lading are not acceptable .
+ original signed airway bill issued by airline
company or its agent made in the name of chase
bank ltd.full name and address for account
applicant . airway bill should be marked
freight collect / freight prepaid .
airway bills must state full name and address
of applicants full name and address
+ signed commercial invoice in copies
"""
print('##################')
print('##################')
print(data_46A)

prompt_layout1 = """
        <|im_start|>system
        DIRECTIVE: EXTRACT DOCUMENT ENCLOSED INTO STRICT JSON FORMAT
        The task is to extract information from a Covering Schedule document. 

        CRITICAL EXTRACTION GUIDELINES:
        - Focus specifically on the "DOCUMENTS RECEIVED" section in the bill/document
        - For each document type listed, extract:
        * Document name as the key
        * Number of originals and copies as the value
        * Convert check marks (✓) to numeric "1" and empty/zero values as "0"
        - Response format MUST be a structured JSON object with:
        * Each document type as a separate key
        * Value should be an object containing {"originals": X, "copies": Y}
        - Exact document names should be preserved as they appear (e.g., "DRAFT", "INV", "BL/AWB", etc.)
        - Pay special attention to:
        * Check marks (✓) which indicate presence of documents
        * Distinguish between "0" entries and blank spaces
        * Maintain case sensitivity as shown in document
        - All visible document types MUST be included in output, even if quantities are zero
        - No additional text, explanations, or annotations allowed in the response

        EXPECTED OUTPUT FORMAT:
        {{
            "DRAFT": {{"originals": 1, "copies": 0}},
            "INV": {{"originals": 1, "copies": 0}},
            ...other documents...
        }}

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

prompt_combined = """
            <|im_start|>system
            DIRECTIVE: EXTRACT DOCUMENT DETAILS FROM COLLECTION SCHEDULE/BILL INTO STANDARDIZED JSON FORMAT

            CRITICAL EXTRACTION GUIDELINES:
            1. Document Extraction Rules:
            - Extract all document types mentioned in the image
            - Convert all quantity formats to originals and copies format
            - Standardize different document section formats into a single structure

            2. Quantity Conversion Rules:
            - For single numbers or  fractional formats (e.g., "2/3") and For "SET" notation (e.g., "2SET"):
                * If under "1 MAIL" or "originals" column: count as originals
                * If under "2 MAIL" or "copies" column: count as copies
            - For check marks (✓):
                * Convert to numeric "1"
            - For blank/empty entries:
                * Convert to numeric "0"

            3. Response Requirements:
            - Output single JSON object with standardized format
            - Preserve exact document names with case sensitivity
            - Include all documents even if quantities are zero
            - No additional text or explanations in response

            EXPECTED OUTPUT FORMAT:
            {
                "DRAFT": {"originals": 2/3, "copies": 0},
                "BILL OF LADING": {"originals": 2/1, "copies": 3},
                "INSURANCE POLICY": {"originals": 1, "copies": 0},
                "PACKING LIST": {"originals": 2, "copies": 0}
            }

            PROCESSING RULES:
            - Preserve exact document names as shown in source
            - Include all visible documents regardless of quantity
            - Ensure numeric values for all quantities
            - Output must be valid JSON only

            The response should ONLY contain the JSON object with the extracted document information. No additional formatting, explanations, or comments.
            <|im_end|>

            <|im_start|>user
            You are an expert in generating JSON outputs and have 20 years of experience in producing structured JSON. Ensure the response is strictly encapsulated within a single JSON object and does not include any non-JSON content, commentary, or formatting artifacts.
            the output should ```<json response> ```

            <|im_end|>

            """


promptlayout2 = """
            <|im_start|>system
            CRITICAL EXTRACTION GUIDELINES:
            - Extract documents organized under "1 MAIL" and "2 MAIL" sections separately
            - Response format MUST be a structured JSON object with:
            * Main sections "first_mail" and "second_mail"
            * Under each mail section, list all documents with their quantities
            * Each document entry should preserve exact naming as shown
            * Convert numeric values exactly as shown (e.g., "2/2", "2/3", "1", "2SET")

            FORMAT SPECIFICATIONS:
            - Maintain document hierarchy:
            * First level: mail sections (first_mail, second_mail)
            * Second level: document names and quantities
            - Preserve exact document names including spacing and case
            - Include all documents listed under each mail section
            - Capture quantity formats exactly as shown (fractions, sets, individual numbers)

            EXAMPLE OUTPUT FORMAT:
            {
                "first_mail": {
                    "DRAFT": "2/2",
                    "BILL OF LADING ORIGINALS": "2/3",
                    "INSURANCE POLICY/CERTIFICATE": "2",
                    "CERTIFICATE OF ORIGIN": "1",
                    "ANALYSIS REPORT FOR STRENGTH": "2SET"
                },
                "second_mail": {
                    "COMMERCIAL INVOICE": "3",
                    "BILL OF LADING COPIES": "2",
                    "PACKING LIST": "2",
                    "LABORATORY ANALYTICS": "2SET",
                    "ANALYSIS REPORT FOR COLOR DIFFERENCE": "2SET"
                }
            }

            ADDITIONAL REQUIREMENTS:
            - Maintain case sensitivity as shown in document
            - Include all documents even if quantity is "0" or blank
            - No additional text, explanations, or annotations allowed in response
            - Response must be valid JSON format only
            <|im_end|>

            <|im_start|>user
            You are an expert in generating JSON outputs and have 20 years of experience in producing structured JSON. Ensure the response is strictly encapsulated within a single JSON object and does not include any non-JSON content, commentary, or formatting artifacts.
            the output should ```<json response> ```

            <|im_end|>

            """


prompt_with_rule = f"""
        <|im_start|>system
        DIRECTIVE: EXTRACT DOCUMENT ENCLOSED INTO STRICT JSON FORMAT
        The task is to extract information from a Covering Schedule document. 

        1. CRITICAL EXTRACTION GUIDELINES:
        - Focus specifically on the "DOCUMENTS RECEIVED" section in the bill/document
        - For each document type listed, extract:
        * Document name as the key
        * Number of originals and copies as the value
        * Convert check marks (✓) to numeric "1" and empty/zero values as "0"
        - Response format MUST be a structured JSON object with:
        * Each document type as a separate key
        * Value should be an object containing {{"originals": X, "copies": Y}}
        - Exact document names should be preserved as they appear (e.g., "DRAFT", "INV", "BL/AWB", etc.)
        - Pay special attention to:
        * Check marks (✓) which indicate presence of documents
        * Distinguish between "0" entries and blank spaces
        * Maintain case sensitivity as shown in document
        - All visible document types MUST be included in output, even if quantities are zero
        - No additional text, explanations, or annotations allowed in the response


        2. LC Requirements Analysis:
        - Parse LC data for required documents and quantities
        - this is the LC Data: {data_46A}
        - Extract key requirements:
            * Draft requirements (e.g., "drafts in duplicate")
            * BL/AWB specifications
            * Certificate requirements
            * Invoice requirements
            * Other specified documents

        3. Compliance Verification:
        - Match extracted documents against LC requirements
        - Verify quantity compliance for each document
        - Check for missing required documents
        - Validate document characteristics as per LC

        EXPECTED OUTPUT FORMAT:
        {{
            "extracted_documents": {{
                "DRAFT": {{"originals": 2, "copies": 0}},
                "BILL OF LADING": {{"originals": 2, "copies": 3}}
                // other documents
            }},
            "lc_requirements": {{
                "DRAFT": {{"required": true, "details": "drafts in duplicate", "compliant": true}},
                "BILL OF LADING": {{"required": true, "details": "full set of original", "compliant": false}}
                // other requirements
            }},
            "verification_summary": {{
                "missing_documents": ["list of missing required docs"],
                "quantity_mismatches": ["documents with incorrect quantities"],
                "overall_compliance": false
            }}
        }}

        VERIFICATION RULES:
        - Match document names considering variations/synonyms
        - Compare quantities against LC specifications
        - Flag missing required documents
        - Identify quantity discrepancies
        - Provide clear compliance status for each document

        The response must include:
        1. Extracted documents with quantities
        2. LC requirements for each document
        3. Verification results showing compliance
        4. Clear explanation of any discrepancies

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


prompt_col_data1 = """
        <|im_start|>system
        DIRECTIVE: EXTRACT DOCUMENT ENCLOSED INTO STRICT JSON FORMAT
        The task is to extract information from a Covering Schedule document. 

        CRITICAL EXTRACTION GUIDELINES:
        - Focus specifically on the "DOCUMENTS RECEIVED" section in the bill/document
        - For each document type listed, extract:
        * Document name as the key
        * Convert check marks (✓) to numeric "1" and empty/zero values as "0"
        - Response format MUST be a structured JSON object with:
        * Each document type as a separate key
        * Value should be an object containing {"column_name1": X, "column_name2": Y}
        * Remember these column_name1 and column_name2 some time might be original and copy , some times 1mail and 2mail 
            it completely depends on the document, what ever mentioned on the document , pleaes make it as a column name and manintain the same in the results
        - Exact document names should be preserved as they appear (e.g., "DRAFT", "INV", "BL/AWB", etc.)
        - Check propery , some times , the give information will be in a structured table or some times directly given
        - Pay special attention to:
        * Check marks (✓) which indicate presence of documents
        * Distinguish between "0" entries and blank spaces
        * Maintain case sensitivity as shown in document
        - All visible document types MUST be included in output, even if quantities are zero
        - No additional text, explanations, or annotations allowed in the response

        EXPECTED OUTPUT FORMAT:
        {{
            "DRAFT": {"column_name1": 1,column_name2: 3},
            "INV": {"column_name1": 1},
            "B/L": {"column_name1": 3c+2NN},
            ...other documents...
        }}

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

prompt_col_data2 = """
        <|im_start|>system
        DIRECTIVE: EXTRACT DOCUMENT ENCLOSED INTO STRICT JSON FORMAT
        The task is to extract information from a Covering Schedule document. 

        EXTRACTION GUIDELINES:

        1. Document Source:
        - Focus specifically on the "DOCUMENTS RECEIVED" section in the bill/document or directly mentioned like "documents"
          or Documents Accompanied or given it like "we enclose the following documents"
        - For each document type listed, extract:
        - Document name as the key
        - Handle both structured tables and unstructured listings

        2. Data Extraction Rules:
        - Document names: preserve exact spelling and case
        - document names should be preserved as they appear (e.g., "DRAFT", "INV", "BL/AWB", "commercial invoice" etc.)
        - Make the Document Name as the key name and provide the information mentiond about the doument . 
        - Values: 
            * Convert ✓ (check marks) to "1"
            * Convert empty/blank to "0"
            * Preserve special notations (e.g., "3c+2NN") exactly as shown

        3. Output Structure:
            * if the information is given as "originals" and "copies"
            The OUTPUT will be like:
            {{
                "DRAFT": {"originals": 1, copies: 3},
                ...other documents...
            }}
            
            * if the information is given as "1st Mail" and "2nd Mail"
            The OUTPUT will be like:
            {{
                "DRAFT": {"1st Mail": 1, "2nd Mail": 3+NN},
                ...other documents...
            }}
            * if the information is given as "1st" and "2nd"
            The OUTPUT will be like:
            {{
                "DRAFT": {"1st": 1, "2nd": 3+NN},
                ...other documents...
            }}

        CRITICAL REQUIREMENTS:
        - Include ALL visible documents
        - Preserve ALL column headers exactly as shown
        - Maintain case sensitivity
        - Include zero values and empty entries
        - Handle both tabular and non-tabular formats
        - No duplicate document entries
        - No explanatory text in output
        - Do not include unnecessary information like **Drawee** or **Drawer**; focus only on the document type.
        
        
        RESPONSE FORMAT:
        Valid JSON only, no additional text or formatting
        <|im_end|>

        <|im_start|>user
        You are an expert in generating JSON outputs and have 20 years of experience in producing structured JSON. Ensure the response is strictly encapsulated within a single JSON object and does not include any non-JSON content, commentary, or formatting artifacts.
        the output should ```<json response> ```

        <|im_end|>

        """

prompt_col_data3 = """
        Analyze the bank document image and extract exactly these details:

        1. List all documents mentioned in the image in this JSON format:
            - Focus on the "Documents" section
            - Any section mentioning "Documents Received" or "Documents Enclosed"
            - the documents are listed line by line or some times in a two columns also, examine carefully and give the results
            Expected Result:
            {
            "documents": [
                "document1",
                "document2"
            ]
            }

        2. Identify and list any column headers associated with the documents section (e.g., [1st, 2nd])

        Rules:
        - Extract only from sections labeled "Documents" or similar headings
        - Maintain exact capitalization and formatting from the image
        - List each document only once
        - Do not include any explanations, interpretations, or additional notes
        - Do not include document quantities or numbers
        - Provide only these two outputs: the JSON list and the column headers list
        """


prompt_col_data4 = """
This is the DOCUMENT ENCLOSED portion from a Covering Schedule document. 
{
  "documents": [
    "Draft",
    "Customs Certificate",
    "Insurance Policy",
    "Weight Note",
    "Bill of Lading",
    "Inspection Certificate",
    "Non-Negotiable B/L",
    "CERT OF ANALYSIS",
    "Invoice",
    "Customs Invoice",
    "Packing List",
    "Cert of origin",
    "Airway Bill",
    "Export Licence"
  ]
}
Now Based on this information , give the 1st mail and 2nd mail information about from the image, doucment wise 
"""

def extract_invoice_details(image_path, imgs_file, labels_file_path, model, processor, results_path):
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
    # possible_labels = read_labels_from_file(labels_file_path)
    # print(possible_labels)
    
    # Check if labels were successfully read
    # if not possible_labels:
    #     return {"error": "No labels found to extract"}

    # Model initialization
    # model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

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

    # Load local image current_img_path
    try:
        # image = Image.open(os.path.join(image_path, imgs_file))
        image = Image.open(current_img_path)
    except FileNotFoundError:
        return {"error": "Image file not found"}
    except IOError:
        return {"error": "Unable to open image file"}

    # Construct prompt with flexible extraction approach
    # labels_str = ", ".join(possible_labels)
    # labels = labels_to_json(possible_labels)
    # print(labels)
    # exit('+++++++++++++++=')

    # Prepare messages for the model
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt_col_data3}
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
    # print(response)
    save_res_path = os.path.join(results_path, imgs_file.replace('.png','.txt'))
    with open(save_res_path, 'w') as f_:
        f_.write(str(response))
    return response
    
    print("### Response")
    try:
        parsed_json = extract_and_validate_json(response, possible_labels)
        return parsed_json
    except Exception as e:
        print(f"unable to parse the response as json: {e}")
        return response



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


import os
import time
import logging
from datetime import datetime

def setup_logging(log_dir):
    """
    Set up logging configuration
    
    Args:
        log_dir (str): Directory where log files will be stored
    
    Returns:
        logging.Logger: Configured logger object
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log filename with timestamp
    log_filename = f"image_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()  # This will print logs to console as well
        ]
    )
    
    return logging.getLogger(__name__)

# Example usage
if __name__ == "__main__":
    # Example invocation
    local_image_path = "/home/data_science/anand/llm_vision_for_structure_document/itf_docs/document_enclose/Images_layout2"
    # local_image_path = "/home/data_science/mani/table_extraction/Non_tabular_data/data/test_samples/362118_Invoice_page_0.png"
    labels_file_path = "/home/data_science/anand/llm_vision_for_structure_document/label.txt"
    results_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/res_layout2'
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_120_page_2.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_119_page_0.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_123_page_1.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_125_page_8.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_127_page_2.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_132_page_1.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_133_page_0.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/document_enclose/Images_layout1/Covering_Schedule_113_page_0.png'
    current_img_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images/Covering_Schedule_122_page_1.png'
    
    # Set up logging
    log_dir = os.path.join(results_path, 'logs')
    logger = setup_logging(log_dir)
    # Create results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    # Process each image
    model, processor = load_model(model_id)
    for imgs in os.listdir(local_image_path):
        logger.info(f"Starting processing of image: {imgs}")
        start_time = time.time()
        local_image_path = '/home/data_science/anand/llm_vision_for_structure_document/itf_docs/Images'
        # # imgs = 'Covering_Schedule(2012_08_28_10_22_59_6788)_462_page_0.png'
        imgs = 'Covering_Schedule_119_page_0.png'
        invoice_details = extract_invoice_details(local_image_path,imgs, labels_file_path,model, processor, results_path)
        # Calculate processing time
        processing_time = time.time() - start_time
        # Log results
        logger.info(f"Completed processing {imgs}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")
        print('>>>>>>>>>>>>%%%%%%%%%%%>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>%%%%%%%%%%%>>>>>>>>>>>>>>>>>>>')
        print('>>>>>>>>>>>>%%%%%%%%%%%>>>>>>>>>>>>>>>>>>>')
        print(invoice_details)
        exit('DONE')
    # print(json.dumps(invoice_details, indent=2))