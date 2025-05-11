from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import base64
import json
import re
import os
from typing import Optional, List, Dict, Any, Union

app = FastAPI(title="Document Extraction API")

# Configuration - These should be set via environment variables in production
api_version = "2024-08-01-preview"
azure_endpoint = "https://ai-akshayamalik3400ai955768269900.openai.azure.com/openai/deployments/gpt-4o/chat/completions?2024-08-01-preview"
api_key = "7jlPMIpHvaEdpt53TE7dsXUbyrJEiVRR9uj1UXysqKhw3HKcSKzwJQQJ99BBACHYHv6XJ3w3AAAAACOGFJYF"

# Initialize Azure OpenAI client
client = openai.AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)

# System prompt as defined in your code
system_prompt = """
You are an AI that extracts key-value pairs from text.

Note:
- If the field is not present or illegible, return "Not Found".
- Preserve the exact format and capitalization of the text within the values.
"""

# Input data model
class ExtractionRequest(BaseModel):
    image_name: str
    b64_encoded_image: str
    fileType: str
    ocr_text: Optional[str] = None

# Response model
class ExtractionResponse(BaseModel):
    extracted_data: Dict[str, Any]

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

def extract_json_from_response(response_text):
    """
    Extracts JSON from the response text using regex patterns.
    """
    patterns = [
        r"'''<json response>\s*(\{.*?\})\s*'''",
        r"```<json response>\s*(\{.*?\})\s*```",
        r"```json\s*(\{.*?\})\s*```",
        r"'''json\s*(\{.*?\})\s*'''",
        r"\*\*<json response>\*\*\s*(\{.*?\})\s*\*\*<json response>\*\*",
        r"<json response>\s*(\{.*?\})\s*</json response>",
        r"\{\"carrier\".*\}"  # Fallback to find just a JSON object
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # If no structured JSON is found, try to load the entire response as JSON
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to extract valid JSON from model response")

@app.post("/extract", response_model=ExtractionResponse)
async def extract_document_info(request: ExtractionRequest):
    try:
        # Verify the base64 string
        try:
            # Check if base64 string is valid
            base64.b64decode(request.b64_encoded_image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 encoded image")

        # Prepare messages for the OpenAI API
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt with OCR text if provided, otherwise just the prompt
        prompt_text = get_prompt(request.ocr_text or "OCR text not provided")
        messages.append({"role": "user", "content": prompt_text})
        
        # Add the image
        messages.append({
            "role": "user", 
            "content": [
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/{request.fileType};base64,{request.b64_encoded_image}"
                    }
                }
            ]
        })

        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )

        # Extract response content
        output_text = response.choices[0].message.content
        
        # Extract JSON from response
        extracted_data = extract_json_from_response(output_text)
        
        # Return the extracted data
        return {"extracted_data": extracted_data}
    
    except Exception as e:
        # Log the error (in a production environment, you would use a proper logging framework)
        print(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

# Optional: Save the extracted data to a file
@app.post("/extract-and-save")
async def extract_and_save(request: ExtractionRequest, output_folder: str = "extraction_results"):
    response = await extract_document_info(request)
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Save to file
    output_filename = f"{request.image_name}.json"
    output_path = os.path.join(output_folder, output_filename)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(response.extracted_data, f, indent=4, ensure_ascii=False)
    
    return {
        "message": f"Extraction completed and saved to {output_path}",
        "extracted_data": response.extracted_data
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)