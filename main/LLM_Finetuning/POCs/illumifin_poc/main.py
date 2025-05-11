from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai
import base64
import json
import re
import os
import pickle
import cv2
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from datetime import datetime
from google.cloud import vision
from google.oauth2.service_account import Credentials
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel
from fastapi.responses import JSONResponse

app = FastAPI(title="Document Processing API")

# Classification model paths
tfidf_path = "/datadrive/rakesh/POCs/illumifin_poc/models/tf_idf_model.pkl"
svm_path = "/datadrive/rakesh/POCs/illumifin_poc/models/svm_model.pkl"

# Extraction configuration
api_version = "2024-08-01-preview"
azure_endpoint = "https://ai-akshayamalik3400ai955768269900.openai.azure.com/openai/deployments/gpt-4o/chat/completions?2024-08-01-preview"
api_key = "7jlPMIpHvaEdpt53TE7dsXUbyrJEiVRR9uj1UXysqKhw3HKcSKzwJQQJ99BBACHYHv6XJ3w3AAAAACOGFJYF"

# Classification class mapping
CLASS_MAPPING = {
    "BEA": 0,
    "CCNA": 1,
    "Cognitive Questionnaire MD": 2,
    "MDS": 3,
    "Medical Records": 4,
    "Medication Administration Records (MAR)": 5,
    "Medication Administration Screen Form": 6,
    "Medication List": 7,
    "Nursing Assessment": 8,
    "Plan of Care (POC)": 9,
    "RFR Facility": 10,
    "RFR General": 11,
    "RFR HHC": 12
}

# Google Vision API credentials
vision_credential_dict = {
    "type": "service_account",
    "project_id": "imagedocumentprocessing",
    "private_key_id": "598eab6cfc358ef3fac5f1a0a1b178dc10706d5b",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDGzlwHe9Qrf89O\nAEcMOKb8SNeEHK0YUES3Yy/5zWBPj+ecR+NKMsP7ay1KMMUjKFVLbixrGr8KJue7\nwx7lFmklYmeIJlaiI/PBgNG3ky8Yth14OHB20bI9E77XRIfHpT9p+WyS4t0WkYxq\nWFLhwKGViPArFVHt2HMvHTHWGM1wS+FOWERZ/1Z01bQstUuZRqKyQCIbukUfEoDM\nKOiaffICsOyzdxhXiqBb+qLUvZKWSAcsUy18OMf+NqcBrt1IN4kID95UVRANKAy7\naX+2mh9RQ9gi2YhMdax7nVZdvWVkrW02q29TMgOuC4EAtGdCutukyqd1DTA+igc2\nDIp+RwJvAgMBAAECggEAINA0rmOI3Hkm/Ufcci7zmNZpA/w7sbSl8uLjK0bzq44j\n+05+PGPupxPEkOdF0oy4r0+K806h04oiW4JUGhm91xbL4dP6Hp7yf7DEbJlVf62n\nZY1jOqlX0u7sY9mC07f5pIMvoXriZPQ3CeJ0I6DIysakZWgKcsh0EoWuERlc5zjE\nf4FeLo5LQNoAMDPBa5PmH4LYB+wej8owkcYPYRaiR+s5ZHjJ7Sz1Pfricg2S6f3p\ntPCrterWVStbmEr5zN9fGAkERLgk44DxvX7gdOp8dOeHGJCs5/0XlzlOD2lZBdkT\nsUYKVfXRNFcxd1zgkA6ugEpVd0boKs9PBXcWtICI0QKBgQDpB01p7vD6f3DMvrMy\nu7BxmY18MWQmEpuiwC3IUePocTD+/mUnyxqordzYVrr9rBVPi/klgHQauJT4DaZf\nqk7AykEG44nH86RHO7V9ltpcTAopwh7Fdn26CBboIw06tRRAi+hsXKK82W0hzLRr\n9bOiYwWIZEMoR1ptwz/1bWCW+QKBgQDaZ2vYgMVIH2Vma0myc/Ss4s9j2BwUZqOu\n4WPexXcxDfkqFm34E0qiuBOy8hHlCTqa+EN8PinSQzOFi5RhgTKlXwJnULFOHok1\ngHIlTBnyIY3vCEsdgotaCCckiePeHs38NqkVF3PR1/I6CJx3Sn14wTGvZI9BvfkN\nXz+Apsw2pwKBgAQe+/B+qEZV0KHeUX75MMKhi44BtZqyw4vaSDT9tcEbl1k7GIDe\np6cKBBjTV9U6oNnaSNqv7d23G/NTEnkoouHn8cR0a7Bcj7AuzyPccholwhxA8Zhe\nYxTSJc2PKSG7qBMIJmEcNkiOs85gN2SdYMLja2qhE0r4EBJW3J+zgnzJAoGAQJq7\nIg3lTZJL8mHmKO14APQOmOBCXh6GaKKvYiRTP6V8gOClou5B86Qz34kPlgqG1XR+\nse/JezL5uJUzTSxkiNpz4Y/TAuKxf6wBF9tNCXy3eW+gmJgXFIi422wWhYvjLqoU\nlfKcBgS02EnXzsmwF9o+Ej3SyoWheD0mka3FVp0CgYEAt5sNlKKDrMnNdjdwzeCv\ntbhOm18fgb2a0ZUGLb8IQrL+o9/w4KYex3yLWkkHoeVoSpjaq2uq4/YS9O07Xm5X\nekKMVir6p0J4w8I1t6uz7OVp1P7mGt7KWFO/KkAPjxJUHWj93cLZIGEINxMtb3MH\nmO2C74/FgxYjk8k2XwKI0J0=\n-----END PRIVATE KEY-----\n",
    "client_email": "document-ai@imagedocumentprocessing.iam.gserviceaccount.com",
    "client_id": "112795268809510843323",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/document-ai%40imagedocumentprocessing.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# System prompt for extraction
system_prompt = """
You are an AI that extracts key-value pairs from text.

Note:
- If the field is not present or illegible, return "Not Found".
- Preserve the exact format and capitalization of the text within the values.
"""

# Global objects
tfidf_vectorizer = None
svm_model = None
vision_credentials = None
openai_client = None

# Input data models
class ImageData(BaseModel):
    image_name: str
    b64_encoded_image: str
    fileType: str
    # ocr_text: Optional[str] = None

# Output models
class ClassificationResult(BaseModel):
    predicted_class: str
    confidence_score: Optional[float] = None
    processing_time: float

class ExtractionResponse(BaseModel):
    extracted_data: Dict[str, Any]

class CombinedResponse(BaseModel):
    classification: ClassificationResult
    extraction: Dict[str, Any]

# Initialize Azure OpenAI client
def get_openai_client():
    global openai_client
    if openai_client is None:
        openai_client = openai.AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint
        )
    return openai_client

# Initialize the classification models
def load_classification_models():
    global tfidf_vectorizer, svm_model
    
    if tfidf_vectorizer is None or svm_model is None:
        try:
            with open(tfidf_path, "rb") as f:
                tfidf_vectorizer = pickle.load(f)
            with open(svm_path, "rb") as f:
                svm_model = pickle.load(f)
        except Exception as e:
            return HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")
    
    return tfidf_vectorizer, svm_model

# OCR functions
def get_ocr_vision_api(image_data):
    """Extract text from image using Google Vision API"""
    global vision_credentials
    
    if vision_credentials is None:
        vision_credentials = Credentials.from_service_account_info(vision_credential_dict)
    
    client = vision.ImageAnnotatorClient(credentials=vision_credentials)
    image = vision.Image(content=image_data)
    response = client.text_detection(image=image)
    
    # Extract all text
    all_text = ""
    if response.text_annotations:
        all_text = response.text_annotations[0].description
        # print(all_text)
    # Extract word coordinates
    word_coordinates = []
    for i, text in enumerate(response.text_annotations):
        if i != 0:  # Skip the first one which is the complete text
            vertices = [(v.x, v.y) for v in text.bounding_poly.vertices]
            x1 = min([v.x for v in text.bounding_poly.vertices])
            x2 = max([v.x for v in text.bounding_poly.vertices])
            y1 = min([v.y for v in text.bounding_poly.vertices])
            y2 = max([v.y for v in text.bounding_poly.vertices])
            
            # Avoid zero dimensions
            if x2 - x1 == 0:
                x2 += 1
            if y2 - y1 == 0:
                y2 += 1
                
            word_coordinates.append({
                "word": text.description,
                "left": x1,
                "top": y1,
                "width": x2 - x1,
                "height": y2 - y1,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })
    
    # Format word coordinates
    save_coords = {}
    for i, word_info in enumerate(word_coordinates):
        save_coords[i] = {
            "text": word_info['word'],
            "word": word_info['word'],
            "left": word_info['left'],
            "top": word_info['top'],
            "width": word_info['width'],
            "height": word_info['height'],
            "x1": word_info['x1'],
            "y1": word_info['y1'],
            "x2": word_info['x2'],
            "y2": word_info['y2'],
            "bbox": [word_info['x1'], word_info['y1'], word_info['x2'], word_info['y2']]
        }
        
    return all_text, save_coords

# Classification functions
def preprocess_text_for_tfidf(text, tfidf_vectorizer):
    """Transform extracted text into TF-IDF features"""
    return tfidf_vectorizer.transform([text]).toarray()

def get_class_name(prediction_label):
    """Convert numerical prediction to class name"""
    for category, label in CLASS_MAPPING.items():
        if label == prediction_label:
            return category
    return "Unknown"

# Extraction functions
def get_extraction_prompt(ocr_all_text):
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
            1. **Carrier (Life Insurance Company Name)**:  
            - Look for the exact phrase **"Carrier"** in the document.  
            - If not found, identify any **life insurance company name** or any company name present in the image (e.g., "ABC Life Insurance", "XYZ Corp").  
            - Extract the corresponding value(s) as a string or list.  
            - If no company name is found, set its value to **["not found"]**.  

            2. **Policy Number**:  
            - Locate the term **"Policy Number"** in the document.  
            - Extract the associated value(s) as a string or list.  
            - If missing, return **["not found"]**.  

            3. **Policy Holder's Name**:  
            - Identify **the name of the policyholder**, ensuring it is explicitly labeled with prefixes such as:  
                - **"Name"**, **"First Name"**, **"Last Name"**, **"Applicant"**, **"Insured"**, or similar labels.  
            - Extract the name(s) into a **list of dictionaries** with the following format:  
                ```json
                [
                {{"first_name": "John", "middle_name": "A", "last_name": "Smith"}}
                ]
                ```
            - Extraction rules:  
                - If labeled separately (e.g., **"First Name: John", "Last Name: Smith"**), use those values for `'first_name'`, `'middle_name'` (if provided), and `'last_name'`.  
                - If a full name is given (e.g., **"Name: John A Smith"**), split it accordingly.  
                - If only a single name is present (**"John"** or **"Name: John"**), set:  
                - `'first_name'`: `"John"`  
                - `'middle_name'`: `"not found"`  
                - `'last_name'`: `"not found"`  
                - If multiple names are explicitly labeled (e.g., **"Applicant: John Smith", "Insured: Jane Doe"**), create separate dictionaries:  
                ```json
                [
                    {{"first_name": "John", "middle_name": "not found", "last_name": "Smith"}},
                    {{"first_name": "Jane", "middle_name": "not found", "last_name": "Doe"}}
                ]
                ```
            - Exclusions:  
                - Ignore names that are not explicitly labeled.  
                - Exclude **company names, application names**, or names found in signatures/unrelated text.  
            - If no valid policyholder name is found, return an **empty list**: `[]`.  

            4. **Policy Holder's Date of Birth (DoB)**:  
            - Identify text labeled as **"DoB"**.  
            - Extract the corresponding value(s) as a string or list.  
            - If missing, set it to **["not found"]**.  

            5. **Policy Holder's Social Security Number (SSN)**:  
            - Identify text labeled as **"Social Security Number"** or similar variations.  
            - Extract the corresponding value(s) as a string or list.  
            - If missing, set it to **["not found"]**.  

            6. **SubFolder / Department**:  
            - Locate the term **"SubFolder / Department"**.  
            - Extract the corresponding value(s) as a string or list.  
            - If missing, set it to **["not found"]**. 

        Steps to follow:  
            1. Analyze the text content of the image.  
            2. For each field (except 'names'), extract the value(s) as a string or list, or assign ["not found"] if missing.  
            3. For 'names', identify all person names, split them into components where possible, and construct a list of dictionaries with 'first_name', 'middle_name', and 'last_name'. Exclude non-person names.  
            4. Return the result as a JSON object with keys using underscores.
            5. If multiple values correspond to a field, extract all values as a list.
                - If a field (e.g., Carrier, Policy Number, DoB, SSN, SubFolder / Department) has multiple values, extract all of them instead of just one.
                - Return them as a list of strings (e.g., ["ABC Life Insurance", "XYZ Corp"]).
                - If only a single value is present, return it as a list with one element (e.g., ["12345ABC"]).
                - If no values are found, return ["not found"].
       

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
    """Extract JSON from the response text using regex patterns"""
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
        return HTTPException(status_code=500, detail="Failed to extract valid JSON from model response")
    
tfidf_vectorizer, svm_model = load_classification_models()
global client_gpt
client_gpt = get_openai_client()

# API Endpoints
@app.post("/classify")#, response_model=ClassificationResult)
async def classify_document(image_data: ImageData):
    """
    Classify a document based on its content.
    """
    start_time = datetime.now()
    
    try:
        # Ensure models are loaded
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.b64_encoded_image)
        except Exception as e:
            return HTTPException(status_code=400, detail={"message": f"Error processing document: {str(e)}",
                                                          "error_code": "400"})
        
        # Extract text using OCR
        try:
            all_text, _ = get_ocr_vision_api(image_bytes)
        except Exception as e:
            # return HTTPException(status_code=401, detail=f"Google Vision OCR failed: {str(e)}")
            return HTTPException(status_code=401, detail={"message": f"Google Vision OCR failed: {str(e)}",
                                     "error_code": "401"})
        if not all_text:
            return ClassificationResult(
                predicted_class="Other",
                confidence_score=0.0,
                processing_time=(datetime.now() - start_time).total_seconds()
            )
        
        # Process text and classify
        tfidf_features = preprocess_text_for_tfidf(all_text, tfidf_vectorizer)
        prediction = svm_model.predict(tfidf_features)
        
        # Get confidence scores if available
        confidence_score = None
        try:
            if hasattr(svm_model, 'predict_proba'):
                probabilities = svm_model.predict_proba(tfidf_features)[0]
                confidence_score = float(probabilities[prediction[0]])
        except Exception as e:
            print(f"Could not get confidence score: {str(e)}")
        
        # Get class name from prediction
        predicted_class = get_class_name(prediction[0])
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        classification_data = {
            "predicted_class":predicted_class,
            "confidence_score":confidence_score,
            "processing_time":processing_time
        }
        status_code=200
        output_response = {
            "image_name":image_data.image_name,
            "classification_result": classification_data,
            "statusCode": status_code
            }
        return JSONResponse(output_response, status_code=200)
    except Exception as e:
        return HTTPException(status_code=500, detail={"message": f"Error processing document: {str(e)}",
                                                          "error_code": "500"})

def replace_not_found(data):
    if isinstance(data, dict):
        return {k: replace_not_found(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_not_found(item) for item in data]
    elif data == "not found":
        return ""
    return data

@app.post("/extract")#, response_model=ExtractionResponse)
async def extract_document_info(request: ImageData):
    """
    Extract key information from document.
    """
    try:
        # Verify the base64 string
        try:
            image_bytes = base64.b64decode(request.b64_encoded_image)
            
        except Exception as e:
            return HTTPException(status_code=400, detail={"message": f"Error processing document: {str(e)}",
                                                          "error_code": "400"})
        # Get OCR text if not provided
        ocr_text = None #request.ocr_text
        if not ocr_text:
            try:
                ocr_text, _ = get_ocr_vision_api(image_bytes)
            except Exception as e:
                return HTTPException(status_code=401, detail={"message": f"Google Vision OCR failed: {str(e)}",
                                     "error_code": "401"})
        
        # Get OpenAI client
        
        # Prepare messages for the OpenAI API
        messages = []
        
        # Add system prompt
        messages.append({"role": "system", "content": system_prompt})
        
        # Add user prompt with OCR text
        prompt_text = get_extraction_prompt(ocr_text or "OCR text not provided")
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
        response = client_gpt.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0
        )

        # Extract response content
        output_text = response.choices[0].message.content
        
        # Extract JSON from response
        extracted_data = extract_json_from_response(output_text)
        print(extracted_data)
        extracted_data = replace_not_found(extracted_data)
        # Return the extracted data
        status_code=200
        output_response = {
            "image_name":request.image_name,
            "extracted_data": extracted_data,JSONResponse
            "statusCode": status_code
            }
        return (output_response, status_code=200)
    
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return HTTPException(status_code=500, detail={"message": f"Error processing document: {str(e)}",
                                                          "error_code": "500"})
        

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6090)