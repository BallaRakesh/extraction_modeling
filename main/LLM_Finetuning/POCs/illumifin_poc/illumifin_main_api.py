from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import pickle
from typing import Dict, Any
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
from fastapi.responses import JSONResponse

# Import all other necessary libraries from your original code

app = FastAPI()

# Input model for the API request
class ImageRequest(BaseModel):
    image_name: str
    b64_encoded_image: str
    fileType: str
    other_params: Dict[str, Any]




# Initialize Azure OpenAI client
def get_openai_client(api_key, api_version, azure_endpoint):
    openai_client = openai.AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint
    )
    return openai_client

# Initialize the classification models
def load_classification_models(tfidf_path, svm_path):
    
    try:
        with open(tfidf_path, "rb") as f:
            tfidf_vectorizer = pickle.load(f)
        with open(svm_path, "rb") as f:
            svm_model = pickle.load(f)
    except Exception as e:
        print(f"Failed to load models: {str(e)}")
        return "", ""
    return tfidf_vectorizer, svm_model

# OCR functions
def get_ocr_vision_api(image_data, vision_credential_dict):
    """Extract text from image using Google Vision API"""
    
    # if vision_credentials is None:
    # image_data = base64.b64decode(image_data)
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

def get_class_name(prediction_label, class_mapping):
    """Convert numerical prediction to class name"""
    for category, label in class_mapping.items():
        if label == prediction_label:
            return category
    return "Unknown"

# Extraction functions
def get_extraction_prompt(ocr_all_text, prompt_template):
    return prompt_template.format(ocr_all_text=ocr_all_text)

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
    

# API Endpoints
def classify_document(image_name, all_text, tfidf_vectorizer, svm_model, svm_class_mapping):
    """
    Classify a document based on its content.
    """
    start_time = datetime.now()
    
    # try:
    if not all_text:
        predicted_class="Other",
        confidence_score=0.0,
        processing_time = round((datetime.now() - start_time).total_seconds(), 2)
    else:
        if tfidf_vectorizer and svm_model:
            # Process text and classify
            tfidf_features = preprocess_text_for_tfidf(all_text, tfidf_vectorizer)
            prediction = svm_model.predict(tfidf_features)
            
            # Get confidence scores if available
            confidence_score = None
            try:
                if hasattr(svm_model, 'predict_proba'):
                    probabilities = svm_model.predict_proba(tfidf_features)[0]
                    # confidence_score = round(float(probabilities[prediction[0]]), 2)
                    confidence_score = float(probabilities[prediction[0]])
            except Exception as e:
                print(f"Could not get confidence score: {str(e)}")
            
            # Get class name from prediction
            predicted_class = get_class_name(prediction[0], svm_class_mapping)
            
            # Calculate processing time
            processing_time = round((datetime.now() - start_time).total_seconds(), 2)
        else:
            predicted_class= "unable to predict"
            confidence_score= 0.0
            processing_time = round((datetime.now() - start_time).total_seconds(), 2)

    classification_data = {
        "predicted_class":predicted_class,
        "confidence_score":confidence_score,
        "processing_time_in_sec":processing_time
    }
    return classification_data
    
    # except Exception as e:
    #     return HTTPException(status_code=500, detail={"message": f"Error processing document: {str(e)}",
    #                                                       "error_code": "500"})

def replace_not_found(data):
    if isinstance(data, dict):
        return {k: replace_not_found(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [replace_not_found(item) for item in data]
    elif data == "not found":
        return ""
    return data

def encode_image(image_path:str):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_document_info(b64_encoded_image, client_gpt, gpt_model_name, image_name, ocr_text, fileType,system_prompt, prompt_text):
    """
    Extract key information from document.
    """
    messages = []
    
    # Add system prompt
    messages.append({"role": "system", "content": system_prompt})
    
    # Add user prompt with OCR text
    print(ocr_text)
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    prompt_text_updated = get_extraction_prompt(ocr_text, prompt_text)
    print(prompt_text_updated)
    print('######################################')
    print('######################################')
    messages.append({"role": "user", "content": prompt_text_updated})
    # Add the image
    messages.append({
        "role": "user", 
        "content": [
            {
                "type": "image_url", 
                "image_url": {
                    "url": f"data:image/{fileType};base64,{b64_encoded_image}"
                }
            }
        ]
    })

    # Call the OpenAI API
    response = client_gpt.chat.completions.create(
        model=gpt_model_name,
        messages=messages,
        temperature=0
    )

    # Extract response content
    output_text = response.choices[0].message.content
    
    # Extract JSON from response
    extracted_data = extract_json_from_response(output_text)
    print(extracted_data)
    extracted_data = replace_not_found(extracted_data)

    return extracted_data


def format_results(image_name, classification_results, extraction_results):
    print('classification_results')
    print(classification_results)
    formatted_output = {
        image_name: {
            "predicted_class": classification_results.get("predicted_class", "not found"),
            # "confidence": f"{classification_results.get('confidence_score', 0.0):.2f}",
            "confidence": classification_results.get('confidence_score', 0.0),
            "keys_extraction": {},
            "keys_bboxes": {},
            "keys_confidence": {}
        }
    }

    for key, value in extraction_results.items():
        if any(value):  # Check if the key contains non-empty values
            if key == "names" and isinstance(value, list) and all(isinstance(v, dict) for v in value):
                # Process names field separately, but do not keep 'names' as a key
                merged_names = {
                    "first_name": [],
                    "middle_name": [],
                    "last_name": []
                }

                for name_entry in value:
                    merged_names["first_name"].append(name_entry.get("first_name", ""))
                    merged_names["middle_name"].append(name_entry.get("middle_name", ""))
                    merged_names["last_name"].append(name_entry.get("last_name", ""))

                # Add merged names directly instead of under 'names' key
                for name_key, name_value in merged_names.items():
                    formatted_output[image_name]["keys_extraction"][name_key] = name_value
                    formatted_output[image_name]["keys_bboxes"][name_key] = []  # Maintain empty list for bbox
                    formatted_output[image_name]["keys_confidence"][name_key] = []  # Maintain empty list for confidence
            else:
                formatted_output[image_name]["keys_extraction"][key] = value
                formatted_output[image_name]["keys_bboxes"][key] = []  # Maintain empty list for bbox
                formatted_output[image_name]["keys_confidence"][key] = []  # Maintain empty list for confidence

    return formatted_output



# Load models (we'll make this a function to call when needed)
def load_models(tfidf_path: str, svm_path: str):
    tfidf_vectorizer, svm_model = load_classification_models(tfidf_path, svm_path)
    return tfidf_vectorizer, svm_model

# Initialize OpenAI client
def initialize_client(api_key: str, api_version: str, azure_endpoint: str):
    return get_openai_client(api_key, api_version, azure_endpoint)

# Classification endpoint
'''@app.post("/classify")
async def classify_image(request: ImageRequest):
    try:
        # Extract parameters
        image_name = request.image_name
        b64_encoded_image = request.b64_encoded_image
        other_params = request.other_params

        # Required parameters from other_params
        required_params = [
            "TFIDF_PATH", "SVM_PATH", "SVM_CLASS_MAPPING", "VISION_CREDENTIAL_DICT"
        ]
        for param in required_params:
            if param not in other_params:
                return HTTPException(status_code=400, detail=f"Missing required parameter: {param}")

        # Decode base64 image
        # image_data = base64.b64decode(b64_encoded_image)
        try:
            image_bytes = base64.b64decode(b64_encoded_image)
            
        except Exception as e:
            return HTTPException(status_code=400, detail={"message": f"Error processing document: {str(e)}",
                                                          "error_code": "400"})
        # Load models
        tfidf_vectorizer, svm_model = load_models(
            other_params["TFIDF_PATH"],
            other_params["SVM_PATH"]
        )

        # Perform OCR
        ocr_text = None #request.ocr_text
        if not ocr_text:
            try:
                all_text, _ = get_ocr_vision_api(image_bytes, other_params["VISION_CREDENTIAL_DICT"])
            except Exception as e:
                return HTTPException(status_code=401, detail={"message": f"Google Vision OCR failed: {str(e)}",
                                     "error_code": "401"})
        
        # all_text, _ = get_ocr_vision_api(image_bytes, other_params["VISION_CREDENTIAL_DICT"])
        # Classify document
        classification_results = classify_document(
            image_name,
            all_text,
            tfidf_vectorizer,
            svm_model,
            other_params["SVM_CLASS_MAPPING"]
        )

        return {
            "status": "success",
            "image_name": image_name,
            "classification_results": classification_results
        }

    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error in classification: {str(e)}")

# Extraction endpoint
@app.post("/extract")
async def extract_image_info(request: ImageRequest):
    try:
        # Extract parameters
        image_name = request.image_name
        b64_encoded_image = request.b64_encoded_image
        file_type = request.fileType
        other_params = request.other_params

        # Required parameters from other_params
        required_params = [
            "API_VERSION", "AZURE_ENDPOINT", "API_KEY", "VISION_CREDENTIAL_DICT",
            "SYSTEM_PROMPT", "PROMPT_TEXT", "GPT_MODEL_NAME"
        ]
        for param in required_params:
            if param not in other_params:
                return HTTPException(status_code=400, detail=f"Missing required parameter: {param}")

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(b64_encoded_image)
            
        except Exception as e:
            return HTTPException(status_code=400, detail={"message": f"Error processing document: {str(e)}",
                                                          "error_code": "400"})
        # image_data = base64.b64decode(b64_encoded_image)

        # Initialize OpenAI client
        client_gpt = initialize_client(
            other_params["API_KEY"],
            other_params["API_VERSION"],
            other_params["AZURE_ENDPOINT"]
        )

        # Perform OCR
        ocr_text = None #request.ocr_text
        if not ocr_text:
            try:
                all_text, _ = get_ocr_vision_api(image_bytes, other_params["VISION_CREDENTIAL_DICT"])
            except Exception as e:
                return HTTPException(status_code=401, detail={"message": f"Google Vision OCR failed: {str(e)}",
                                     "error_code": "401"})

        # Perform OCR
        # all_text, _ = get_ocr_vision_api(image_bytes, other_params["VISION_CREDENTIAL_DICT"])

        # Extract document info
        extraction_results = extract_document_info(
            b64_encoded_image,
            client_gpt,
            other_params["GPT_MODEL_NAME"],
            image_name,
            all_text,
            file_type,
            other_params["SYSTEM_PROMPT"],
            other_params["PROMPT_TEXT"]
        )

        return {
            "status": "success",
            "image_name": image_name,
            "extraction_results": extraction_results
        }

    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error in extraction: {str(e)}")

'''

TFIDF_PATH = '/home/ntlpt19/Desktop/TF_release/extraction_modeling/main/LLM_Finetuning/POCs/illumifin_poc/models/tf_idf_model.pkl'
SVM_PATH = '/home/ntlpt19/Desktop/TF_release/extraction_modeling/main/LLM_Finetuning/POCs/illumifin_poc/models/svm_model.pkl'
tfidf_vectorizer, svm_model = load_classification_models(TFIDF_PATH, SVM_PATH)

# Combined endpoint for classification and extraction
@app.post("/illumifin_document/process")
async def process_image(request: ImageRequest):
    try:
        # Extract parameters
        image_name = request.image_name
        b64_encoded_image = request.b64_encoded_image
        file_type = request.fileType
        other_params = request.other_params

        # Required parameters from other_params for both classification and extraction
        required_params = ["SVM_CLASS_MAPPING", "VISION_CREDENTIAL_DICT",
            "API_VERSION", "AZURE_ENDPOINT", "API_KEY", "SYSTEM_PROMPT", "PROMPT_TEXT", "GPT_MODEL_NAME"]
        
        for param in required_params:
            if param not in other_params:
                return HTTPException(status_code=400, detail=f"Missing required parameter: {param}")

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(b64_encoded_image)
        except Exception as e:
            return HTTPException(status_code=400, detail={"message": f"Error decoding base64 image: {str(e)}",
                                                            "error_code": "400"})

        # Perform OCR
        ocr_text = None  # Could be extended to accept pre-provided OCR text if needed
        if not ocr_text:
            try:
                all_text, _ = get_ocr_vision_api(image_bytes, other_params["VISION_CREDENTIAL_DICT"])
            except Exception as e:
                return HTTPException(status_code=401, detail={"message": f"Google Vision OCR failed: {str(e)}",
                                                                "error_code": "401"})

        # Load classification models


        # Classify document
        classification_results = classify_document(
            image_name,
            all_text,
            tfidf_vectorizer,
            svm_model,
            other_params["SVM_CLASS_MAPPING"]
        )

        # Initialize OpenAI client
        client_gpt = get_openai_client(
            other_params["API_KEY"],
            other_params["API_VERSION"],
            other_params["AZURE_ENDPOINT"]
        )

        # Extract document info
        extraction_results = extract_document_info(
            b64_encoded_image,
            client_gpt,
            other_params["GPT_MODEL_NAME"],
            image_name,
            all_text,
            file_type,
            other_params["SYSTEM_PROMPT"],
            other_params["PROMPT_TEXT"]
        )

        # Combine results
        combined_results = format_results(image_name, classification_results, extraction_results)

        # return {
        #     "status": "success",
        #     "image_name": image_name,
        #     "results": combined_results
        # }
        return JSONResponse(combined_results, status_code=200)
    except Exception as e:
        return HTTPException(status_code=500, detail=f"Error in processing: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Example usage of the API would look like:
"""
{
    "image_name": "abc.png",
    "b64_encoded_image": "Base64_string_of_the_image",
    "fileType": "png",
    "other_params": {
        "TFIDF_PATH": "/path/to/tf_idf_model.pkl",
        "SVM_PATH": "/path/to/svm_model.pkl",
        "API_VERSION": "2024-08-01-preview",
        "AZURE_ENDPOINT": "https://your-endpoint.openai.azure.com/",
        "API_KEY": "your-api-key",
        "SVM_CLASS_MAPPING": {...},
        "VISION_CREDENTIAL_DICT": {...},
        "SYSTEM_PROMPT": "...",
        "PROMPT_TEXT": "...",
        "GPT_MODEL_NAME": "gpt-4o"
    }
}
"""