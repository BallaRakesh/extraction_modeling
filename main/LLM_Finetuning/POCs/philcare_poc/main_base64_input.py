import os
import re
import json
import base64
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pytesseract
from PIL import Image
import cv2
import numpy as np
import io
import openai
from prompts_utility import system_prompt, prompt_text
from typing import List, Dict, Optional

# Configure logging
logging.basicConfig(
    filename='image_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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


# Initialize FastAPI app
app = FastAPI(title="Image Classification and Extraction API")

# Define keywords for classification
keywords = {
    'invoice': ['invoice', 'vat exempt'],
    'claim_form': ['patient', 'claim form'],
    'id_card': ['identification', 'identification card', 'regulation'],
    'license': ['license', 'transportation'],
    'authorization_letter': ['authorization', 'letter of authorization']
}

def classify_document(image, keywords: Dict[str, list]) -> str:
    """Classify a document based on keyword presence in the text."""
    logger.info("Starting keyword-based document classification")
    text = tesseract_generate_ocr_string(image)
    
    if not text:
        logger.warning("No text provided for classification")
        return "error"
    
    text = text.lower()
    scores = {doc_type: 0 for doc_type in keywords}
    
    for doc_type, kw_list in keywords.items():
        for kw in kw_list:
            if kw in text:
                scores[doc_type] += 1
    
    max_score = max(scores.values())
    if max_score >= 1:
        for doc_type, score in scores.items():
            if score == max_score:
                logger.info(f"Document classified as: {doc_type}")
                return doc_type
    logger.info("Document classified as: unknown")
    return "unknown"

def tesseract_generate_ocr_string(image: Image.Image) -> str:
    """Simplified OCR function to extract text from an image."""
    logger.info("Starting OCR text extraction")
    try:
        image = image.convert('RGB')
        np_array = np.array(image)
        image_cv = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
        text = pytesseract.image_to_string(image_cv, lang='eng', config='--psm 6 --oem 3')
        logger.info("OCR text extraction completed successfully")
        return text
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return ""

def extract_json_from_response(response_text: str) -> Dict:
    """Extract JSON from AI model response using regex patterns."""
    logger.info("Extracting JSON from AI response")
    patterns = [
        r"'''<json response>\s*(\{.*?\})\s*'''",
        r"```<json response>\s*(\{.*?\})\s*```",
        r"```json\s*(\{.*?\})\s*```",
        r"'''json\s*(\{.*?\})\s*'''",
        r"\*\*<json response>\*\*\s*(\{.*?\})\s*\*\*<json response>\*\*",
        r"<json response>\s*(\{.*?\})\s*</json response>"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response_text, re.DOTALL)
        if match:
            json_str = match.group(1)
            try:
                logger.info("JSON extracted successfully")
                return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON format in response")
                continue
    
    logger.error("No valid JSON found in response")
    return {"error": "No valid JSON found in response"}

def extract_content(image_base64: str) -> Dict:
    """Extract content from base64-encoded image using AI model."""
    logger.info("Starting content extraction with AI model")
    messages = []
    messages.append({"role": "user", "content":prompt_text})
    # messages.append({"role": "user", "content": gpt_4o_general_table_extraction_with_csv_response_user_prompt})
    messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]})
    print('EXTRACTION STARTED')
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
            seed=42 
        )
        output_text = response.choices[0].message.content
        logger.info("AI model response received")
        return extract_json_from_response(output_text)
    except Exception as e:
        logger.error(f"AI content extraction failed: {e}")
        return {"error": f"AI extraction failed: {str(e)}"}


def get_classification_prompt(doc_types: List[str], keywords: Dict[str, List[str]]) -> str:
    """Generate a detailed prompt for GPT-based document classification."""
    prompt = (
        "You are tasked with classifying a document based on its content in the provided image. "
        "The document must be classified as one of the following types: "
        f"{', '.join(doc_types)}, or 'unknown' if it doesn’t match any type.\n\n"
        "Here are the document types with examples of key terms or phrases typically found in each:\n"
    )
    
    for doc_type, kw_list in keywords.items():
        prompt += f"- {doc_type}: Contains terms like {', '.join(kw_list)}.\n"
    
    prompt += (
        "\nAnalyze the image and determine the document type based on its content. "
        "Return your response in this exact JSON format:\n"
        "```json\n"
        "{\"document_type\": \"<classified_type>\"}\n"
        "```\n"
        "Where <classified_type> is one of the listed types or 'unknown'. "
        "Do not include any additional text outside the JSON."
    )
    return prompt

# Example usage in classify_document_with_gpt_image
def classify_document_with_gpt_image(image_base64: str, client, keywords: Dict[str, List[str]]) -> str:
    # if not os.path.exists(image_path):
    #     return "error"
    
    # with open(image_path, "rb") as image_file:
    #     image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
    doc_types = list(keywords.keys())
    prompt_text = get_classification_prompt(doc_types, keywords)
    
    messages = [
        {"role": "user", "content": prompt_text},
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}]}
    ]
    print('CLASSIFICATION STARTED')
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            
        )
        output_text = response.choices[0].message.content.strip()
        
        # Parse the JSON response
        try:
            result = extract_json_from_response(output_text)
            doc_type = result.get("document_type", "unknown")
            if doc_type in doc_types or doc_type == "unknown":
                return doc_type
            return "unknown"
        except json.JSONDecodeError:
            return "unknown"  # Fallback if response isn’t valid JSON
    except Exception as e:
        print(f"GPT classification error: {e}")
        return "error"
    


from constants import use_gpt_classify

def process_image(image_base64: str) -> Dict:
    """Process a base64-encoded image for classification and extraction."""
    logger.info("Starting image processing")
    try:
        # Decode base64 to image
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        logger.info("Base64 image decoded successfully")
        
        # OCR text extraction
        
        # Classification (using GPT or keyword-based)
        doc_type = classify_document_with_gpt_image(image_base64, client, keywords) if use_gpt_classify else classify_document(image, keywords)
        
        # Content extraction
        extraction_data = extract_content(image_base64)
        
        logger.info(f"Image processing completed. Document type: {doc_type}")
        return {
            "document_type": doc_type,
            "extraction_data": extraction_data
        }
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        return {
            "document_type": "error",
            "extraction_data": {"error": f"Processing failed: {str(e)}"}
        }

# Pydantic model for input payload
class ImageInput(BaseModel):
    image_name: str
    b64_encoded_image: str
    fileType: str

# FastAPI Endpoint
@app.post("/process_image", response_model=Dict)
async def process_image_endpoint(input: ImageInput):
    """Endpoint to process base64-encoded image from JSON payload."""
    logger.info(f"Received request to process image: {input.image_name}")
    if input.fileType.lower() not in ["png", "jpg", "jpeg"]:
        logger.error(f"Unsupported file type: {input.fileType}")
        raise HTTPException(status_code=400, detail="Unsupported file type. Use PNG, JPG, or JPEG.")
    
    result = process_image(input.b64_encoded_image)
    logger.info("Request processing completed")
    print('DONE')
    return JSONResponse(content=result)

# Run the app
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000)