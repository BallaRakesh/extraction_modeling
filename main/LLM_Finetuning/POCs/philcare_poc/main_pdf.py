import os
import re
import json
import base64
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pdf2image
import pytesseract
from PyPDF2 import PdfReader
from PIL import Image
import cv2
import numpy as np
from tempfile import TemporaryDirectory
import re
import json
import openai
import os, re, base64
from prompts_utility import system_prompt, prompt_text

api_version = "2024-08-01-preview"
azure_endpoint = "https://ai-akshayamalik3400ai955768269900.openai.azure.com/openai/deployments/gpt-4o/chat/completions?2024-08-01-preview"
api_key = "7jlPMIpHvaEdpt53TE7dsXUbyrJEiVRR9uj1UXysqKhw3HKcSKzwJQQJ99BBACHYHv6XJ3w3AAAAACOGFJYF"

client = openai.AzureOpenAI(
    api_key=api_key,
    api_version=api_version,
    azure_endpoint=azure_endpoint
)




# Initialize FastAPI app
app = FastAPI(title="Document Classification and Extraction API")

# Define keywords for classification
keywords = {
    'invoice': ['invoice', 'vat exempt'],
    'claim_form': ['patient', 'claim form'],
    'id_card': ['identification', 'identification card', 'regulation'],
    'license': ['license', 'transportation'],
    'authorization_letter': ['authorization', 'letter of authorization']
}

def get_num_pages(pdf_path: str) -> int:
    """Get the number of pages in the PDF."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PdfReader(f)
            return len(reader.pages)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return 0

def classify_document(text: Optional[str], keywords: Dict[str, List[str]]) -> str:
    """Classify a document based on keyword presence in the text."""
    if not text:
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
                return doc_type
    return "unknown"

def tesseract_generate_ocr_string(image: Image.Image) -> str:
    """Simplified OCR function to extract text from an image."""
    try:
        image = image.convert('RGB')
        np_array = np.array(image)
        image_cv = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
        text = pytesseract.image_to_string(image_cv, lang='eng', config='--psm 6 --oem 3')
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def extract_json_from_response(response_text: str) -> Dict:
    """Extract JSON from AI model response using regex patterns."""
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
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    return {"error": "No valid JSON found in response"}

def extract_content(image_path: str) -> Dict:
    """Placeholder for AI model extraction (e.g., GPT-4o). Replace with actual implementation."""
    # Simulate encoding image to base64 and calling AI model
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    messages = []
    messages.append({"role": "user", "content":prompt_text})
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

    return extract_json_from_response(output_text)

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
def classify_document_with_gpt_image(image_path: str, client, keywords: Dict[str, List[str]]) -> str:
    if not os.path.exists(image_path):
        return "error"
    
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode("utf-8")
    
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
            temperature=0
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

def process_page(image_path: str) -> Dict[str, Optional[str]]:
    """Process a single image for classification and extraction."""
    try:
        image = Image.open(image_path)
        text = tesseract_generate_ocr_string(image)
        if use_gpt_classify:
            doc_type = classify_document_with_gpt_image(image_path, client, keywords)
        else:
            doc_type = classify_document(text, keywords)
        extraction_data = extract_content(image_path)
        
        return {
            "document_type": doc_type,
            "extraction_data": extraction_data
        }
    except Exception as e:
        return {
            "document_type": "error",
            "extraction_data": {"error": f"Processing failed: {str(e)}"}
        }

def process_pdf(pdf_path: str) -> List[Dict]:
    """Process a PDF file, converting each page to an image and analyzing it."""
    num_pages = get_num_pages(pdf_path)
    if num_pages == 0:
        return [{"page": 1, "document_type": "error", "extraction_data": {"error": "Invalid PDF"}}]
    
    results = []
    with TemporaryDirectory() as temp_dir:
        try:
            images = pdf2image.convert_from_path(pdf_path, output_folder=temp_dir, fmt='png')
            for i, image in enumerate(images, start=1):
                image_path = os.path.join(temp_dir, f"page_{i}.png")
                image.save(image_path, 'PNG')
                page_result = process_page(image_path)
                page_result["page"] = i
                results.append(page_result)
        except Exception as e:
            results.append({"page": 1, "document_type": "error", "extraction_data": {"error": f"PDF conversion failed: {str(e)}"}})
    
    return results

def process_image(image_path: str) -> Dict:
    """Process a single image file."""
    result = process_page(image_path)
    return result

def decode_base64_to_image(base64_string: str, temp_dir: str) -> str:
    """Decode base64 string to an image file and return its path."""
    try:
        image_data = base64.b64decode(base64_string)
        image_path = os.path.join(temp_dir, "decoded_image.png")
        with open(image_path, "wb") as f:
            f.write(image_data)
        return image_path
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Base64 decoding failed: {str(e)}")

# Pydantic model for base64 input
class Base64Input(BaseModel):
    base64_string: str

# FastAPI Endpoints
@app.post("/process_file", response_model=List[Dict] | Dict)
async def process_file(file: UploadFile = File(...)):
    """Endpoint to process uploaded PDF or image files."""
    filename = file.filename.lower()
    with TemporaryDirectory() as temp_dir:
        file_path = os.path.join(temp_dir, filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        if filename.endswith(".pdf"):
            results = process_pdf(file_path)
            return JSONResponse(content=results)
        elif filename.endswith((".png", ".jpg", ".jpeg")):
            result = process_image(file_path)
            return JSONResponse(content=result)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use PDF, PNG, or JPG.")

@app.post("/process_base64", response_model=Dict)
async def process_base64(input: Base64Input):
    """Endpoint to process base64 encoded image."""
    with TemporaryDirectory() as temp_dir:
        image_path = decode_base64_to_image(input.base64_string, temp_dir)
        result = process_image(image_path)
        return JSONResponse(content=result)

# Run the app (for testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)