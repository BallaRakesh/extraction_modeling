from transformers import AutoModelForCausalLM, AutoTokenizer
import json  
import time
import pytesseract
from PIL import Image


model_name = "Qwen/Qwen2.5-0.5B"
# model_name = '/home/azureuser/llm_/quantize_models/quantize_models'
model_name = "meta-llama/Llama-3.2-1B-Instruct"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
def get_ocr_tesseract(image_path):
	"""
    Performs OCR (Optical Character Recognition) using Tesseract OCR engine.

    Args:
        image_path (str): Path to the image file.

    Returns:
        tuple: A tuple containing word coordinates (list of dictionaries) and all the extracted text (str).

    """
	img=None
	word_coordinates, all_text = [],""
	print("called Image OCR...", end="")
	try:
		img = Image.open(image_path)
		d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
		all_text = pytesseract.image_to_string(img)
		for i in range(len(d['text'])):
			word = d['text'][i]
			conf = float(d['conf'][i])
			if conf > 0:
				x, y, w, h = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
				word_coordinates.append({
					"word": word,
					"confidence": conf,
					"left": x,
					"top": y,
					"width": w,
					"height": h,
					"x1": x,
					"y1": y,
					"x2": x + w,
					"y2": y + h
				})
	except Exception as e:
		print(f"exception: {e}")	
	finally:
		if hasattr(img,"close"):
			img.close()
	return word_coordinates, all_text

def generate_responce(prompt_):
    
    inputs = tokenizer(prompt_, return_tensors="pt")
    input_length = inputs.input_ids.shape[1]
    
    output = model.generate(**inputs, max_new_tokens=2000, num_return_sequences=1, temperature=0.7)
                # do_sample=True,top_k=50,top_p=0.95, max_new_tokens=2000, max_length=2000
                
    generated_texts = tokenizer.decode(output[0][input_length:], skip_special_tokens=True)
    # generated_texts = tokenizer.batch_decode(output[0], skip_special_tokens=True)
    
    return generated_texts

def extract_text_from_word_coords(word_coordinates: list):
    all_text = ' '.join([item['word'] for item in word_coordinates])
    return all_text
    
# ocr_file = '/home/ntlpt19/Desktop/TF_release/extraction_modeling/data/BNI_Indonesian 1.json'
image_path = '/home/azureuser/llm_/bni_indonesia/BNI_Indonesian.png'
image_path = '/home/azureuser/llm_/bni_indonesia/1.png'
# image_path = '/home/azureuser/llm_/bni_indonesia/2.png'
# image_path = '/home/azureuser/llm_/bni_indonesia4.jpg'
# image_path = '/home/azureuser/llm_/bni_indonesia5.jpg'
# image_path = '/home/azureuser/llm_/bni_indonesia7.png'
# image_path = '/home/azureuser/llm_/bni_indonesia8.jpg'
# image_path = '/home/azureuser/llm_/bni_indonesia9.jpg'
# image_path = '/home/azureuser/llm_/bni_indonesia10.jpg'#need to test with new prompt


# word_coordinates, all_text = get_ocr_tesseract(image_path)
# print(all_text)
# Open and read the JSON file  
# with open(ocr_file, 'r') as file:  
#     ocr_data = json.load(file)

# all_text_data = ocr_data['all_text']

ocr_file = '/home/ntlpt19/Downloads/lc 3/OCR/20886202_29_all_text.txt'
with open(ocr_file, 'r') as file:
    all_text = file.read()
    
print('>>>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>>>.')
all_text_data = all_text
print('all_text_data: ', all_text_data)
print('#################################')
print('#################################')

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype="auto",
    # device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = f"""You are an Expert in document key value extaction
            language: Indonesian (Bahasa Indonesia).
            ocr :{all_text_data} 
            from the provided ocr please extract the possible keys and values
            Note: provide the results in json format"""
            




prompt1 = f"""
You are an Expert in document key-value extraction.
Document Language: Indonesian (Bahasa Indonesia).
OCR text: 
{all_text_data}
Instructions:
- Carefully extract all identifiable key-value pairs from the provided text.
- Each key should match the label found in the text (e.g., "Nama" for Name, "Tempat/Tgl Lahir" for Place/Date of Birth).
- For fields with hierarchical context, such as "PROVINSI" (Province) and "KABUPATEN" (Regency), specify the hierarchy in the JSON format.
- Include any missing values as "null" if not provided.
- Ensure "Berlaku Hingga" (Valid Until) has the value "SEUMUR HIDUP" if present.
  
Note: Return the result in JSON format:
"""

prompt1_1 = f"""
You are an Expert in document key-value extraction.
Document Language: Indonesian (Bahasa Indonesia).
OCR text: 
{all_text_data}
Instructions:
- Carefully extract all identifiable key-value pairs from the provided text.
- "NIK" for Population Identification Number.
- "Gol Darah" for Blood Type, default to "null" if not provided.
- Each key should match the label found in the text (e.g., "Nama" for Name, "Tempat/Tgl Lahir" for Place/Date of Birth).
- For fields with hierarchical context, such as "PROVINSI" (Province) and "KABUPATEN" (Regency), specify the hierarchy in the JSON format.
- Include any missing values as "null" if not provided.
- "Kewarganegaraan" for Nationality.
- Ensure "Berlaku Hingga" (Valid Until) value.
  
Note: Return the result in JSON format:
"""



prompt2 = f"""
You are an Expert in document key-value extraction.
Document Language: Indonesian (Bahasa Indonesia).
OCR text: 
{all_text_data}
Instructions:
- Carefully extract all identifiable key-value pairs from the provided text.
- Match each key exactly as it appears in the text, including:
  - "NIK" for Population Identification Number.
  - "Nama" for Name.
  - "PROVINSI" for Province and "KABUPATEN" for Regency, specifying hierarchy as "Provinsi" and "Kabupaten" in JSON format.
  - "Tempat/Tgl Lahir" for Place/Date of Birth.
  - "Jenis Kelamin" for Gender.
  - "Alamat" for Address, with nested fields for "Desa" (Village/Sub-district), "RT/RW" (RT/RW Number), "Jaga" (Neighborhood), and "Kecamatan" (District).
  - "Agama" for Religion.
  - "Gol Darah" for Blood Type, default to "null" if not provided.
  - "Pekerjaan" for Occupation.
  - "Status Perkawinan" for Marital Status.
  - "Berlaku Hingga" for Valid Until; use "SEUMUR HIDUP" if the text specifies lifetime validity.
  - "Kewarganegaraan" for Nationality.
  - "Tanggal Dikeluarkan" for Issued Date and "Issued Place" for the location, if provided.

- Include any missing values as "null" in the JSON output if not found in the text.
- Organize data in JSON format with a nested structure for "Alamat" to reflect the hierarchy as shown in the example below.
- Ensure correct data extraction for "Berlaku Hingga," using "SEUMUR HIDUP" when present.

Note: Return the result in JSON format:
"""


prompt_main = f""" You are an expert in extracting key-value pairs from OCR text. 
Your task is to analyze and extract all identifiable key-value pairs  from the provided Letter of Credit (LC) text. 

OCR text:
{all_text_data}

### Instructions:
1. Carefully analyze the OCR text to identify all instances of keys embedded between colons.
2. For each identified key, extract its corresponding value, which may span multiple lines until another key is encountered or the text ends.
3. Ensure the extracted key-value pairs are structured in valid JSON format.
4. give the direct responce, no need the python logic to extract the key-value pairs.
### Output format:
The result must be in this JSON format:
{{
    ":52A:": "value1",
    ":31C:": "value2",
}}
"""


start_time = time.time()
prediction = generate_responce(prompt_main)
end_time = time.time()
elapsed_time = end_time - start_time
print('prediction: ', prediction)
print('TIME TAKEN :', elapsed_time)
exit('OK')
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)

import torch
from transformers import pipeline
from huggingface_hub import login
login("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")
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# model_id = "/home/ntlpt19/.cache/huggingface/hub/models--MBZUAI--MobiLlama-05B"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print('>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>')
print(outputs[0]["generated_text"][-1])
