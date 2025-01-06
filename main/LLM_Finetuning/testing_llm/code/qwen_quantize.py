from transformers import AutoModelForCausalLM, AutoTokenizer
import json  
from awq import AutoAWQForCausalLM
import time
import pytesseract
from PIL import Image


# model_name = "Qwen/Qwen2.5-0.5B"
model_name = '/home/azureuser/llm_/quantize_models/quantize_models'
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


word_coordinates, all_text = get_ocr_tesseract(image_path)
print(all_text)
# Open and read the JSON file  
# with open(ocr_file, 'r') as file:  
#     ocr_data = json.load(file)

# all_text_data = ocr_data['all_text']
###################
###################
all_text_data = all_text
###################
###################

print('all_text_data: ', all_text_data)
print('#################################')
print('#################################')
model = AutoAWQForCausalLM.from_quantized(
    model_name,
    # torch_dtype="auto",
    # device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
print('>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>')
print('>>>>>>>>>>>>>')
print(response)