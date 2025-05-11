import os
import requests
from transformers import LayoutLMv2Processor
from transformers import LayoutLMv2ForTokenClassification, AdamW

# processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased", force_download = True, revision="no_ocr")
model_path = '/home/ntlpt19/Downloads/Final_Delivery_Training_itter_2/testing_finetuning/processor__001'

processor = LayoutLMv2Processor.from_pretrained(pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'), 
													config=os.path.join(model_path, 'preprocessor_config.json'), encoding="utf8", errors='ignore')

exit('>>?>>>')
from transformers import AutoProcessor, AutoModel
from datasets import load_dataset


processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-large-uncased", force_download = True)

model_path = '/New_Volume/Rakesh/DATA_LMV2/LMV2_BASE_itter_2/Bol_1044/Best_Model'

processor = LayoutLMv2Processor.from_pretrained(pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'), 
													config=os.path.join(model_path, 'config.json'), revision="no_ocr")

model = LayoutLMv2ForTokenClassification.from_pretrained(
    pretrained_model_name_or_path=os.path.join(model_path, 'pytorch_model.bin'),
    config=os.path.join(model_path, 'config.json'), num_labels=46)

exit('>>>>>>>>')
model = LayoutLMv2ForTokenClassification.from_pretrained('microsoft/layoutlmv2-large-uncased',
                                                            num_labels=5)
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HTTP_PROXY'] = "http://127.0.0.1:7890"
os.environ['HTTPS_PROXY'] = "http://127.0.0.1:7890"
os.environ['ALL_PROXY'] = "socks5://127.0.0.1:7890"
hf_token = os.environ["HUGGINGFACE_TOKEN"] 

model_id = "sentence-transformers/all-MiniLM-L6-v2"
api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}

def query(texts):
    response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

texts = ["How do I get a replacement Medicare card?",
        "What is the monthly premium for Medicare Part B?",
        "How do I terminate my Medicare Part B (medical insurance)?",
        "How do I sign up for Medicare?",
        "Can I sign up for Medicare Part B if I am working and have health insurance through an employer?",
        "How do I sign up for Medicare Part B if I already have Part A?",
        "What are Medicare late enrollment penalties?",
        "What is Medicare and who can get it?",
        "How can I get help with my Medicare Part A and Part B premiums?",
        "What are the different parts of Medicare?",
        "Will my Medicare premiums be higher because of my higher income?",
        "What is TRICARE ?",
        "Should I sign up for Medicare Part B if I have Veterans' Benefits?"]

output = query(texts)