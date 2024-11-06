# from transformers import LayoutLMv3ForSequenceClassification
from transformers import AutoProcessor
from transformers import AutoProcessor, AutoModelForTokenClassification
from datetime import datetime
import os
from step4_result_generation import image_result

if __name__ == "__main__":
	folder_path: str = "/home/ntlpt19/Desktop/TF_release/doc_enclosed_files/ROOT/doc_enclose_eval_136"
	count = 0
	print("Loading Model...")
	t_start = datetime.now()
	model= "Best_Model"
	model_path = os.path.join(folder_path,model)
	# AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base",)
	model = AutoModelForTokenClassification.from_pretrained(model_path)
	processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
	device = 'cpu'
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)
	t_end = datetime.now()
	path = "Images"
	print("Loaded Model")
	print("Time Taken:", t_end - t_start)
	for file in os.listdir(os.path.join(folder_path, "Images")):
		# try:
			count += 1
			ans = image_result(file, model, processor, device)
		# except Exception as e:
		# 	print(e)
		# 	continue
	print("***************Processed all the files!****************")
