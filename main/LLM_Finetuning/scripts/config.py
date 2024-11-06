class config:
    TOKEN_LENGTH = 70
    DEVICE = 'cuda'
    DOCUMENT_CODE = {
    "COO": "certificate of origin", 
    "CS": "covering schedule",
    "BOL": "bill of landing",
    "CI": "commercial invoice",
    "AWB": "airway bill",
    "PI": "performa invoice",
    "IC":"insurance certificate",
    "PO": "performa invoice"
    }
    # THE LAST DIRECTORY YOU SHOULD BE PRESENT IN IS LLM_Finetuning
    CURRENT_DIR = '/home/gpu1admin/rakesh'
    SHEET_KEY = 'ground truth'
    SHEET_NAME = ''
    EXTENSION = '.xlsx'
    DOCUMENT_NAME = 'BOL'
    MODEL_PATH = '/home/gpu1admin/rakesh/ITF-Training/training/LLM_Finetuning/scripts/merged_model_coo'
    GROUND_TRUTH_SHEET = 'ground_truth_key_names_change'
    
    ROOT_PATH = "/home/ntlpt19/LLM_training/EVAL/BOL"
    GV_KEY = "/home/ntlpt19/Desktop/TF_release/training_code/ITF-Training/spheric-time-383904-f1b421d86eef.json"
    RESULT_FOLDER = '/home/ntlpt19/LLM_training/EVAL/CS/CS_results/text_files'