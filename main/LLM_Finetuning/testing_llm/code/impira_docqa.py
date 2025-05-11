from transformers import pipeline
import time
from datetime import datetime

# Initialize the pipeline
nlp = pipeline(
    "document-question-answering",
    model="impira/layoutlm-document-qa",
)

# List of keys to extract
keys_to_extract = [
    'transaction_date',
    'transaction_amonunt_value',  # Note: seems to be a typo, should be 'amount'?
    'invoice_no',
    'date_of_invoice',
    'invoice_currency',
    'invoice_amount',
    'invoice_discount_ccy',
    'invoice_discount_amount',
    'invoice_tax_ccy',
    'invoice_tax_amount',
    'invoice_amount_in_words',
    'total_quantity_of_goods',
    'rate_per_unit',
    'gross_weight',
    'net_weight',
    'dimension',
    'hs_code_no',
    'delivery_terms',
    'tenor_type',
    'incoterm',
    'country_of_origin_of_goods',
    'port_of_loading',
    'port_of_discharge',
    'country_of_final_destination',
    'pre_carriage_by',
    'vessel_flight_no',
    'bill_of_lading_no',
    'bill_of_lading_date',
    'awb_number',
    'awb_date',
    'page_no'
]

# Path to the document
document_path = "/home/ntlpt19/Downloads/Final_Delivery_Training_itter_5/Eval_data/CI/v2/Images/Invoice(2012_08_21_18_12_07_4875)_489_0.png"

# Process each key and print result with timestamp
for key in keys_to_extract:
    start_time = time.time()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create question for current key
    question = f"What is the {key}?"
    
    # Get answer from the pipeline
    result = nlp(document_path, question)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Print result with timestamp and processing time
    print(f"[{current_time}] Key: {key}")
    print(f"Result: {result}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print("-" * 50)
