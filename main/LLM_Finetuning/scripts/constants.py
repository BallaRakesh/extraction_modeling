single_file = True
documents_name = "certificate of origin"
train_test_split_ratio= 0.8
key_mapping= {

    "awb_number":"airway_bill_number",
    "awb_date": "airway_bill_date"
}

keys_not_to_consider = ['doc_settlement_instructions', 'signature', \
    'signed_stamp', 'certificate_stamped', 'signature', 'signed_by', 'stamp']
do_ocr = False