keys_to_find = [
    "beneficiary_address",
    "beneficiary_drawer_exporter_seller_supplier_name",
    "consignee_address",
    "consignee_name",
    "date_of_invoice",
    "declaration",
    "description_of_goods",
    "dimension",
    "gross_weight",
    "incoterm",
    "invoice_no",
    "lc_ref_no",
    "pre_carriage_by",
    "remitter_address",
    "remitter_drawee_applicant_importer_buyer_name",
    "total_quantity_of_goods",
    "vessel_flight_no",
    "vessel_name",
    "page_no",
    "original_copy",
    "net_weight"
]


# New input format (mapping indices to field names)
field_indices = {
    0: 'awb_date',
    1: 'awb_number',
    2: 'beneficiary_address',
    3: 'beneficiary_drawer_exporter_seller_supplier_name',
    4: 'beneficiary_drawer_exporter_seller_supplier_tin_no',
    5: 'bill_of_lading_date',
    6: 'bill_of_lading_number',
    7: 'consignee_address',
    8: 'consignee_name',
    9: 'consignor_address',
    10: 'consignor_name',
    11: 'shipper_name',
    12: 'shipper_address',
    13: 'country_of_final_destination',
    14: 'country_of_origin_origin_of_goods',
    15: 'date_of_invoice',
    16: 'declaration',
    17: 'description_of_goods',
    18: 'dimension',
    19: 'gross_weight',
    20: 'incoterm',
    21: 'invoice_amount',
    22: 'invoice_currency',
    23: 'invoice_no',
    24: 'lc_date',
    25: 'lc_ref_no',
    26: 'net_weight',
    27: 'notify_party_address',
    28: 'notify_party_name',
    29: 'original_copy',
    30: 'page_no',
    31: 'payment_terms_terms_of_delivery_&_payment',
    32: 'port_of_discharge',
    33: 'port_of_loading',
    34: 'pre_carriage_by',
    35: 'rate_per_unit',
    36: 'remitter_drawee_applicant_importer_buyer_name',
    37: 'remitter_address',
    38: 'total_quantity_of_goods',
    39: 'track_reference',
    40: 'transaction_amount_value',
    41: 'transaction_currency',
    42: 'transaction_date',
    43: 'vessel_flight_no',
    44: 'vessel_name',
    45: 'iec_no',
    46: 'stamp',
    47: 'signature'
}

# Reverse lookup: find indices for given field names
result_indices = {key: [idx for idx, field in field_indices.items() if field == key] for key in keys_to_find}

# Flatten the indices into a list
val_list = [idx for indices in result_indices.values() for idx in indices]

# Print the results
print(result_indices)  # Dictionary mapping field names to indices
print(val_list)  # List of all found indices
total_keys = len(field_indices)

keys_not_to_consider = []
for i in range(0, total_keys):
    if i not in val_list:
        keys_not_to_consider.append(i)
print('$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$')
print('$$$$$$$$$$$$$$$$$$$$$$$$$$')
print(keys_not_to_consider)

