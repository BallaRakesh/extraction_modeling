keys_to_find = [
    "certificate_no",
    "consignee_address",
    "consignee_name",
    "consignor_address",
    "consignor_name",
    "coo_issuer_address",
    "coo_issuer_name",
    "country_of_origin_of_goods",
    "declaration_by_certification",
    "final_destination",
    "from_place",
    "invoice_date",
    "invoice_no",
    "issue_date",
    "lc_date",
    "lc_ref_no",
    "means_of_transport",
    "original_or_copy",
    "page_no",
    "to_place",
    "vessel_details",
    "original_number"
]


# New input format (mapping indices to field names)
field_indices = {
    0: 'coo_issuer_name',
    1: 'coo_issuer_address',
    2: 'ref_no',
    3: 'issue_date',
    4: 'country_of_origin_of_goods',
    5: 'consignee_name',
    6: 'consignee_address',
    7: 'consignor_name',
    8: 'consignor_address',
    9: 'means_of_transport',
    10: 'vessel_details',
    11: 'from_place',
    12: 'to_place',
    13: 'marks_and_no_of_packages',
    14: 'description_of_goods',
    15: 'gross_weight',
    16: 'net_weight',
    17: 'invoice_no',
    18: 'invoice_date',
    19: 'declaration_by_exporter',
    20: 'certificate_stamped',
    21: 'original_or_copy',
    22: 'original_number',
    23: 'lc_ref_no',
    24: 'lc_date',
    25: 'signature',
    26: 'page_no',
    27: 'certificate_no',
    28: 'final_destination',
    29: 'declaration_by_certification',
    30: 'expiry_date'
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

