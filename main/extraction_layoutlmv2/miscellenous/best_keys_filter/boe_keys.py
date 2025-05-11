keys_to_find = keys = [
    "amount_in_words",
    "bill_exchange_date",
    "bill_exchange_no",
    "boe_amount",
    "boe_currency",
    "diclaration_by",
    "drawee_bank_address",
    "drawee_bank_name",
    "drawer_bank_address",
    "drawer_bank_name",
    "drawer_name",
    "lc_date",
    "tenore_details",
    "lc_ref_no",
    "drawee_address",
    "drawee_name",
    "drawer_address",
    "original_or_copy",
    "issue_date"
]

# New input format (mapping indices to field names)
field_indices = {
    0: 'bill_exchange_no',
    1: 'bill_exchange_date',
    2: 'boe_currency',
    3: 'boe_amount',
    4: 'drawee_name',
    5: 'drawee_address',
    6: 'drawee_country',
    7: 'drawer_name',
    8: 'drawer_address',
    9: 'drawer_country',
    10: 'country_of_origin',
    11: 'invoice_no',
    12: 'invoice_date',
    13: 'invoice_currency',
    14: 'invoice_amount',
    15: 'tenor_type',
    16: 'usance_tenor',
    17: 'tenor_indicator',
    18: 'indicator_type',
    19: 'indicator_date',
    20: 'invoice_due_date',
    21: 'original_or_copy',
    22: 'original_number',
    23: 'lc_ref_no',
    24: 'lc_date',
    25: 'signature',
    26: 'drawer_bank_name',
    27: 'drawee_bank_name',
    28: 'issue_place',
    29: 'stamp',
    30: 'amount_in_words',
    31: 'diclaration_by',
    32: 'drawee_bank_address',
    33: 'drawer_bank_address',
    34: 'total_original_number',
    35: 'goods_discription',
    36: 'tenore_details',
    37: 'issuing_bank',
    38: 'issuing_bank_address',
    39: 'stamp_endorsed',
    40: 'issue_date'
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

