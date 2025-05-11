keys_to_find = [
    "csh_bill_amount",
    "csh_bill_currency",
    "csh_drawn_under_lc_number",
    "csh_drawn_under_rules",
    "csh_presentation_date",
    "csh_ref_no",
    "currency_amount",
    "doc_charge_instructions",
    "doc_delivery_instruction",
    "drawee_address",
    "drawee_bank_address",
    "drawee_bank_name",
    "drawee_name",
    "drawer_address",
    "drawer_bank_address",
    "drawer_bank_bic",
    "drawer_bank_name",
    "drawer_name",
    "nostro_bank_address",
    "nostro_bank_bic",
    "nostro_bank_name",
    "original",
    "page_no",
    "tenor_indicator",
    "tenor_indicator_date",
    "tenor_indicator_type",
    "tenor_type",
    "usance_tenor",
    "original_number",
    "total_page"
]

# New input format (mapping indices to field names)
field_indices = {
    0: 'csh_presentation_date',
    1: 'csh_ref_no',
    2: 'csh_bill_currency',
    3: 'csh_bill_amount',
    4: 'csh_due_date',
    5: 'csh_drawn_under_rules',
    6: 'csh_drawn_under_lc_number',
    7: 'tenor_type',
    8: 'usance_tenor',
    9: 'tenor_indicator',
    10: 'tenor_indicator_type',
    11: 'tenor_indicator_date',
    12: 'drawer_bank_name',
    13: 'drawer_bank_bic',
    14: 'drawer_bank_address',
    15: 'drawer_bank_country',
    16: 'drawer_bank_bottom_name',
    17: 'drawer_bank_bottom_bic',
    18: 'drawer_bank_bottom_address',
    19: 'drawer_name',
    20: 'drawer_address',
    21: 'drawer_country',
    22: 'drawer_ref_no',
    23: 'drawee_name',
    24: 'drawee_address',
    25: 'drawee_country',
    26: 'drawee_bank_name',
    27: 'drawee_bank_bic',
    28: 'drawee_bank_address',
    29: 'drawee_bank_country',
    30: 'nostro_bank_name',
    31: 'nostro_bank_bic',
    32: 'nostro_bank_address',
    33: 'nostro_bank_country',
    34: 'doc_delivery_instruction',
    35: 'doc_charge_instructions',
    36: 'doc_settlement_instructions',
    37: 'page_no',
    38: 'signed_stamp',
    39: 'document_enclosed',
    40: 'currency_amount',
    41: 'signature',
    42: 'original',
    43: 'total_page',
    44: 'our_charges',
    45: 'your_charges',
    46: 'less_your_charges',
    47: 'plus_our_charges',
    48: 'commissioned_charges',
    49: 'less_commission_charges',
    50: 'swift_charges',
    51: 'postal_charges',
    52: 'other_charges',
    53: 'total_bill_amount',
    54: 'original_number',
    55: 'freight_charges',
    56: 'insurance_charges',
    57: 'draft_number',
    58: 'charges',
    59: 'final_destination',
    60: 'advanced_charges',
    61: 'additional_charges',
    62: 'payment_at_sight',
    63: 'payment_before_sight',
    64: 'advance_payment'
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

