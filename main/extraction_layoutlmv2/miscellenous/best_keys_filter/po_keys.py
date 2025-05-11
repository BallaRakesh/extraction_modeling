keys_to_find = [
    "quantity",
    "unit_price",
    "po_number",
    "unit",
    "issue_date",
    "remitter_drawee_buyer_importer_billto_shippedto_despacthto_name",
    "remitter_drawee_buyer_importer_billto_shippedto_despacthto_address",
    "drawer_exporter_seller_beneficiary_issuer_supplier_name",
    "drawer_exporter_seller_beneficiary_issuer_supplier_address",
    "page_no",
    "total_amount_in_figure",
    "currency",
    "cst_no",
    "payment_terms",
    "ecc_no",
    "drawer_ref_no",
    "total_amount_in_words",
    "dimension",
    "incoterm",
    "original_or_copy",
    "total_pages"
]


# New input format (mapping indices to field names)
field_indices = {
    0: 'po_number',
    1: 'issue_date',
    2: 'drawer_exporter_seller_beneficiary_issuer_supplier_name',
    3: 'drawer_exporter_seller_beneficiary_issuer_supplier_address',
    4: 'remitter_drawee_buyer_importer_billto_shippedto_despacthto_name',
    5: 'remitter_drawee_buyer_importer_billto_shippedto_despacthto_address',
    6: 'item_description',
    7: 'drawer_ref_no',
    8: 'drawee_ref_no',
    9: 'unit',
    10: 'quantity',
    11: 'unit_price',
    12: 'total_amount_in_figure',
    13: 'tax_amount',
    14: 'total_amount_in_words',
    15: 'mode_of_dispatch',
    16: 'incoterm',
    17: 'delivery_terms',
    18: 'insurance_issuer_name',
    19: 'payment_terms',
    20: 'signed_by',
    21: 'signature',
    22: 'stamped',
    23: 'documents_required',
    24: 'freight_and_transportation_information',
    25: 'packaging_details',
    26: 'page_no',
    27: 'total_pages',
    28: 'currency',
    29: 'price_conditions',
    30: 'warranty_terms',
    31: 'quotation_no',
    32: 'taxes_condition',
    33: 'discount_condition',
    34: 'transport_condition',
    35: 'vat_no',
    36: 'ecc_no',
    37: 'cst_no',
    38: 'tin_no',
    39: 'destination',
    40: 'dimension',
    41: 'diclaration_by',
    42: 'original_or_copy',
    43: 'at_place'
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

