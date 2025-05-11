keys_to_find = [
    "beneficiary_account_no",
    "drawer_exporter_seller_beneficiary_issuer_supplier_address",
    "beneficiary_bank_address",
    "beneficiary_bank",
    "beneficiary_bank_swift",
    "drawer_exporter_seller_beneficiary_issuer_supplier_name",
    "consignee_address",
    "consinee_name",
    "country_of_origin_of_goods",
    "invoice_date",
    "goods_description",
    "total_amount_in_numeric",
    "total_amount_in_words",
    "currency_in_numeric",
    "performa_invoice_no",
    "payment_terms",
    "purchase_order_no_and_agreement_ref_no",
    "remitter_drawee_buyer_importer_billto_shippedto_despacthto_address",
    "remitter_drawee_buyer_importer_billto_shippedto_despacthto_name",
    "quantity",
    "page_no",
    "total_pages"
]


# New input format (mapping indices to field names)
field_indices = {
    0: 'performa_invoice_no',
    1: 'invoice_date',
    2: 'invoice_due_date',
    3: 'drawer_ref_no',
    4: 'drawer_exporter_seller_beneficiary_issuer_supplier_name',
    5: 'drawer_exporter_seller_beneficiary_issuer_supplier_address',
    6: 'remitter_drawee_buyer_importer_billto_shippedto_despacthto_name',
    7: 'remitter_drawee_buyer_importer_billto_shippedto_despacthto_address',
    8: 'beneficiary_bank',
    9: 'beneficiary_bank_address',
    10: 'beneficiary_bank_swift',
    11: 'beneficiary_account_no',
    12: 'shipper_name',
    13: 'shipper_address',
    14: 'consignor_name',
    15: 'consignor_address',
    16: 'notify_party_name',
    17: 'notify_party_address',
    18: 'consinee_name',
    19: 'consignee_address',
    20: 'purchase_order_no_and_agreement_ref_no',
    21: 'order_date',
    22: 'goods_description',
    23: 'quantity',
    24: 'unit',
    25: 'rate_unit',
    26: 'total_amount_in_numeric',
    27: 'currency_in_numeric',
    28: 'total_amount_in_words',
    29: 'tax_amount',
    30: 'amount_due',
    31: 'advance_amount',
    32: 'drawee_buyer_importer_tin_no',
    33: 'drawer_exporter_tin_no',
    34: 'country_of_origin_of_goods',
    35: 'country_of_final_destination',
    36: 'pre_carriage_by',
    37: 'port_of_loading',
    38: 'port_of_discharge',
    39: 'from',
    40: 'to',
    41: 'final_destination',
    42: 'vessel_name',
    43: 'gross_weight',
    44: 'net_weight',
    45: 'page_no',
    46: 'total_pages',
    47: 'signature',
    48: 'stamped',
    49: 'signed_by',
    50: 'terms_of_delivery',
    51: 'payment_terms',
    52: 'packaging_description',
    53: 'partial_shipment',
    54: 'transhipment',
    55: 'shipment_date',
    56: 'transport_details',
    57: 'importer_your_ref_no',
    58: 'exporter_our_ref_no',
    59: 'mode_of_despatch',
    60: 'shipment_date',
    61: 'tolerance_of_quantity',
    62: 'incoterms',
    63: 'diclaration_by',
    64: 'original_or_copy',
    65: 'presentation_period'
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

