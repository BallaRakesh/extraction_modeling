keys_to_find = [
    "agent_address",
    "agent_name",
    "airport_of_departure",
    "airport_of_destination",
    "at_place",
    "awb_bill_issue_date",
    "awb_original_number",
    "awb_original_or_copy",
    "carrier_address",
    "carrier_name",
    "consignee_address",
    "consignee_name",
    "declared_value_of_carriage",
    "dimension",
    "flight_details",
    "freight_collect_or_prepaid",
    "goods_description",
    "gross_weight",
    "house_awb_bill_no",
    "invoice_date",
    "invoice_number",
    "net_weight",
    "notify_party_address",
    "notify_party_name",
    "shipper_address",
    "shipper_name",
    "total_original"
]

# New input format (mapping indices to field names)
field_indices = {
    0: 'transaction_date',
    1: 'awb_bill_no',
    2: 'master_awb_bill_no',
    3: 'house_awb_bill_no',
    4: 'awb_bill_issue_date',
    5: 'flight_no',
    6: 'flight_date',
    7: 'shipper_name',
    8: 'shipper_address',
    9: 'shipper_country',
    10: 'consignee_name',
    11: 'consignee_address',
    12: 'consignee_country',
    13: 'notify_party_name',
    14: 'notify_party_address',
    15: 'notify_party_country',
    16: 'carrier_name',
    17: 'carrier_address',
    18: 'agent_name',
    19: 'agent_address',
    20: 'place_of_receipt',
    21: 'airport_of_departure',
    22: 'airport_of_destination',
    23: 'final_destination',
    24: 'declared_value_of_carriage',
    25: 'amount_insurance',
    26: 'goods_description',
    27: 'gross_quantity',
    28: 'gross_weight',
    29: 'net_weight',
    30: 'good_marks',
    31: 'invoice_number',
    32: 'invoice_date',
    33: 'lc_no',
    34: 'lc_date',
    35: 'freight_collect_or_prepaid',
    36: 'freight_collected_at',
    37: 'signed_by_carrier',
    38: 'signed_by_agent',
    39: 'awb_original_number',
    40: 'awb_original_or_copy',
    41: 'flight_details',
    42: 'declared_value_of_custom',
    43: 'carriage_condition',
    44: 'dimension',
    45: 'at_place',
    46: 'stamp',
    47: 'signature',
    48: 'consignor_name',
    49: 'acceptance_instructions',
    50: 'shipment_date',
    51: 'stamped_onboard_date',
    52: 'stamped_flight_date',
    53: 'master_name',
    54: 'signed_by_master',
    55: 'stamped_vessel_or_cargo_name'
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

