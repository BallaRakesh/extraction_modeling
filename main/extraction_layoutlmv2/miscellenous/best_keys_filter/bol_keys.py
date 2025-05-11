keys_to_find = [
    "bill_of_lading_issue_date",
    "bill_of_lading_number",
    "bol_original_number",
    "bol_original_or_copy",
    "consignee_address",
    "consignee_name",
    "container_number",
    "dimension",
    "final_destination",
    "freight_collect_or_prepaid",
    "gross_weight",
    "lc_ref_number",
    "measurement",
    "page_no",
    "place_of_issue",
    "port_of_discharge",
    "port_of_loading",
    "shipper_address",
    "shipper_name",
    "vessel_name",
    "voyage_number",
    "lc_ref_number",
    "carrier_name",
    "lc_date",
    "shipped_onboard_date",
    "goods_description",
    "goods_quantity",
    "net_weight",
    "signed_by_master"
]

field_indices = {
    "transaction_date": 0,
    "bill_of_lading_number": 1,
    "bill_of_lading_issue_date": 2,
    "shipper_name": 3,
    "shipper_address": 4,
    "consignee_name": 5,
    "consignee_address": 6,
    "notify_party_name": 7,
    "notify_party_address": 8,
    "carrier_name": 9,
    "carrier_country": 10,
    "agent_name": 11,
    "agent_country": 12,
    "pre_carriage_by": 13,
    "vessel_name": 14,
    "voyage_number": 15,
    "container_number": 16,
    "place_of_receipt": 17,
    "port_of_loading": 18,
    "port_of_discharge": 19,
    "final_destination": 20,
    "shipped_onboard_date": 21,
    "bol_original_number": 22,
    "bol_original_or_copy": 23,
    "freight_collect_at": 24,
    "freight_collect_or_prepaid": 25,
    "place_of_issue": 26,
    "signed_by_carrier": 27,
    "signed_By_agent": 28,
    "lc_ref_number": 29,
    "lc_date": 30,
    "signature": 31,
    "stamp": 32,
    "net_weight": 33,
    "gross_weight": 34,
    "page_no": 35,
    "country_of_origin": 36,
    "goods_description": 37,
    "goods_quantity": 38,
    "dimension": 39,
    "goods_marks_and_nos": 40,
    "mode_of_transport": 41,
    "measurement": 42,
    "master_name": 43,
    "signed_by_master": 44,
    "total_original_no": 45,
    "stamped_vessel_or_cargo_name": 46,
    "charter": 47,
    "stamped_port_of_loading": 48,
    "stamped_onboard_date": 49,
    "carriage_condition": 50,
    "stamped_shipped_onboard_date": 51,
    "signed_by_master": 52,
    "shipment_date": 53
}

total_keys = 54

# Extract the indices
result_indices = {key: field_indices.get(key, None) for key in keys_to_find}
val_list = []
for key, value in result_indices.items():
    val_list.append(value)
# Print the result
print(result_indices)
print(val_list)
keys_not_to_consider = []
for i in range(0, total_keys):
    if i not in val_list:
        keys_not_to_consider.append(i)
print(keys_not_to_consider)
#[0, 7, 8, 11, 12, 13, 17, 24, 27, 28, 31, 32, 36, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]