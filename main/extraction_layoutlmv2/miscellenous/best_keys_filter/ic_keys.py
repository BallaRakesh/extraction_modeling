keys_to_find = [
    "address_of_assured",
    "agent_address",
    "end_date",
    "from_place",
    "insurance_issuer_address",
    "insurance_issuer_name",
    "issue_date",
    "mode_of_transport",
    "name_of_assured",
    "original_copy",
    "page_no",
    "policy_no",
    "start_date",
    "sum_insured_amount",
    "sum_insured_currency",
    "to_place",
    "vessel_or_flight_name",
    "vessel_or_flight_number",
    "agent_name",
    "lc_date",
    "lc_ref_no",
    "original_Number",
    "total_original"
]




field_indices = {
    "insurance_issuer_name": 0,
    "insurance_issuer_address": 1,
    "issue_date": 2,
    "expiry_date": 3,
    "place_of_expiry": 4,
    "name_of_assured": 5,
    "address_of_assured": 6,
    "mode_of_transport": 7,
    "vessel_or_flight_name": 8,
    "vessel_or_flight_number": 9,
    "from_place": 10,
    "to_place": 11,
    "sail_on_or_about_to_date": 12,
    "subject_matter_insured": 13,
    "premium_amount": 14,
    "country_of_origin_origin_of_goods": 15,
    "conditions_of_coverage": 16,
    "claim_payable_in": 17,
    "claim_payable_by_name": 18,
    "claim_payable_by_address": 19,
    "declaration_by": 20,
    "certificate_Stamped": 21,
    "original_copy": 22,
    "page_no": 23,
    "lc_ref_no": 24,
    "lc_date": 25,
    "original_Number": 26,
    "certificate_no": 27,
    "policy_no": 28,
    "start_date": 29,
    "end_date": 30,
    "sum_insured_amount": 31,
    "sum_insured_currency": 32,
    "signature": 33,
    "insurance_issuer_name_bottom": 34,
    "insurance_issuer_address_bottom": 35,
    "invoice_no": 36,
    "invoice_date": 37,
    "place_of_issue": 38,
    "agent_name": 39,
    "agent_address": 40,
    "premium_currency": 41,
    "signed_by_issuer": 42,
    "signed_by_underwriter": 43,
    "signed_by_agent": 44,
    "signed_by_proxy": 45,
    "total_original": 46,
    "original_number": 47,
    "insurance_coverage_amount": 48,
    "shipment_date": 49,
    "final_destination": 50,
    "goods_description": 51,
    "container_no": 52,
    "hscode": 53,
    "bl_no": 54,
    "bl_date": 55,
    "consignee_name": 56,
    "consignee_address": 57,
    "policy_effective_date": 58,
    "open_cover_no": 59,
    "marks_and_no": 60,
    "stamp": 61,
    "franchise_and_excess": 62,
    "voyage_no": 63,
    "mark": 64
}
total_keys = len(field_indices)

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