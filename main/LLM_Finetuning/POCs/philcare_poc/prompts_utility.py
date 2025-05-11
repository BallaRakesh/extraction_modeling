


system_prompt = """
You are an AI that extracts key-value pairs from text.

Note:
- If the field is not present or illegible, return "Not Found".
- Preserve the exact format and capitalization of the text within the values.
"""

prompt_text = """
        You are an advanced language model with image analysis capabilities. I will provide you with an image containing text, and your task is to extract specific fields from it. The fields may appear as single values or within a tabular format. Additionally, if the image contains an ID card, extract its information into a separate 'id_card_info' dictionary with possible keys and their corresponding values. Follow these rules:

            1. **Single Values**: If a field is present as a standalone key-value pair (not in a table), extract its value(s). If a single value is found, return it as a string. If multiple values are found for the same key, return them as a list of strings (e.g., ["value1", "value2"]). If the key or its value is not found, return ["not found"]. However, for fields related to tables (e.g., 'service_code' and 'service_description' for 'service_details', 'doctors_name' and 'patient_name' for 'doctor_patient_details'), these should be set to ["not found"] if the corresponding table is present, and the table should be omitted if the single values are present.
            2. **Tabular Format**: Only extract tables named 'doctor_patient_details' or 'service_details'. Ignore any other tables in the image. If any of the specified fields (or a combination of them) are found in these tables, extract the entire table. The table should be represented as a list of rows, where each row is a dictionary with column names in lowercase and separated by underscores (e.g., 'service_code') as keys and their corresponding values. Include all columns in the table, even if they are not part of the specified fields. If a 'service_details' table is present, set the single-value fields 'service_code' and 'service_description' to ["not found"]. If a 'doctor_patient_details' table is present, set the single-value fields 'doctors_name' and 'patient_name' to ["not found"]. Conversely, if the single-value fields are present, set the corresponding table to an empty list.
            3. **ID Card Information**: If the image contains an ID card, extract all identifiable fields into an 'id_card_info' dictionary. Possible keys may include (but are not limited to): 'full_name', 'nationality', 'sex', 'date_of_birth', 'height', 'address', 'license_number', 'issue_date', 'expiration_date', 'signature', 'conditions', 'authority', or any other relevant fields present on the ID card (use lowercase keys with underscores). If a single value is found for a key, return it as a string. If multiple values are found for the same key, return them as a list of strings (e.g., ["value1", "value2"]). If a key is not found, omit it from the dictionary (do not include "not found" for missing keys in this section).
            4. **Top Date**: Check the top right and top left areas of the image for any date. If a single date is found, include it under the key 'date' as a string. If multiple dates are found, include them as a list of strings (e.g., ["date1", "date2"]). If no date is found in those areas, set its value to ["not found"].
            5. **Output Format**: Return the result as a JSON object. For single-value fields, use the field name with underscores (e.g., 'certificate_number') as the key and the value(s) as a string or list of strings. For tabular data, use the table names 'doctor_patient_details' and 'service_details' as keys and provide the list of row dictionaries as their values (with lowercase keys separated by underscores). If an ID card is detected, include the 'id_card_info' dictionary with the extracted fields (lowercase keys with underscores). Ensure that 'service_details' and its single-value fields ('service_code', 'service_description') are mutually exclusive, and 'doctor_patient_details' and its single-value fields ('doctors_name', 'patient_name') are mutually exclusive.

        Here are the fields to extract:  
            1. certificate_number  
            2. approval_number  
            3. availment_date  
            4. diagnosis  
            5. doctors_name  
            6. patient_name  
            7. amount  
            8. service_code  
            9. service_description  
            10. date  

        Tips to follow for extracting:  
            1. **certificate_number**: Look for the exact phrases "Certificate number" or "Certificate No" in the document. Ensure that the extracted value is strictly an alphanumeric code associated with this key. Do not include any other numbers, unrelated codes, or extra text. If no valid certificate number is found, return ["not found"].
            2. **approval_number**: Locate the terms "Approval number" or "Approval code" in the document. Extract only the exact alphanumeric code related to approval and do not include any other numbers or unrelated values. If no valid approval number is found, return ["not found"].
            3. **availment_date**: Identify the text "Availment date" and extract its corresponding value(s) as a string or list. If missing, set it to ["not found"]. do not include other dates in to this fiels, we get strightly on this name
            4. **date**: Check the top right and top left areas of the image for any date. If a date is present in these locations, extract it as a string or list of strings. If no date is found in those areas, return ["not found"].
            5. **patient_name**: Identify the text "Patient Name" and extract its corresponding value(s) as a string or list. Additionally, this field may appear as "Patient of Name" or "Patient Name & TIN". Extract the exact value found next to any of these variations. Ensure that the **complete name** is extracted as mentioned in the image **without trimming or modifying** it. Do not extract unrelated names. If missing, set it to ["not found"].  
            6. **doctor_name**: Identify the text "Doctor's Name" and extract its corresponding value(s) as a string or list. This field may also appear as "Attending Physician". Extract the exact value found next to either of these labels. Ensure that the **complete name** is extracted as mentioned in the image **without trimming or modifying** it. If missing, set it to ["not found"].  
            7. **service_description**: Extract the relevant service-related details from the document. Do not include any diagnosis information—ensure that the extracted content strictly describes the provided service and excludes any medical conditions, symptoms, or diagnoses. If no valid service description is found, return ["not found"].

        Steps to follow:  
            1. Analyze the text content of the image.  
            2. Determine if each field is a standalone value or part of a table named 'doctor_patient_details' or 'service_details'. Ignore any other tables.  
            3. For standalone fields, extract the value(s) as a string or list, or assign ["not found"] if missing. Set 'service_code' and 'service_description' to ["not found"] if 'service_details' table is present, and set 'service_details' to [] if 'service_code' or 'service_description' are present. Set 'doctors_name' and 'patient_name' to ["not found"] if 'doctor_patient_details' table is present, and set 'doctor_patient_details' to [] if 'doctors_name' or 'patient_name' are present.  
            4. For fields in the specified tables, extract the complete table as a list of dictionaries, where each dictionary represents a row with column names in lowercase and separated by underscores and their values.  
            5. If an ID card is detected, extract all identifiable fields into an 'id_card_info' dictionary with relevant keys (lowercase with underscores) and value(s) as a string or list.  
            6. Check the top right and top left of the image for any date and extract it under 'date' as a string or list, or set it to ["not found"] if absent.  
            7. Return the result as a JSON object with keys using underscores.
            8. If a value appears multiple times, extract it multiple times.
            
        Identifying the Tables:
            The 'doctor_patient_details' table can be identified by the presence of a table with the details of doctor and patient. some times , the table might be extened, please make sure to extract the data from the table.
            The 'service_details' table can be identified by detecting columns related to 'services', 'item code', 'particulars', 'DOC#' and some department field. If these columns exist, it indicates a service-related table.

        Example output with mixed single values, a service_details table, and an ID card with multiple values:  
        ```json
        {
        "certificate_number": ["ABC12345"],
        "approval_number": ["not found"],
        "availment_date": ["2023-10-15"],
        "diagnosis": ["not found"],
        "doctors_name": ["Dr. John Smith"],
        "patient_name": ["Jane Doe"],
        "amount": ["500.00"],
        "date": ["2025-03-06"],
        "id_card_info": {
            "full_name": ["Malagapo, Patrick Steven Pornillos"],
            "nationality": ["Filipino"],
            "sex": ["M"],
            "date_of_birth": ["1991/12/11"],
            "height": ["1.77"],
            "address": ["Zaballero, Pasig City Quezon"],
            "license_number": ["N07-98-5679"],
            "issue_date": ["02/03/2019"],
            "expiration_date": ["02/03/2024", "02/03/2025"]
        },
        "doctor_patient_details": [],
        "service_details": [
            {"service_code": "s123", "service_description": "consultation", "duration": "30 mins"},
            {"service_code": "s124", "service_description": "x-ray", "duration": "15 mins"}
        ]
        }
        ```

        Example output with only single values and no ID card:  
        ```json
        {
        "certificate_number": ["ABC12345"],
        "approval_number": ["not found"],
        "availment_date": ["2023-10-15"],
        "diagnosis": ["not found"],
        "doctors_name": ["Dr. John Smith"],
        "patient_name": ["Jane Doe"],
        "amount": ["500.00"],
        "service_code": ["S123"],
        "service_description": ["not found"],
        "date": ["not found"],
        "id_card_info": {},
        "doctor_patient_details": [],
        "service_details": []
        }
        ```

        Example output with only an ID card and a top date with multiple values:  
        ```json
        {
        "certificate_number": ["not found"],
        "approval_number": ["not found"],
        "availment_date": ["not found"],
        "diagnosis": ["not found"],
        "doctors_name": ["not found"],
        "patient_name": ["not found"],
        "amount": ["not found"],
        "service_code": ["not found"],
        "service_description": ["not found"],
        "date": ["2025-03-06", "2024-03-06"],
        "id_card_info": {
            "full_name": ["Malagapo, Patrick Steven Pornillos"],
            "nationality": ["Filipino"],
            "sex": ["M"],
            "date_of_birth": ["1991/12/11"],
            "height": ["1.77"],
            "address": ["Zaballero, Pasig City Quezon"],
            "license_number": ["N07-98-5679"],
            "issue_date": ["02/03/2019"],
            "expiration_date": ["02/03/2024"]
        },
        "doctor_patient_details": [],
        "service_details": []
        }
        ```

        Now, please process the image I will provide and extract the fields accordingly, following the rules above.

    """
    
##### Setup Azure OpenAI in Python #####
# Use the API Key and Endpoint to configure the OpenAI client.
