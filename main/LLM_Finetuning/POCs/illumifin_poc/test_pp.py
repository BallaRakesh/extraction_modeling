from datetime import datetime
import time

start_time = datetime.now()
time.sleep(1) # simulate some processing time

predicted_class = "unable to predict"
confidence_score = 0.0
processing_time = round((datetime.now() - start_time).total_seconds(), 2)

print(predicted_class)
print(confidence_score)
print(processing_time)
print(type(predicted_class))
print(type(confidence_score))
print(type(processing_time))

exit('DONE')






def format_results(image_name, classification_results, extraction_results):
    formatted_output = {
        image_name: {
            "predicted_class": classification_results.get("predicted_class", "not found"),
            "confidence": f"{classification_results.get('confidence_score', 0.0):.8f}",
            "keys_extraction": {},
            "keys_bboxes": {},
            "keys_confidence": {}
        }
    }

    for key, value in extraction_results.items():
        if any(value):  # Check if the key contains non-empty values
            if key == "names" and isinstance(value, list) and all(isinstance(v, dict) for v in value):
                # Process names field separately, but do not keep 'names' as a key
                merged_names = {
                    "first_name": [],
                    "middle_name": [],
                    "last_name": []
                }

                for name_entry in value:
                    merged_names["first_name"].append(name_entry.get("first_name", ""))
                    merged_names["middle_name"].append(name_entry.get("middle_name", ""))
                    merged_names["last_name"].append(name_entry.get("last_name", ""))

                # Add merged names directly instead of under 'names' key
                for name_key, name_value in merged_names.items():
                    formatted_output[image_name]["keys_extraction"][name_key] = name_value
                    formatted_output[image_name]["keys_bboxes"][name_key] = []  # Maintain empty list for bbox
                    formatted_output[image_name]["keys_confidence"][name_key] = []  # Maintain empty list for confidence
            else:
                formatted_output[image_name]["keys_extraction"][key] = value
                formatted_output[image_name]["keys_bboxes"][key] = []  # Maintain empty list for bbox
                formatted_output[image_name]["keys_confidence"][key] = []  # Maintain empty list for confidence

    return formatted_output

# Example usage
classification_results = {
    'predicted_class': 'MDS',
    'confidence_score': 0.97,
    'processing_time_in_sec': 0.03
}

extraction_results = {
    'carrier': ['Transamerica'],
    'policy_number': [''],
    'names': [
        {'first_name': 'Samathan', 'middle_name': '', 'last_name': 'Wawiernia'},
        {'first_name': '', 'middle_name': 'Cheryl', 'last_name': 'Roepke'}
    ],
    'dob': [''],
    'social_security_number': [''],
    'subfolder_department': ['']
}

image_name = "doc_000042.png"
formatted_output = format_results(image_name, classification_results, extraction_results)
print(formatted_output)
