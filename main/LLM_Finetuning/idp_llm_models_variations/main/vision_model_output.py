from PIL import Image
import base64
import io
import json
import matplotlib.pyplot as plt
import torch
torch.cuda.empty_cache()
import os
from llm_guard.util import get_logger, lazy_load_dep
from llm_guard.output_scanners.base import Scanner
JSON_PATTERN = r"(?<!\\)(?:\\\\)*\{(?:[^{}]|(?R))*\}"


import re
import json
from typing import Optional, Dict, Any

def extract_json_from_string(input_string: str) -> Optional[str]:
    """
    Extracts a JSON string from the input string using regex.

    Args:
        input_string (str): The input string containing JSON.

    Returns:
        Optional[str]: The extracted JSON string, or None if no JSON is found.
    """
    # Regex pattern to match JSON (assumes JSON starts with { and ends with })
    pattern = r'\{.*\}'
    
    # Search for JSON in the input string
    match = re.search(pattern, input_string, re.DOTALL)  # re.DOTALL allows . to match newlines
    
    if match:
        return match.group(0)  # Return the matched JSON string
    return None  # Return None if no JSON is found


def process_string_for_json(input_string: str) -> Optional[Dict[str, Any]]:
    # Step 1: Extract JSON from the string
    json_string = extract_json_from_string(input_string)
    
    if not json_string:
        print("Extracted JSON is invalid.")
        return None


    json_repair = lazy_load_dep("json_repair")
    try:
        repaired_json = json_repair.repair_json(
            json_string, skip_json_loads=True, return_objects=False
        )
    except ValueError:
        return json_string

    return repaired_json