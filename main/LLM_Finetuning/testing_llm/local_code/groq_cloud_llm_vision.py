import requests
import os
from groq import Groq

import os
import base64
import mimetypes

# Set your API key

os.environ["GROQ_API_KEY"] = "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"
client = Groq()

# API endpoint
url = "https://api.groq.com/openai/v1/chat/completions"

# Headers
headers = {
    "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
    "Content-Type": "application/json"
}


def image_to_data_url(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        mime_type = mimetypes.guess_type(image_path)[0]
        return f"data:{mime_type};base64,{encoded_string}"

# Usage
image_path = "/home/ntlpt19/TF_testing_EXT/dummy_responces/RE_ Regarding_ANB_Demo_Documents/output1/testing.jpg"
data_url = image_to_data_url(image_path)

prompt1 = """
"extract the information from the given image in the key and value pairs"
"""


prompt2 = """
what is the document class for the given image [Importer, Exporter]
"""

prompt3 = """
"summary of the given image
"""



# Request body
data = {
    "model": "llama-3.2-11b-vision-preview",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt1
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url
                    }
                },
            ]
        }
    ],
    "temperature": 0,
    "max_tokens": 2000,
    "top_p": 1,
    "stream": True,
    "stop": None

}

'''
                {
                    "type": "text",
                    "text": prompt2
                },
                {
                    "type": "text",
                    "text": prompt3
                }
'''
response = client.chat.completions.create(**data)

# Process the response
for chunk in response:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()

