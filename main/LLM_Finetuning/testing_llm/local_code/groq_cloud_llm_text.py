import requests
import os

# Set your API key
os.environ["GROQ_API_KEY"] = "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"

# API endpoint
url = "https://api.groq.com/openai/v1/chat/completions"

# Headers
headers = {
    "Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
    "Content-Type": "application/json"
}

# Request body
data = {
    # "model": "mixtral-8x7b-32768",  # or another available model
    "model": "llama3-70b-8192",  # or another available model
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "write a python program to concatinate two input strings when its binary is divisible by two"}
    ]
}

# Make the API call
response = requests.post(url, json=data, headers=headers)

# Check the response
if response.status_code == 200:
    print(response.json()['choices'][0]['message']['content'])
else:
    print(f"Error: {response.status_code}")
    print(response.text)