from transformers import pipeline, set_seed

summarizer = pipeline('text2text-generation', model='describeai/gemini-small')
code = "print('hello world!')"

response = summarizer(code, max_length=100, num_beams=3)
print("Summarized code: " + response[0]['generated_text'])
