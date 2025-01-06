from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("MBZUAI/MobiLlama-05B", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("MBZUAI/MobiLlama-05B", trust_remote_code=True)

model.to('cuda')
text = "I was walking towards the river when "
input_ids = tokenizer(text, return_tensors="pt").to('cuda').input_ids
outputs = model.generate(input_ids, max_length=1000, repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.batch_decode(outputs[:, input_ids.shape[1]:-1])[0].strip())
