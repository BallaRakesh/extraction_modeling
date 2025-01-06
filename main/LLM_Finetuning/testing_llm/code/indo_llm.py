from transformers import AutoModelForCausalLM, AutoTokenizer
import time
# from time import time

device = "cpu"

model = AutoModelForCausalLM.from_pretrained("robinsyihab/Sidrap-7B-v1")
tokenizer = AutoTokenizer.from_pretrained("robinsyihab/Sidrap-7B-v1")

messages = [
    {"role": "user", "content": "buatkan kode program, sebuah fungsi untuk memvalidasi alamat email menggunakan regex"}
]
messages = [
    {"role": "user", "content": "buatkan kode program, sebuah fungsi untuk memvalidasi alamat email menggunakan regex"}
]
encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

model_inputs = encodeds.to(device)
model.to(device)
start_time = time.time()

generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
        
end_time = time.time()
elapsed_time = end_time - start_time
print('elapsed_time: ', elapsed_time)
print(decoded[0])