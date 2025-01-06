from dotenv import load_dotenv
from llama_cpp import Llama, LlamaDiskCache
import time

# Load environment variables
load_dotenv()

# Initialize cache
cache = LlamaDiskCache(cache_dir="dex_llama_cache")

# Model path
model_path2 = '/home/ntlpt19/LLM_training/quantize_models/gemma-2-2b-it-Q4_K_S.gguf'
model_path2 = '/home/ntlpt19/LLM_training/quantize_models/gemma-2-2b-it-Q3_K_L.gguf'

# Initialize model
model = Llama(
    model_path=model_path2,
    verbose=True,
    n_threads=None,
    n_ctx=500,
    cache=cache,
    seed=50,
    chat_format="gemma"  # Specify the chat format as "gemma"
)

def generate_response(prompt, temperature=0.0, top_k=10, top_p=0.7):
    start_time = time.time()
    
    # For Gemma, we'll use a simpler message format
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    try:
        output = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
        
        response = output['choices'][0]['message']['content']
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return {
            'response': response,
            'time_taken': elapsed_time
        }
    
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    prompt1_1 = "What is machine learning?"  # Replace with your actual prompt
    
    result = generate_response(prompt1_1)
    
    if result:
        print('Prediction:', result['response'])
        print('Time taken:', result['time_taken'])
    
    model.reset()