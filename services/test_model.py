import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

def download_and_test_model():
    """Download and test a small language model locally"""
    
    # Choose a small model
    model_name = os.environ.get("MODEL_NAME")
    
    print(f"Downloading model: {model_name}")
    
    # Download tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Model downloaded successfully!")
    print(f"Model size: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test the model
    def generate_response(user_input, max_length=100):
        # Encode input
        inputs = tokenizer.encode(user_input, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(user_input):].strip()
    
    # Test conversation
    print("\n=== Testing Model ===")
    test_inputs = [
        "Hello, how are you?",
        "What's the weather like?",
        "Tell me a joke"
    ]
    
    for test_input in test_inputs:
        response = generate_response(test_input)
        print(f"Input: {test_input}")
        print(f"Response: {response}")
        print("-" * 40)
    
    return tokenizer, model

if __name__ == "__main__":
    tokenizer, model = download_and_test_model()