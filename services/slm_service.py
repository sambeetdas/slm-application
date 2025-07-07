import os
import requests
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class SlmService:

    tokenizer = None
    model = None
    model_name = None
    max_history_length = None
    def __init__(self):
        self.model_name = os.environ.get("MODEL_NAME")
        self.max_history_length = 1000

    async def generate_response(self, request):
        user_input = request.question
        # Encode input
        inputs = self.tokenizer.encode(user_input, return_tensors="pt")
        
        max_length = self.max_history_length

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length= max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id= self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(user_input):].strip()
        


    async def load_model(self):
        """Load the model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("Model loaded successfully!")