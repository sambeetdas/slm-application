import os
import requests

class SlmService:

    #api_url: str = None
    #hf_token: str = None
    def __init__(self):
        self.api_url = os.environ.get("API_URL")
        self.hf_token = os.environ.get("HF_TOKEN")

    def process_question(self, request):
        headers = {
                     "Authorization": f"Bearer {self.hf_token}",
                  }       
        payload = {
                    "messages": [
                        {
                            "role": "user",
                            "content": request.question
                        }
                    ],
                    "model": "meta-llama/llama-3.1-8b-instruct",
                    "stream": False
                    }
              
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()
