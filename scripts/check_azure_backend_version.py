import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

def check_backend_model():
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL")
    )
    
    deployment_name = "gpt-4o"
    print(f"Calling Azure Deployment: {deployment_name}")
    
    response = client.chat.completions.create(
        model=deployment_name,
        messages=[{"role": "user", "content": "Hello"}],
        temperature=0.1,
        max_tokens=5
    )
    
    # Azure OpenAI always returns the EXACT base model version in the response object
    exact_model_used = response.model
    print("-" * 50)
    print(f"Exact Base Model Used Under The Hood: {exact_model_used}")
    print("-" * 50)
    print("This is the exact string you can cite in your paper to prove transparency!")

if __name__ == "__main__":
    check_backend_model()
