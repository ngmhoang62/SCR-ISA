import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Make sure we load the project .env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

def test_models():
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key or not base_url:
        print("Error: OPENAI_API_KEY and OPENAI_BASE_URL must be set in .env")
        return
        
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    
    models_to_test = ["gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18", "gpt-35-turbo-0125"]
    
    print(f"Testing Azure OpenAI connection at {base_url}")
    print("-" * 50)
    
    for model in models_to_test:
        print(f"Testing model deployment: '{model}' ... ", end="", flush=True)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say 'hello' briefly."}],
                temperature=0.1,
                max_tokens=5
            )
            result = response.choices[0].message.content.strip()
            print(f"[OK] SUCCESS! (Response: {result})")
        except Exception as e:
            error_msg = str(e)
            if "404" in error_msg or "DeploymentNotFound" in error_msg:
                print("[FAIL] FAILED: Deployment Not Found (404)")
            else:
                print(f"[FAIL] FAILED: {error_msg}")

if __name__ == "__main__":
    test_models()
