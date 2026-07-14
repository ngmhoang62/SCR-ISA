import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# Make sure we load the project .env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

def test_models():
    models_to_test = ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"]
    
    print("Testing OpenAI / Azure OpenAI connections")
    print("-" * 50)
    
    for model in models_to_test:
        print(f"Testing model: '{model}' ... ", end="", flush=True)
        try:
            if model == "gpt-3.5-turbo":
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("[FAIL] FAILED: OPENAI_API_KEY not found in .env")
                    continue
                client = OpenAI(api_key=api_key)
            else:
                api_key = os.getenv("AZURE_OPENAI_API_KEY")
                base_url = os.getenv("AZURE_OPENAI_BASE_URL")
                if not api_key or not base_url:
                    print("[FAIL] FAILED: AZURE credentials not found in .env")
                    continue
                client = OpenAI(api_key=api_key, base_url=base_url)
                
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
            elif "401" in error_msg:
                print("[FAIL] FAILED: Unauthorized / Invalid API Key")
            else:
                print(f"[FAIL] FAILED: {error_msg}")

if __name__ == "__main__":
    test_models()
