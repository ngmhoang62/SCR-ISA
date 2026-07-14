import os
import requests
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file in the root acm2026/ directory
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    load_dotenv(env_path)
    
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    
    if not api_key or api_key == "your_api_key_here":
        print("[ERROR] Please provide a valid OPENAI_API_KEY in the .env file")
        return
        
    if not base_url or "YOUR-RESOURCE-NAME" in base_url:
        print("[ERROR] Please provide a valid OPENAI_BASE_URL in the .env file")
        return
        
    # v1 endpoint to list models
    url = f"{base_url.rstrip('/')}/models"
    
    headers = {
        "api-key": api_key, # Azure OpenAI specific header
        "Authorization": f"Bearer {api_key}" # Fallback for some systems parsing Bearer token
    }
    
    output_file = os.path.join(os.path.dirname(__file__), "azure_models_list.txt")
    
    print(f"[INFO] Fetching data from: {url}...")
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            models_data = response.json()
            
            # Write results to txt file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("=== AZURE OPENAI DEPLOYMENT MODELS ===\n")
                f.write("-" * 50 + "\n")
                count = 0
                for model in models_data.get('data', []):
                    model_id = model.get('id', 'Unknown')
                    f.write(f"{model_id}\n")
                    print(f"[INFO] Found model: {model_id}")
                    count += 1
                f.write("-" * 50 + "\n")
                f.write(f"Total: {count} models.\n")
                
            print(f"\n[INFO] Successfully saved list to: {output_file}")
        else:
            print(f"[ERROR] Connection failed: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"[ERROR] An error occurred during API call: {e}")

if __name__ == "__main__":
    main()
