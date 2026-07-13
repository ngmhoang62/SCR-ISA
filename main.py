import argparse
import logging
import os
from dotenv import load_dotenv

from src.utils import setup_logging, load_yaml
from src.prompts import PromptLoader
from src.llm import LLMClient
from src.methods import ChainOfThoughtMethod
from src.evaluation import Evaluator

def main():
    # Load environment variables from .env if present
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="LLM Prompting Design CLI")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default_config.yaml", 
        help="Path to config YAML file."
    )
    parser.add_argument(
        "--method", 
        type=str, 
        default="cot", 
        choices=["cot"], 
        help="Prompting method strategy."
    )
    parser.add_argument(
        "--context", 
        type=str, 
        default="Paris is the capital and most populous city of France.", 
        help="Context to populate prompt."
    )
    parser.add_argument(
        "--question", 
        type=str, 
        default="What is the capital of France?", 
        help="Question to query the LLM."
    )
    parser.add_argument(
        "--reference", 
        type=str, 
        default="Paris", 
        help="Expected ground truth answer."
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_yaml(args.config)
    except FileNotFoundError:
        print(f"Configuration file not found at {args.config}. Using default structures.")
        config = {}
    
    # Setup Logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    setup_logging(level=log_level)
    logger = logging.getLogger("main")
    
    logger.info("Initializing prompting pipeline...")
    
    # Extract config sections
    llm_cfg = config.get("llm", {})
    prompt_cfg = config.get("prompts", {})
    
    # Configure prompt loader
    loader = PromptLoader(
        system_prompt_path=prompt_cfg.get("system_prompt_path", "prompts/system_prompts/default.txt"),
        user_template_path=prompt_cfg.get("user_template_path", "prompts/templates/cot_template.txt")
    )
    
    # LLM Settings
    provider = llm_cfg.get("provider", "gemini")
    model = llm_cfg.get("model", "gemini-2.5-flash")
    temperature = llm_cfg.get("temperature", 0.2)
    max_tokens = llm_cfg.get("max_tokens", 1024)
    
    logger.info(f"Configured LLM: Provider={provider}, Model={model}, Temp={temperature}")
    
    # Check if API Keys are configured
    required_key = "GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY"
    if not os.getenv(required_key):
        logger.warning(
            f"'{required_key}' environment variable is missing. "
            "API invocation will fail. Please copy '.env.example' to '.env' and insert your API key."
        )
        
    try:
        client = LLMClient(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    except Exception as e:
        logger.error(f"Failed to initialize LLM client: {e}")
        return

    # Instantiate the selected method
    if args.method == "cot":
        method = ChainOfThoughtMethod(client, loader)
    else:
        logger.error(f"Unsupported prompting method strategy: '{args.method}'")
        return

    # Run Prompt inference
    variables = {
        "context": args.context,
        "question": args.question
    }
    
    logger.info(f"Executing prompt pipeline '{args.method}'...")
    try:
        response = method.run(variables)
        print("\n=== Model Output ===")
        print(response)
        print("====================\n")
        
        # Run Evaluation
        if args.reference:
            logger.info("Evaluating output accuracy...")
            eval_result = Evaluator.exact_match(
                prediction=response,
                reference=args.reference,
                use_boxed_extraction=True
            )
            print("=== Evaluation Results ===")
            print(f"Extracted Answer: '{eval_result['extracted_prediction']}'")
            print(f"Ground Truth:     '{eval_result['ground_truth']}'")
            print(f"Match status:     {eval_result['exact_match']}")
            print("==========================\n")
            
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")

if __name__ == "__main__":
    main()
