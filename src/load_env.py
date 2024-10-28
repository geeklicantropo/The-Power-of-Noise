import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Get the main HF token
    hf_token = os.getenv('HF_TOKEN')
    
    # Use the same token for all models
    tokens = {
        'HF_TOKEN': hf_token,
        'LLAMA2_TOKEN': hf_token,  # Using same token for Llama 2
        'FALCON_TOKEN': hf_token,  # Using same token for Falcon
        'MPT_TOKEN': hf_token,     # Using same token for MPT
    }
    
    # Log token configuration
    if hf_token:
        logger.info("Using HuggingFace token 'rag_improvement' for all models")
    else:
        logger.warning("No HuggingFace token found!")
    
    return tokens

def get_huggingface_token():
    """Get HuggingFace token from environment"""
    token = os.getenv('HF_TOKEN')
    if token:
        return token
    else:
        logger.warning("HuggingFace token not found. Please set HF_TOKEN in .env file")
        return None

def verify_model_access():
    """Verify access to all required models"""
    from transformers import AutoTokenizer
    
    models = {
        'Llama 2': 'meta-llama/Llama-2-7b-chat-hf',
        'Falcon': 'tiiuae/falcon-7b-instruct',
        'MPT': 'mosaicml/mpt-7b-instruct'
    }
    
    results = {}
    for model_name, model_id in models.items():
        try:
            AutoTokenizer.from_pretrained(model_id)
            results[model_name] = "Access OK"
        except Exception as e:
            results[model_name] = f"Access Failed: {str(e)}"
    
    return results

if __name__ == "__main__":
    # Test the environment loading and model access
    tokens = load_environment()
    print("\nToken Configuration:")
    print("Using HuggingFace token 'rag_improvement' for all models")
    
    print("\nVerifying model access...")
    access_results = verify_model_access()
    for model, status in access_results.items():
        print(f"{model}: {status}")