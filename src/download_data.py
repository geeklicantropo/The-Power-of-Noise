import os
from datasets import load_dataset
import json
import pickle
from tqdm import tqdm
import requests
from huggingface_hub import login
from load_env import load_environment, get_huggingface_token
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_credentials():
    """Setup HuggingFace credentials"""
    token = get_huggingface_token()
    if token:
        try:
            login(token)
            logger.info("Successfully logged in to HuggingFace")
            return True
        except Exception as e:
            logger.error(f"Error logging in to HuggingFace: {e}")
            return False
    logger.warning("No HuggingFace token found")
    return False

def download_huggingface_data():
    """Download datasets from HuggingFace"""
    try:
        # Setup credentials
        if not setup_credentials():
            logger.warning("Proceeding without HuggingFace authentication. Some downloads may fail.")
        
        logger.info("Downloading corpus from HuggingFace...")
        corpus = load_dataset('florin-hf/wiki_dump2018_nq_open')
        
        logger.info("Downloading NQ dataset from HuggingFace...")
        dataset = load_dataset('florin-hf/nq_open_gold')
        
        # Save the datasets
        os.makedirs('data/raw', exist_ok=True)
        
        logger.info("Processing and saving corpus...")
        # Convert dataset to list of dictionaries and save
        corpus_data = []
        for idx, item in enumerate(tqdm(corpus['train'], desc="Processing corpus")):
            corpus_data.append({
                'text': item['text'],
                'title': item['title'],
                'full_corpus_idx': idx
            })
        
        with open('data/raw/corpus.json', 'w') as f:
            json.dump(corpus_data, f)
            
        logger.info("Processing and saving NQ datasets...")
        # Convert train dataset
        train_data = []
        for item in tqdm(dataset['train'], desc="Processing train dataset"):
            train_data.append({
                'example_id': item['example_id'],
                'question': item['question'],
                'answers': item['answers'],
                'text': item['text'],
                'idx_gold_in_corpus': item['idx_gold_in_corpus']
            })
        
        # Convert test dataset
        test_data = []
        for item in tqdm(dataset['test'], desc="Processing test dataset"):
            test_data.append({
                'example_id': item['example_id'],
                'question': item['question'],
                'answers': item['answers'],
                'text': item['text'],
                'idx_gold_in_corpus': item['idx_gold_in_corpus']
            })
        
        with open('data/raw/nq_train.json', 'w') as f:
            json.dump(train_data, f)
        with open('data/raw/nq_test.json', 'w') as f:
            json.dump(test_data, f)
            
        logger.info("Successfully downloaded and processed all datasets!")
        
    except Exception as e:
        logger.error(f"Error in data download process: {e}")
        raise

def create_sample_results():
    """Create sample results for testing the analysis pipeline"""
    logger.info("Creating sample results for analysis...")
    
    # Create directory structure
    for exp_type in ['classic', 'mixed', 'multi_corpus']:
        os.makedirs(f'results/raw/{exp_type}', exist_ok=True)
    
    # Sample result for classic experiment
    classic_sample = {
        'example_id': '123',
        'query': 'who owned the millennium falcon before han solo',
        'documents': [
            {'text': 'Sample document 1', 'score': 0.9},
            {'text': 'Sample document 2', 'score': 0.8}
        ],
        'gold_position': 0,
        'generated_answer': 'Lando Calrissian',
        'ans_match_after_norm': True,
        'document_indices': [1, 2, 3, 4, 5],
        'prompt_tokens_len': 512,
        'attention_scores': [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]  # Sample attention scores
    }
    
    # Sample result for mixed/multi_corpus experiments
    mixed_sample = {
        'example_id': '123',
        'query': 'who owned the millennium falcon before han solo',
        'retrieved_docs': [
            {'text': 'Sample retrieved doc', 'score': 0.9}
        ],
        'random_docs': [
            {'text': 'Sample random doc', 'score': 0.5}
        ],
        'generated_answer': 'Lando Calrissian',
        'ans_match_after_norm': True,
        'document_indices': [1, 2, 3],
        'prompt_tokens_len': 512,
        'attention_scores': [[0.1, 0.2], [0.2, 0.3]]  # Sample attention scores
    }
    
    # Save sample results for each model and experiment type
    models = ['Llama2', 'MPT', 'Phi-2', 'Falcon']
    positions = ['far', 'mid', 'near']
    
    for model in tqdm(models, desc="Creating sample results"):
        # Classic experiment results
        for pos in positions:
            filename = f'results/raw/classic/{model}_gold_at0_{pos}_info_all_extended.json'
            with open(filename, 'w') as f:
                # Create multiple samples with varying properties
                samples = [
                    {**classic_sample, 
                     'gold_position': i,
                     'ans_match_after_norm': i % 2 == 0  # Alternate between True/False
                    } for i in range(5)
                ]
                json.dump(samples, f)
        
        # Mixed experiment results
        filename = f'results/raw/mixed/{model}_mixed_info_all_extended.json'
        with open(filename, 'w') as f:
            samples = [
                {**mixed_sample,
                 'num_retrieved_documents': i,
                 'num_random_documents': 3-i
                } for i in range(4)
            ]
            json.dump(samples, f)
            
        # Multi-corpus experiment results
        filename = f'results/raw/multi_corpus/{model}_multi_corpus_info_all_extended.json'
        with open(filename, 'w') as f:
            samples = [
                {**mixed_sample,
                 'num_main_documents': i,
                 'num_other_documents': 3-i
                } for i in range(4)
            ]
            json.dump(samples, f)

def main():
    logger.info("Starting data preparation...")
    
    # Load environment variables
    tokens = load_environment()
    
    # Create necessary directories
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('results/raw', exist_ok=True)
    
    # Download HuggingFace data
    try:
        download_huggingface_data()
    except Exception as e:
        logger.error(f"Error downloading HuggingFace data: {e}")
        logger.info("Continuing with sample results creation...")
    
    # Create sample results
    create_sample_results()
    
    logger.info("\nData preparation complete!")
    logger.info("You can now run the analysis pipeline.")

if __name__ == "__main__":
    main()