import os
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_directories():
    """Check if all required directories exist"""
    required_dirs = [
        'results/raw',
        'results/organized',
        'results/analysis',
        'data/raw',
    ]
    
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Checked directory: {dir_path}")

def check_files():
    """Check if required files exist"""
    required_files = [
        '.env',
        'src/analysis/paper_results.json',
        'scripts/run_complete_analysis.sh'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            logger.error(f"Missing required file: {file_path}")
    
    return len(missing_files) == 0

def check_raw_results():
    """Check if we have raw results to analyze"""
    experiments = ['classic', 'mixed', 'multi_corpus']
    models = ['Llama2', 'MPT', 'Phi-2', 'Falcon']
    
    for exp in experiments:
        path = f'results/raw/{exp}'
        if not os.path.exists(path) or not os.listdir(path):
            logger.warning(f"No results found for {exp} experiment")
            return False
    
    logger.info("Raw results found for analysis")
    return True

def main():
    logger.info("Starting setup check...")
    
    # Check directories
    check_directories()
    
    # Check files
    if not check_files():
        logger.error("Missing required files!")
        sys.exit(1)
    
    # Check raw results
    if not check_raw_results():
        logger.warning("Some raw results might be missing. Analysis might be incomplete.")
    
    logger.info("Setup check complete - ready to run analysis!")

if __name__ == "__main__":
    main()