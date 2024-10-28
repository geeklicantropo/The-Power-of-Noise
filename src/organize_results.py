import os
import shutil
from typing import List
import json
from tqdm import tqdm
import glob

def organize_results(
    source_dir: str,
    output_dir: str,
    models: List[str] = ['Llama2', 'MPT', 'Phi-2', 'Falcon']
):
    """Organize results into proper directory structure"""
    print("Organizing results...")
    
    # Create main experiment directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model directories and subdirectories
    for model in tqdm(models, desc="Creating model directories"):
        model_dir = os.path.join(output_dir, model)
        for exp_type in ['classic', 'mixed', 'multi_corpus']:
            os.makedirs(os.path.join(model_dir, exp_type), exist_ok=True)
    
    # Move files to appropriate directories
    for file in tqdm(glob.glob(os.path.join(source_dir, "*.json")), desc="Moving result files"):
        filename = os.path.basename(file)
        
        # Determine which model and experiment type
        for model in models:
            if model.lower() in filename.lower():
                if "gold_at" in filename:
                    dest_dir = os.path.join(output_dir, model, "classic")
                elif "retr" in filename and "rand" in filename:
                    dest_dir = os.path.join(output_dir, model, "mixed")
                else:
                    dest_dir = os.path.join(output_dir, model, "multi_corpus")
                    
                # Copy file to destination
                shutil.copy2(file, os.path.join(dest_dir, filename))
                break
    
    print("Results organized successfully!")
    return output_dir