import os
import json
import pickle
import random
import argparse
import numpy as np
import torch
from typing import Any, Dict, List, Optional, Tuple
import ijson

def seed_everything(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def str2bool(v):
    """Convert string to boolean"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_json(file_path: str) -> Any:
    """Read JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def write_json(data: Any, file_path: str):
    """Write to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def read_pickle(file_path: str) -> Any:
    """Read pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def write_pickle(data: Any, file_path: str):
    """Write to pickle file"""
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_corpus_subset(
    full_to_subset_path: str,
    subset_to_full_path: str,
    corpus_path: str
) -> Tuple[List[Dict], Dict[int, int]]:
    """Load corpus subset with mapping"""
    full_to_subset_idx_map = read_pickle(full_to_subset_path)
    subset_to_full_idx_map = read_pickle(subset_to_full_path)
    corpus = read_corpus_json(corpus_path, subset_to_full_idx_map)
    return corpus, full_to_subset_idx_map

def read_corpus_json(
    data_path: str,
    subset_to_full_idx_map: Optional[Dict[int, int]] = None
) -> List[Dict]:
    """Read corpus with optional index mapping"""
    corpus = []
    with open(data_path, "rb") as f:
        for idx, record in enumerate(ijson.items(f, "item")):
            if subset_to_full_idx_map:
                record['full_corpus_idx'] = subset_to_full_idx_map[idx]
            else:
                record['full_corpus_idx'] = idx
            corpus.append(record)
    return corpus

def load_specific_subset(subset_type: str) -> Tuple[List[Dict], Dict[int, int]]:
    """Load specific corpus subset based on type"""
    base_path = "data/processed"
    mappings_path = "data/mappings"
    
    paths = {
        "random": {
            "full_to_subset": f"{mappings_path}/full_to_subset_random_at60_in_corpus.pkl",
            "subset_to_full": f"{mappings_path}/subset_to_full_random_at60_in_corpus.pkl",
            "corpus": f"{base_path}/corpus_with_random_at60.json"
        },
        "contriever": {
            "full_to_subset": f"{mappings_path}/full_to_subset_contriever_at150_in_corpus.pkl",
            "subset_to_full": f"{mappings_path}/subset_to_full_contriever_at150_in_corpus.pkl",
            "corpus": f"{base_path}/corpus_with_contriever_at150.json"
        },
        "adore": {
            "full_to_subset": f"{mappings_path}/full_to_subset_adore_at200_in_corpus.pkl",
            "subset_to_full": f"{mappings_path}/subset_to_full_adore_at200_in_corpus.pkl",
            "corpus": f"{base_path}/corpus_with_adore_at200.json"
        },
        "random_contriever": {
            "full_to_subset": f"{mappings_path}/full_to_subset_random_contriever_in_corpus.pkl",
            "subset_to_full": f"{mappings_path}/subset_to_full_random_contriever_in_corpus.pkl",
            "corpus": f"{base_path}/corpus_with_random_contriever.json"
        },
        "test_random_bm25": {
            "full_to_subset": f"{mappings_path}/full_to_subset_test_random_bm25_in_corpus.pkl",
            "subset_to_full": f"{mappings_path}/subset_to_full_test_random_bm25_in_corpus.pkl",
            "corpus": f"{base_path}/test_corpus_with_random_bm25.json"
        },
        "test_random_contriever": {
            "full_to_subset": f"{mappings_path}/full_to_subset_test_random_contriever_in_corpus.pkl",
            "subset_to_full": f"{mappings_path}/subset_to_full_test_random_contriever_in_corpus.pkl",
            "corpus": f"{base_path}/test_corpus_with_random_contriever.json"
        }
    }
    
    if subset_type not in paths:
        raise ValueError(f"Unknown subset type: {subset_type}")
        
    paths_dict = paths[subset_type]
    return load_corpus_subset(
        paths_dict["full_to_subset"],
        paths_dict["subset_to_full"],
        paths_dict["corpus"]
    )