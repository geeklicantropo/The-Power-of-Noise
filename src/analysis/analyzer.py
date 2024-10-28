import os
import json
from typing import Dict, List, Optional
import numpy as np
from collections import defaultdict
from tqdm import tqdm

class ResultsAnalyzer:
    """Analyzes experiment results and computes metrics"""
    
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        
    def load_experiment_results(self, experiment_type: str) -> Dict:
        """Load results for a specific experiment type"""
        pattern = {
            'classic': '*numdoc*_gold_at*_info_all_extended.json',
            'mixed': '*numdoc*_retr*_rand*_info_all_extended.json',
            'multi_corpus': '*numdoc*_main*_other*_info_all_extended.json'
        }
        
        results = {}
        for filename in tqdm(os.listdir(self.results_dir), desc=f"Loading {experiment_type} results"):
            if filename.endswith('_extended.json'):
                with open(os.path.join(self.results_dir, filename), 'r') as f:
                    results[filename] = json.load(f)
                    
        return results
    
    def compute_accuracy_metrics(
        self,
        results: Dict,
        group_by: Optional[str] = None
    ) -> Dict[str, float]:
        """Compute accuracy metrics, optionally grouped by a field"""
        if not group_by:
            correct = sum(1 for r in results if r['ans_match_after_norm'])
            total = len(results)
            return {'accuracy': correct/total if total > 0 else 0}
            
        grouped_results = defaultdict(list)
        for r in results:
            key = r.get(group_by)
            if key is not None:
                grouped_results[key].append(r['ans_match_after_norm'])
                
        metrics = {}
        for key, values in grouped_results.items():
            metrics[key] = sum(values)/len(values) if values else 0
            
        return metrics
    
    def analyze_position_impact(
        self,
        results: Dict,
        positions: List[int]
    ) -> Dict[int, float]:
        """Analyze impact of document position on accuracy"""
        position_metrics = {}
        
        for pos in tqdm(positions, desc="Analyzing positions"):
            filtered_results = [
                r for r in results 
                if r.get('gold_position') == pos
            ]
            
            metrics = self.compute_accuracy_metrics(filtered_results)
            position_metrics[pos] = metrics['accuracy']
            
        return position_metrics
    
    def analyze_document_counts(
        self,
        results: Dict,
        doc_type: str
    ) -> Dict[int, float]:
        """Analyze impact of number of documents on accuracy"""
        count_metrics = defaultdict(list)
        
        for r in tqdm(results, desc=f"Analyzing {doc_type} document counts"):
            if doc_type == 'distracting':
                count = len([
                    d for d in r.get('document_indices', [])
                    if d != r.get('gold_document_idx')
                ])
            elif doc_type == 'random':
                count = r.get('num_random_documents', 0)
            else:
                continue
                
            count_metrics[count].append(r['ans_match_after_norm'])
            
        return {
            count: sum(values)/len(values) 
            for count, values in count_metrics.items()
        }
    
    def compute_attention_stats(
        self,
        results: Dict
    ) -> Dict[str, float]:
        """Compute attention-related statistics"""
        stats = defaultdict(list)
        
        for r in tqdm(results, desc="Computing attention statistics"):
            if 'attention_scores' in r:
                scores = np.array(r['attention_scores'])
                stats['mean_attention'].append(scores.mean())
                stats['attention_entropy'].append(
                    -np.sum(scores * np.log(scores + 1e-10))
                )
                
        return {
            k: np.mean(v) for k, v in stats.items()
            if v
        }

    def get_retrieval_statistics(
        self,
        results: Dict
    ) -> Dict[str, float]:
        """Get statistics about retrieval performance"""
        stats = defaultdict(list)
        
        for r in tqdm(results, desc="Computing retrieval statistics"):
            if r.get('gold_document_idx') and r.get('document_indices'):
                try:
                    pos = r['document_indices'].index(r['gold_document_idx'])
                    stats['gold_position'].append(pos)
                except ValueError:
                    stats['gold_not_found'].append(1)
                
        if stats['gold_position']:
            stats['mean_gold_position'] = np.mean(stats['gold_position'])
            stats['median_gold_position'] = np.median(stats['gold_position'])
        
        total = len(results)
        stats['gold_retrieval_rate'] = 1 - (len(stats['gold_not_found']) / total) if total > 0 else 0
        
        return dict(stats)