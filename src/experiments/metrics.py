from typing import List, Dict, Optional
from collections import defaultdict
from normalize_answers import normalize_answer, is_answer_in_text

class MetricsCalculator:
    """Calculate and analyze experiment metrics"""
    
    @staticmethod
    def calculate_accuracy(results: List[Dict]) -> Dict[str, float]:
        """Calculate accuracy metrics for the entire result set"""
        total = len(results)
        if total == 0:
            return {
                'total_accuracy': 0,
                'in_docs_accuracy': 0,
                'no_docs_accuracy': 0,
                'total_examples': 0,
                'correct_answers': 0,
                'docs_with_answer': 0,
                'docs_without_answer': 0
            }
            
        correct_answers = sum(1 for r in results if r['ans_match_after_norm'])
        
        # Split results based on answer presence in documents
        docs_with_answer = [r for r in results if r.get('ans_in_documents', False)]
        docs_without_answer = [r for r in results if not r.get('ans_in_documents', False)]
        
        # Calculate specific accuracies
        docs_accuracy = (
            sum(1 for r in docs_with_answer if r['ans_match_after_norm']) / 
            len(docs_with_answer) if docs_with_answer else 0
        )
        
        no_docs_accuracy = (
            sum(1 for r in docs_without_answer if r['ans_match_after_norm']) / 
            len(docs_without_answer) if docs_without_answer else 0
        )
        
        return {
            'total_accuracy': correct_answers / total,
            'in_docs_accuracy': docs_accuracy,
            'no_docs_accuracy': no_docs_accuracy,
            'total_examples': total,
            'correct_answers': correct_answers,
            'docs_with_answer': len(docs_with_answer),
            'docs_without_answer': len(docs_without_answer)
        }

    @staticmethod
    def calculate_metrics_by_position(
        results: List[Dict],
        position: int
    ) -> Dict[str, float]:
        """Calculate metrics for a specific gold document position"""
        position_results = [
            r for r in results 
            if r.get('document_indices') and 
            len(r['document_indices']) > position
        ]
        
        metrics = MetricsCalculator.calculate_accuracy(position_results)
        metrics['position'] = position
        return metrics

    @staticmethod
    def analyze_retrieval_performance(
        results: List[Dict],
        gold_position: Optional[int] = None
    ) -> Dict[str, Dict[str, float]]:
        """Analyze retrieval performance with detailed metrics"""
        # Basic metrics
        basic_metrics = MetricsCalculator.calculate_accuracy(results)
        
        # Position-specific metrics if gold position is provided
        position_metrics = {}
        if gold_position is not None:
            position_metrics = MetricsCalculator.calculate_metrics_by_position(
                results, 
                gold_position
            )
        
        # Analyze retrieved document positions
        retrieved_positions = defaultdict(int)
        for result in results:
            if result.get('gold_document_idx') and result.get('document_indices'):
                try:
                    pos = result['document_indices'].index(result['gold_document_idx'])
                    retrieved_positions[pos] += 1
                except ValueError:
                    retrieved_positions['not_found'] += 1
        
        # Calculate retrieval distribution
        total_queries = len(results)
        position_distribution = {
            str(pos): count/total_queries 
            for pos, count in retrieved_positions.items()
        }
        
        return {
            'basic_metrics': basic_metrics,
            'position_metrics': position_metrics,
            'retrieval_distribution': position_distribution
        }

    @staticmethod
    def calculate_experiment_summary(
        experiment_results: Dict[str, List[Dict]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate summary metrics for