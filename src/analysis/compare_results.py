import argparse
import json
import os
from typing import Dict, List
import matplotlib.pyplot as plt
from visualization import ResultsVisualizer
from analyzer import ResultsAnalyzer

def parse_args():
    parser = argparse.ArgumentParser(description="Compare experiment results with paper")
    parser.add_argument('--results_dir', type=str, required=True,
                      help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='analysis_results',
                      help='Directory to save analysis results')
    parser.add_argument('--paper_results', type=str, required=True,
                      help='Path to JSON file containing paper results')
    return parser.parse_args()

def load_paper_results(path: str) -> Dict:
    """Load paper results from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analysis tools
    analyzer = ResultsAnalyzer(args.results_dir)
    visualizer = ResultsVisualizer(args.output_dir)
    
    # Load results
    paper_results = load_paper_results(args.paper_results)
    
    # Analyze different experiment types
    experiment_types = ['classic', 'mixed', 'multi_corpus']
    for exp_type in experiment_types:
        results = analyzer.load_experiment_results(exp_type)
        
        if exp_type == 'classic':
            # Analyze distracting documents impact
            distracting_metrics = analyzer.analyze_document_counts(results, 'distracting')
            visualizer.plot_distracting_impact(
                distracting_metrics,
                list(paper_results['distracting'].keys()),
                os.path.join(args.output_dir, f'{exp_type}_distracting_impact.png')
            )
            
            # Analyze position impact
            position_metrics = analyzer.analyze_position_impact(results, [0, 6, 12])
            visualizer.plot_position_comparison(
                position_metrics,
                list(paper_results['position'].keys()),
                os.path.join(args.output_dir, f'{exp_type}_position_impact.png')
            )
            
        elif exp_type == 'mixed':
            # Analyze random documents impact
            random_metrics = analyzer.analyze_document_counts(results, 'random')
            visualizer.plot_random_impact(
                random_metrics,
                list(paper_results['random'].keys()),
                os.path.join(args.output_dir, f'{exp_type}_random_impact.png')
            )
            
        # Create comparison tables
        our_results = analyzer.compute_accuracy_metrics(results)
        comparison_df = visualizer.create_comparison_table(
            paper_results[exp_type],
            our_results,
            os.path.join(args.output_dir, f'{exp_type}_comparison.csv')
        )
        print(f"\nComparison for {exp_type}:")
        print(comparison_df)
        
        # If attention scores are available, create attention heatmap
        for result in results:
            if 'attention_scores' in result:
                visualizer.plot_attention_heatmap(
                    result['attention_scores'],
                    [f"Doc {i}" for i in range(len(result['document_indices']))],
                    os.path.join(args.output_dir, f'{exp_type}_attention_heatmap.png')
                )
                break

if __name__ == "__main__":
    main()