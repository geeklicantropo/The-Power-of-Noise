import os
import argparse
from typing import Dict, List
import json
from tqdm import tqdm
import logging
from src.analysis.analyzer import ResultsAnalyzer
from src.analysis.visualization import ResultsVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_classic_experiments(
    analyzer: ResultsAnalyzer,
    visualizer: ResultsVisualizer,
    paper_results: Dict,
    output_dir: str,
    models: List[str]
):
    """Analyze classic experiments with distracting and random documents"""
    logger.info("\nAnalyzing classic experiments...")
    
    # Load results for each model
    results = {}
    for model in tqdm(models, desc="Loading model results"):
        model_dir = os.path.join(analyzer.results_dir, model)
        results[model] = analyzer.load_experiment_results('classic')
        if not results[model]:
            logger.warning(f"No results found for {model} in classic experiments")
    
    if not any(results.values()):
        logger.error("No classic experiment results found to analyze!")
        return

    # Generate comparisons
    for setting in tqdm(['distracting', 'random'], desc="Analyzing settings"):
        for position in tqdm(['far', 'mid', 'near'], desc=f"Processing {setting} positions"):
            try:
                # Plot accuracy vs number of documents
                visualizer.plot_position_comparison(
                    {model: results[model] for model in models if results[model]},
                    models,
                    save_path=os.path.join(output_dir, f'{setting}_{position}_comparison.png')
                )
                
                # Create comparison table
                comparison_df = visualizer.create_comparison_table(
                    paper_results[setting],
                    {model: analyzer.compute_accuracy_metrics(results[model]) 
                     for model in models if results[model]},
                    save_path=os.path.join(output_dir, f'{setting}_{position}_comparison.csv')
                )
                
                logger.info(f"Generated analysis for {setting} - {position}")
            except Exception as e:
                logger.error(f"Error analyzing {setting} {position}: {str(e)}")

def analyze_mixed_experiments(
    analyzer: ResultsAnalyzer,
    visualizer: ResultsVisualizer,
    paper_results: Dict,
    output_dir: str
):
    """Analyze mixed retrieval experiments"""
    logger.info("\nAnalyzing mixed retrieval experiments...")
    contriever_results = analyzer.load_experiment_results('mixed')
    
    if not contriever_results:
        logger.error("No mixed experiment results found to analyze!")
        return

    try:
        contriever_metrics = analyzer.analyze_document_counts(contriever_results, 'mixed')
        
        # Plot comparisons
        visualizer.plot_random_impact(
            contriever_metrics,
            paper_results['mixed_retrieval']['Contriever'],
            save_path=os.path.join(output_dir, 'contriever_mixed_comparison.png')
        )
        
        # Create comparison table
        comparison_df = visualizer.create_comparison_table(
            paper_results['mixed_retrieval'],
            {'Contriever': analyzer.compute_accuracy_metrics(contriever_results)},
            save_path=os.path.join(output_dir, 'mixed_retrieval_comparison.csv')
        )
        
        logger.info("Generated mixed experiment analysis")
    except Exception as e:
        logger.error(f"Error in mixed experiment analysis: {str(e)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run complete analysis pipeline")
    parser.add_argument('--raw_results_dir', type=str, required=True,
                      help='Directory containing raw experiment result files')
    parser.add_argument('--organized_results_dir', type=str, default='results/organized',
                      help='Directory to store organized results')
    parser.add_argument('--analysis_dir', type=str, default='results/analysis',
                      help='Directory to save analysis results')
    parser.add_argument('--models', nargs='+', 
                      default=['Llama2', 'MPT', 'Phi-2', 'Falcon'],
                      help='List of models to analyze')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.analysis_dir, exist_ok=True)
    
    # Initialize analysis components
    analyzer = ResultsAnalyzer(args.raw_results_dir)
    visualizer = ResultsVisualizer(args.analysis_dir)
    
    # Load paper results
    paper_results_path = os.path.join(os.path.dirname(__file__), 'analysis', 'paper_results.json')
    logger.info(f"Loading paper results from {paper_results_path}")
    try:
        with open(paper_results_path, 'r') as f:
            paper_results = json.load(f)
    except Exception as e:
        logger.error(f"Error loading paper results: {str(e)}")
        return

    # Run analyses
    analyze_classic_experiments(
        analyzer, visualizer, paper_results, 
        args.analysis_dir, args.models
    )
    
    analyze_mixed_experiments(
        analyzer, visualizer, paper_results, 
        args.analysis_dir
    )
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main()