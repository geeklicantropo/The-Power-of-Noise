import argparse
from experiments.config import ExperimentConfig
from experiments.experiment_manager import ExperimentManager
from utils import str2bool

def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG experiments")
    
    # Basic settings
    parser.add_argument("--output_dir", type=str, default="experiment_results")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--experiment_type", type=str, choices=["classic", "mixed", "multi_corpus", "only_query"])
    
    # Model settings
    parser.add_argument("--llm_id", type=str)
    parser.add_argument("--model_max_length", type=int)
    parser.add_argument("--device", type=str)
    parser.add_argument("--batch_size", type=int)
    
    # Data settings
    parser.add_argument("--load_full_corpus", type=str2bool)
    parser.add_argument("--use_test", type=str2bool)
    
    # Classic experiment settings
    parser.add_argument("--use_random", type=str2bool)
    parser.add_argument("--use_adore", type=str2bool)
    parser.add_argument("--gold_position", type=int)
    parser.add_argument("--num_documents_in_context", type=int)
    parser.add_argument("--get_documents_without_answer", type=str2bool)
    
    # Mixed experiment settings
    parser.add_argument("--use_bm25", type=str2bool)
    parser.add_argument("--num_retrieved_documents", type=int)
    parser.add_argument("--num_random_documents", type=int)
    parser.add_argument("--put_retrieved_first", type=str2bool)
    
    # Multi-corpus settings
    parser.add_argument("--use_corpus_nonsense", type=str2bool)
    parser.add_argument("--num_main_documents", type=int)
    parser.add_argument("--num_other_documents", type=int)
    parser.add_argument("--put_main_first", type=str2bool)
    
    # Output settings
    parser.add_argument("--save_every", type=int, default=250)
    
    args = parser.parse_args()
    return args

def create_config_from_args(args):
    """Create config from command line arguments"""
    config_dict = vars(args)
    
    # Remove None values
    config_dict = {k: v for k, v in config_dict.items() if v is not None}
    
    if args.config:
        # Load base config from file and update with command line args
        config = ExperimentConfig.from_json(args.config)
        for k, v in config_dict.items():
            setattr(config, k, v)
    else:
        # Create config directly from args
        config = ExperimentConfig(**config_dict)
    
    return config

def main():
    # Parse arguments
    args = parse_args()
    
    # Create config
    config = create_config_from_args(args)
    
    # Create experiment manager
    manager = ExperimentManager(config)
    
    # Run experiments
    print(f"Running {config.experiment_type} experiments...")
    results = manager.run_experiments()
    
    # Print summary
    total = len(results)
    correct = sum(1 for r in results if r['ans_match_after_norm'])
    accuracy = correct / total if total > 0 else 0
    
    print("\nExperiment Results:")
    print(f"Total examples: {total}")
    print(f"Correct answers: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()