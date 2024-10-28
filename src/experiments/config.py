from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class ExperimentConfig:
    """Configuration for RAG experiments following the paper structure"""
    # Model settings
    llm_id: str = "meta-llama/Llama-2-7b-chat-hf"
    model_max_length: int = 4096
    device: str = "cuda"
    batch_size: int = 32
    
    # Experiment settings
    experiment_type: str = "classic"  # classic, mixed, or multi_corpus
    load_full_corpus: bool = True
    
    # Retriever settings
    use_random: bool = False
    use_adore: bool = False
    use_bm25: bool = False
    
    # Document settings
    gold_position: Optional[int] = None
    num_documents_in_context: int = 7
    get_documents_without_answer: bool = True
    
    # Mixed experiment settings
    num_retrieved_documents: Optional[int] = None
    num_random_documents: Optional[int] = None
    put_retrieved_first: bool = False
    
    # Multi-corpus settings
    use_corpus_nonsense: bool = False
    num_main_documents: Optional[int] = None
    num_other_documents: Optional[int] = None
    put_main_first: bool = False
    
    # Data paths
    use_test: bool = False
    corpus_path: str = "data/corpus.json"
    train_data_path: str = "data/10k_train_dataset.json"
    test_data_path: str = "data/test_dataset.json"
    contriever_results_path: str = "data/contriever_search_results_at150.pkl"
    bm25_results_path: str = "data/bm25_test_search_results_at250.pkl"
    adore_results_path: str = "data/adore_search_results_at200.pkl"
    random_results_path: str = "data/10k_random_results_at60.pkl"
    nonsense_results_path: str = "data/nonsense_random_results.pkl"
    reddit_results_path: str = "data/reddit_test_random_results.pkl"
    
    # Output settings
    output_dir: str = "experiment_results"
    save_every: int = 250

    @classmethod
    def from_json(cls, json_path: str) -> "ExperimentConfig":
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    def save(self, json_path: str):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)

    def validate(self):
        """Validate configuration settings"""
        if self.experiment_type not in ["classic", "mixed", "multi_corpus", "only_query"]:
            raise ValueError("Invalid experiment type")
            
        if self.experiment_type == "mixed":
            if self.num_retrieved_documents is None or self.num_random_documents is None:
                raise ValueError("Mixed experiments require retrieved and random document counts")
                
        if self.experiment_type == "multi_corpus":
            if self.num_main_documents is None or self.num_other_documents is None:
                raise ValueError("Multi-corpus experiments require main and other document counts")