import os
import json
import pickle
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from .config import ExperimentConfig
from llm import LLM
from normalize_answers import normalize_answer, is_answer_in_text
from utils import read_json, read_pickle

class ExperimentManager:
    """Manages different types of RAG experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.validate()
        
        # Initialize LLM
        self.llm = LLM(
            self.config.llm_id,
            self.config.device,
            quantization_bits=4,
            model_max_length=self.config.model_max_length
        )
        
        # Load data
        self._load_data()

    def _load_data(self):
        """Load necessary data based on experiment type"""
        # Load dataset
        data_path = self.config.test_data_path if self.config.use_test else self.config.train_data_path
        self.dataset = read_json(data_path)
        
        # Load corpus if needed
        if self.config.experiment_type != "only_query":
            if self.config.load_full_corpus:
                self.corpus = read_json(self.config.corpus_path)
            else:
                # Load appropriate subset based on experiment type
                if self.config.use_random:
                    self.corpus = read_pickle("data/processed/corpus_with_random_at60.json")
                elif self.config.use_adore:
                    self.corpus = read_pickle("data/processed/corpus_with_adore_at200.json")
                else:
                    self.corpus = read_pickle("data/processed/corpus_with_contriever_at150.json")

            # Load search results
            if self.config.experiment_type == "classic":
                if self.config.use_random:
                    self.search_results = read_pickle(self.config.random_results_path)
                elif self.config.use_adore:
                    self.search_results = read_pickle(self.config.adore_results_path)
                else:
                    self.search_results = read_pickle(self.config.contriever_results_path)
                    
            elif self.config.experiment_type == "mixed":
                self.retriever_results = read_pickle(
                    self.config.bm25_results_path if self.config.use_bm25 
                    else self.config.contriever_results_path
                )
                self.random_results = read_pickle(self.config.random_results_path)
                
            elif self.config.experiment_type == "multi_corpus":
                self.main_results = read_pickle(
                    self.config.bm25_results_path if self.config.use_bm25 
                    else self.config.contriever_results_path
                )
                self.other_results = read_pickle(
                    self.config.nonsense_results_path if self.config.use_corpus_nonsense
                    else self.config.reddit_results_path
                )

    def _build_prompt(self, query: str, documents: List[str] = None) -> str:
        """Build prompt based on experiment type"""
        task_instruction = "You are given a question and you MUST respond by EXTRACTING a short answer (max 5 tokens) from one of the provided documents. If none of the documents contain the answer, respond with NO-RES."
        
        if documents:
            documents_str = '\n'.join(documents)
            prompt = f"{task_instruction}\nDocuments:\n{documents_str}\nQuestion: {query}\nAnswer:"
        else:
            # Only query case
            prompt = f"You are given a question and you MUST respond with a short answer (max 5 tokens) based on your internal knowledge. If you do not know the answer, please respond with NO-RES.\nQuestion: {query}\nAnswer:"
            
        # Handle MPT model's special prompt format
        if 'mpt' in self.config.llm_id:
            INSTRUCTION_KEY = "### Instruction:"
            RESPONSE_KEY = "### Response:"
            INTRO_BLURB = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
            prompt = f"{INTRO_BLURB}\n{INSTRUCTION_KEY}\n{prompt}\n{RESPONSE_KEY}"
            
        return prompt

    def run_experiments(self) -> Dict[str, Any]:
        """Run experiments based on configuration"""
        results = []
        
        for example in tqdm(self.dataset):
            result = {
                'example_id': example['example_id'],
                'query': example['question'],
                'answers': example['answers']
            }
            
            # Build documents for context
            if self.config.experiment_type == "only_query":
                documents = None
            else:
                documents = self._get_documents(example)
                result['document_indices'] = [doc['full_corpus_idx'] for doc in documents]
                if 'idx_gold_in_corpus' in example:
                    result['gold_document_idx'] = example['idx_gold_in_corpus']
            
            # Build prompt and generate answer
            prompt = self._build_prompt(example['question'], documents)
            result['prompt'] = prompt
            
            response = self.llm.generate([prompt], max_new_tokens=15)[0]
            answer_start = response.find("Answer:") + len("Answer:") if "Answer:" in response else response.find("### Response:") + len("### Response:")
            generated_answer = response[answer_start:].strip()
            
            result['generated_answer'] = generated_answer
            result['ans_match_after_norm'] = any(
                normalize_answer(ans) in normalize_answer(generated_answer)
                for ans in example['answers']
            )
            
            if documents:
                result['ans_in_documents'] = any(
                    is_answer_in_text(doc['text'], example['answers'])
                    for doc in documents
                )
            
            results.append(result)
            
            # Save intermediate results
            if len(results) % self.config.save_every == 0:
                self._save_results(results, f"intermediate_{len(results)}")
        
        # Save final results
        self._save_results(results, "final")
        return results

    def _get_documents(self, example: Dict) -> List[Dict]:
        """Get documents based on experiment type and configuration"""
        if self.config.experiment_type == "classic":
            return self._get_classic_documents(example)
        elif self.config.experiment_type == "mixed":
            return self._get_mixed_documents(example)
        else:  # multi_corpus
            return self._get_multi_corpus_documents(example)

    def _get_classic_documents(self, example: Dict) -> List[Dict]:
        """Get documents for classic experiments"""
        idx = example['example_id']
        retrieved_indices = self.search_results[idx][0]
        documents = []
        
        # Handle gold document if specified
        if self.config.gold_position is not None:
            gold_idx = example['idx_gold_in_corpus']
            retrieved_indices = (
                retrieved_indices[:self.config.gold_position] + 
                [gold_idx] + 
                retrieved_indices[self.config.gold_position:]
            )
        
        for doc_idx in retrieved_indices[:self.config.num_documents_in_context]:
            doc = self.corpus[int(doc_idx)]
            if not self.config.get_documents_without_answer or not is_answer_in_text(doc['text'], example['answers']):
                documents.append(doc)
                
        return documents[:self.config.num_documents_in_context]

    def _get_mixed_documents(self, example: Dict) -> List[Dict]:
        """Get documents for mixed experiments"""
        idx = example['example_id']
        retrieved_indices = self.retriever_results[idx][0][:self.config.num_retrieved_documents]
        random_indices = self.random_results[idx][0][:self.config.num_random_documents]
        
        documents = []
        if self.config.put_retrieved_first:
            doc_indices = retrieved_indices + random_indices
        else:
            doc_indices = random_indices + retrieved_indices
            
        for doc_idx in doc_indices:
            documents.append(self.corpus[int(doc_idx)])
            
        return documents

    def _get_multi_corpus_documents(self, example: Dict) -> List[Dict]:
        """Get documents for multi-corpus experiments"""
        idx = example['example_id']
        main_indices = self.main_results[idx][0][:self.config.num_main_documents]
        other_indices = self.other_results[idx][0][:self.config.num_other_documents]
        
        documents = []
        if self.config.put_main_first:
            doc_indices = main_indices + other_indices
        else:
            doc_indices = other_indices + main_indices
            
        for doc_idx in doc_indices:
            documents.append(self.corpus[int(doc_idx)])
            
        return documents

    def _save_results(self, results: List[Dict], suffix: str):
        """Save experiment results"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Create appropriate filename based on experiment type and configuration
        if self.config.experiment_type == "classic":
            filename = f"numdoc{self.config.num_documents_in_context}_"
            if self.config.gold_position is not None:
                filename += f"gold_at{self.config.gold_position}_"
            if self.config.use_random:
                filename += "rand_"
            if self.config.get_documents_without_answer:
                filename += "answerless_"
        elif self.config.experiment_type == "mixed":
            filename = f"numdoc{self.config.num_documents_in_context}_"
            if self.config.put_retrieved_first:
                filename += f"retr{self.config.num_retrieved_documents}_rand{self.config.num_random_documents}_"
            else:
                filename += f"rand{self.config.num_random_documents}_retr{self.config.num_retrieved_documents}_"
        elif self.config.experiment_type == "multi_corpus":
            filename = f"numdoc{self.config.num_documents_in_context}_"
            if self.config.put_main_first:
                filename += f"main{self.config.num_main_documents}_other{self.config.num_other_documents}_"
            else:
                filename += f"other{self.config.num_other_documents}_main{self.config.num_main_documents}_"
            if self.config.use_corpus_nonsense:
                filename += "nonsense_"
            else:
                filename += "reddit_"
        else:  # only_query
            filename = "only_query_"
            
        filename += f"results_{suffix}.json"
        
        with open(os.path.join(self.config.output_dir, filename), 'w') as f:
            json.dump(results, f, indent=2)