import torch
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, StoppingCriteriaList, StoppingCriteria
)
from typing import List, Tuple, Optional

class StopOnTokens(StoppingCriteria):
    """Custom stopping criteria for text generation"""
    def __init__(self, tokenizer, stop_strings: List[str]):
        self.tokenizer = tokenizer
        self.stop_token_ids = [
            torch.LongTensor(self.tokenizer(x)['input_ids'])
            for x in stop_strings
        ]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

class LLM:
    """
    A class for loading and generating text using Language Models with support for 
    quantization and custom stopping criteria.
    """
    def __init__(
        self, 
        model_id: str, 
        device: str = 'cuda', 
        quantization_bits: Optional[int] = None, 
        stop_list: Optional[List[str]] = None, 
        model_max_length: int = 4096
    ):
        self.device = device
        self.model_max_length = model_max_length
        self.llm_id = model_id

        self.stop_list = stop_list
        if stop_list is None:
            self.stop_list = ['\nHuman:', '\n```\n', '\nQuestion:', '<|endoftext|>', '\n']
        
        self.bnb_config = self._set_quantization(quantization_bits)
        self.model, self.tokenizer = self._initialize_model_tokenizer(model_id)
        

    def _set_quantization(self, quantization_bits: Optional[int]) -> Optional[BitsAndBytesConfig]:
        """Configure quantization settings"""
        if quantization_bits in [4, 8]:
            bnb_config = BitsAndBytesConfig()
            if quantization_bits == 4:
                bnb_config.load_in_4bit = True
                bnb_config.bnb_4bit_quant_type = 'nf4'
                bnb_config.bnb_4bit_use_double_quant = True
                bnb_config.bnb_4bit_compute_dtype = torch.bfloat16
            elif quantization_bits == 8:
                bnb_config.load_in_8bit = True
            return bnb_config
        return None
 

    def _initialize_model_tokenizer(self, model_id: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """Initialize model and tokenizer"""
        model_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_config.max_seq_len = self.model_max_length

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=self.bnb_config,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            padding_side="left", 
            truncation_side="left",
            model_max_length=self.model_max_length
        )
        tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    
    def generate(self, prompts: List[str], max_new_tokens: int = 15) -> List[str]:
        """Generate text responses for the given prompts"""
        # Handle different stopping sequences based on model type
        stop_sequence = "\n### Human:" if 'mpt' in self.llm_id else "\nQuestion:"
        stopping_criteria = StoppingCriteriaList([
            StopOnTokens(
                self.tokenizer,
                [stop_sequence, "\n```\n", "<|endoftext|>", "\n"]
            )
        ])

        all_responses = []
        for prompt in prompts:
            inputs = self.tokenizer(
                prompt, 
                padding=True, 
                truncation=True, 
                max_length=self.model_max_length, 
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=1.1,
                    stopping_criteria=stopping_criteria,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
                response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                all_responses.append(response)

        return all_responses