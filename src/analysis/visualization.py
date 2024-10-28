import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict
import json
import os

class ResultsVisualizer:
    """Handles visualization of experiment results"""
    
    def __init__(self, output_dir: str = "analysis_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_theme()
        sns.set_palette("husl")
        
    def plot_distracting_impact(
        self,
        results: Dict[str, List[float]],
        model_names: List[str],
        save_path: str = None
    ):
        """Plot impact of distracting documents on accuracy"""
        plt.figure(figsize=(12, 6))
        
        for model in model_names:
            accuracies = results[model]
            plt.plot(range(len(accuracies)), accuracies, marker='o', label=model)
            
        plt.xlabel("Number of Distracting Documents")
        plt.ylabel("Accuracy")
        plt.title("Impact of Distracting Documents on Model Accuracy")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_position_comparison(
        self,
        results: Dict[str, Dict[str, List[float]]],
        model_names: List[str],
        positions: List[str] = ["far", "mid", "near"],  # Changed to lowercase
        save_path: str = None
    ):
        """Plot accuracy comparison for different document positions"""
        plt.figure(figsize=(15, 6))
        
        x = np.arange(len(positions))
        width = 0.8 / len(model_names)
        
        for i, model in enumerate(model_names):
            try:
                # Get accuracies for each position, with error handling
                accuracies = []
                for pos in positions:
                    try:
                        # Try to get the first value if it's a list
                        if isinstance(results[model][pos], list):
                            accuracies.append(results[model][pos][0])
                        else:
                            accuracies.append(results[model][pos])
                    except (KeyError, IndexError) as e:
                        print(f"Warning: Missing data for model {model}, position {pos}")
                        accuracies.append(0)  # or some default value
                
                plt.bar(x + i*width, accuracies, width, label=model)
            except Exception as e:
                print(f"Error processing model {model}: {str(e)}")
                continue
            
        plt.xlabel("Gold Document Position")
        plt.ylabel("Accuracy")
        plt.title("Impact of Gold Document Position on Model Accuracy")
        plt.xticks(x + width*len(model_names)/2, [p.capitalize() for p in positions])
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_random_impact(
        self,
        results: Dict[str, List[float]],
        model_names: List[str],
        save_path: str = None
    ):
        """Plot impact of random documents on accuracy"""
        plt.figure(figsize=(12, 6))
        
        for model in model_names:
            accuracies = results[model]
            plt.plot(range(len(accuracies)), accuracies, marker='o', label=model)
            
        plt.xlabel("Number of Random Documents")
        plt.ylabel("Accuracy")
        plt.title("Impact of Random Documents on Model Accuracy")
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_comparison_table(
        self,
        paper_results: Dict[str, Dict[str, float]],
        our_results: Dict[str, Dict[str, float]],
        save_path: str = None
    ) -> pd.DataFrame:
        """Create comparison table between paper results and our results"""
        data = []
        
        try:
            for model in paper_results.keys():
                paper_metrics = paper_results[model]
                our_metrics = our_results.get(model, {})
                
                for metric in paper_metrics.keys():
                    data.append({
                        'Model': model,
                        'Metric': metric,
                        'Paper Result': paper_metrics[metric],
                        'Our Result': our_metrics.get(metric, float('nan')),
                        'Difference': our_metrics.get(metric, float('nan')) - paper_metrics[metric]
                    })
        except Exception as e:
            print(f"Error creating comparison table: {str(e)}")
            data = []
        
        df = pd.DataFrame(data)
        
        if save_path:
            df.to_csv(save_path, index=False)
            
        return df
    
    def plot_attention_heatmap(
        self,
        attention_scores: np.ndarray,
        document_labels: List[str],
        save_path: str = None
    ):
        """Plot attention heatmap"""
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(
            attention_scores,
            cmap='Blues',
            xticklabels=document_labels,
            yticklabels=[f'Layer {i+1}' for i in range(len(attention_scores))],
            cbar_kws={'label': 'Attention Score'}
        )
        
        plt.xlabel("Documents in Context")
        plt.ylabel("Attention Layers")
        plt.title("Attention Distribution Across Documents")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()