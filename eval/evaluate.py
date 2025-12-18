# eval/evaluate.py

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import json

from model.dqn_model import RecommendationDQN, DQNWithTargetNetwork
from dataset.dataloader import MovieLensRecommendationDataset
from eval.eval_metrics import (
    RecommendationMetrics, 
    print_metrics, 
    print_comparison,
    compare_metrics
)


class RecommendationEvaluator:
    """
    Evaluator for recommendation DQN model.
    """
    
    def __init__(
        self,
        model: RecommendationDQN,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        k_values: List[int] = [1, 5, 10]
    ):
        """
        Args:
            model: Trained DQN model
            device: Device to use
            k_values: K values for ranking metrics
        """
        self.model = model
        self.device = device
        self.metrics_calculator = RecommendationMetrics(k_values=k_values)
        self.k_values = k_values
        
        self.model.to(device)
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_classification(
        self,
        dataset: MovieLensRecommendationDataset,
        batch_size: int = 128
    ) -> Dict[str, float]:
        """
        Evaluate binary classification performance.
        
        Args:
            dataset: Test dataset
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with classification metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_rewards = []
        
        # Process in batches
        num_samples = len(dataset)
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for i in tqdm(range(num_batches), desc="Evaluating classification"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)
            
            # Get batch
            batch_inputs = []
            batch_labels = []
            batch_rewards = []
            
            for j in range(start_idx, end_idx):
                sample = dataset[j]
                batch_inputs.append(sample['input_features'])
                batch_labels.append(sample['label'])
                batch_rewards.append(sample['reward'])
            
            # Stack and move to device
            inputs = torch.stack(batch_inputs).to(self.device)
            labels = torch.stack(batch_labels).squeeze().to(self.device)
            rewards = torch.stack(batch_rewards).squeeze().to(self.device)
            
            # Forward pass
            q_values = self.model(inputs)
            predictions = q_values.argmax(dim=1)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
            all_rewards.append(rewards.cpu())
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_rewards = torch.cat(all_rewards)
        
        # Compute metrics
        metrics = self.metrics_calculator.compute_all_metrics(
            all_predictions, all_labels
        )
        
        # Add average reward
        metrics['avg_reward'] = all_rewards.mean().item()
        
        return metrics
    
    @torch.no_grad()
    def evaluate_ranking(
        self,
        dataset: MovieLensRecommendationDataset,
        num_candidates: int = 100,
        rating_threshold: float = 4.0
    ) -> Dict[str, float]:
        """
        Evaluate ranking performance (top-K recommendation).
        
        Args:
            dataset: Test dataset
            num_candidates: Number of candidate items to rank per user
            rating_threshold: Threshold for positive items
            
        Returns:
            Dictionary with ranking metrics
        """
        self.model.eval()
        
        # Group samples by user
        user_samples = {}
        for sample in dataset.samples:
            user_id = sample['user_id']
            if user_id not in user_samples:
                user_samples[user_id] = []
            user_samples[user_id].append(sample)
        
        print(f"Evaluating {len(user_samples)} users for ranking...")
        
        user_rankings = {}
        
        for user_id, samples in tqdm(user_samples.items(), desc="Ranking evaluation"):
            # Get ground truth positive items
            ground_truth = [
                s['movie_id'] for s in samples 
                if s['reward'] >= rating_threshold
            ]
            
            if len(ground_truth) == 0:
                continue
            
            # Limit candidates if needed
            if len(samples) > num_candidates:
                samples = np.random.choice(samples, num_candidates, replace=False)
            
            # Score all candidates
            scores = []
            movie_ids = []
            
            for sample in samples:
                input_vec = torch.cat([
                    torch.FloatTensor(sample['user_state']),
                    torch.FloatTensor(sample['candidate_features'])
                ]).unsqueeze(0).to(self.device)
                
                q_values = self.model(input_vec)
                q_recommend = q_values[0, 1].item()  # Q(recommend)
                
                scores.append(q_recommend)
                movie_ids.append(sample['movie_id'])
            
            # Rank by Q(recommend)
            ranked_indices = np.argsort(scores)[::-1]  # Descending
            ranked_items = [movie_ids[i] for i in ranked_indices]
            
            user_rankings[user_id] = (ranked_items, ground_truth)
        
        # Compute ranking metrics
        ranking_metrics = {}
        for k in self.k_values:
            hit_scores = []
            precision_scores = []
            recall_scores = []
            
            for user_id, (ranked_items, ground_truth) in user_rankings.items():
                hit_scores.append(
                    self.metrics_calculator.compute_hit_at_k(ranked_items, ground_truth, k)
                )
                precision_scores.append(
                    self.metrics_calculator.compute_precision_at_k(ranked_items, ground_truth, k)
                )
                recall_scores.append(
                    self.metrics_calculator.compute_recall_at_k(ranked_items, ground_truth, k)
                )
            
            ranking_metrics[f'hit@{k}'] = np.mean(hit_scores)
            ranking_metrics[f'precision@{k}'] = np.mean(precision_scores)
            ranking_metrics[f'recall@{k}'] = np.mean(recall_scores)
        
        return ranking_metrics
    
    def evaluate_full(
        self,
        dataset: MovieLensRecommendationDataset,
        batch_size: int = 128,
        include_ranking: bool = True,
        num_candidates: int = 100
    ) -> Dict[str, float]:
        """
        Full evaluation: classification + ranking.
        
        Args:
            dataset: Test dataset
            batch_size: Batch size
            include_ranking: Whether to compute ranking metrics
            num_candidates: Number of candidates for ranking
            
        Returns:
            Complete metrics dictionary
        """
        metrics = {}
        
        # Classification metrics
        print("Computing classification metrics...")
        classification_metrics = self.evaluate_classification(dataset, batch_size)
        metrics.update(classification_metrics)
        
        # Ranking metrics
        if include_ranking:
            print("Computing ranking metrics...")
            ranking_metrics = self.evaluate_ranking(dataset, num_candidates)
            metrics.update(ranking_metrics)
        
        return metrics


def evaluate_before_after_unlearning(
    model_before_path: str,
    model_after_path: str,
    retain_dataset: MovieLensRecommendationDataset,
    forget_dataset: MovieLensRecommendationDataset,
    device: str = 'cuda',
    save_path: Optional[str] = None
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate and compare models before and after unlearning.
    
    Args:
        model_before_path: Path to model before unlearning
        model_after_path: Path to model after unlearning
        retain_dataset: Retain set dataset
        forget_dataset: Forget set dataset
        device: Device to use
        save_path: Optional path to save results
        
    Returns:
        Dictionary with all results
    """
    print("\n" + "="*80)
    print("EVALUATION: BEFORE vs AFTER UNLEARNING")
    print("="*80)
    
    # Load models
    print("\nLoading models...")
    model_before = RecommendationDQN()
    model_before.load_state_dict(torch.load(model_before_path, map_location=device))
    
    model_after = RecommendationDQN()
    model_after.load_state_dict(torch.load(model_after_path, map_location=device))
    
    # Create evaluators
    evaluator_before = RecommendationEvaluator(model_before, device)
    evaluator_after = RecommendationEvaluator(model_after, device)
    
    results = {}
    
    # Evaluate RETAIN set
    print("\n" + "="*80)
    print("RETAIN SET EVALUATION")
    print("="*80)
    
    print("\nBefore unlearning:")
    retain_before = evaluator_before.evaluate_full(retain_dataset)
    print_metrics(retain_before, "RETAIN - Before Unlearning")
    
    print("\nAfter unlearning:")
    retain_after = evaluator_after.evaluate_full(retain_dataset)
    print_metrics(retain_after, "RETAIN - After Unlearning")
    
    # Compare
    retain_comparison = compare_metrics(retain_before, retain_after)
    print_comparison(retain_comparison, "RETAIN SET: Before vs After")
    
    results['retain'] = {
        'before': retain_before,
        'after': retain_after,
        'comparison': retain_comparison
    }
    
    # Evaluate FORGET set
    print("\n" + "="*80)
    print("FORGET SET EVALUATION")
    print("="*80)
    
    print("\nBefore unlearning:")
    forget_before = evaluator_before.evaluate_full(forget_dataset)
    print_metrics(forget_before, "FORGET - Before Unlearning")
    
    print("\nAfter unlearning:")
    forget_after = evaluator_after.evaluate_full(forget_dataset)
    print_metrics(forget_after, "FORGET - After Unlearning")
    
    # Compare
    forget_comparison = compare_metrics(forget_before, forget_after)
    print_comparison(forget_comparison, "FORGET SET: Before vs After")
    
    results['forget'] = {
        'before': forget_before,
        'after': forget_after,
        'comparison': forget_comparison
    }
    
    # Summary
    print("\n" + "="*80)
    print("UNLEARNING SUMMARY")
    print("="*80)
    print(f"{'Metric':<25} {'Retain Change':>15} {'Forget Change':>15}")
    print("-"*80)
    
    key_metrics = ['accuracy', 'precision', 'recall', 'hit@5', 'hit@10']
    for metric in key_metrics:
        if metric in retain_comparison and metric in forget_comparison:
            retain_change = retain_comparison[metric]['change_pct']
            forget_change = forget_comparison[metric]['change_pct']
            print(f"{metric:<25} {retain_change:+14.1f}% {forget_change:+14.1f}%")
    
    print("="*80)
    
    # Save results
    if save_path:
        with open(save_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_to_native(obj):
                if isinstance(obj, dict):
                    return {k: convert_to_native(v) for k, v in obj.items()}
                elif isinstance(obj, (np.integer, np.floating)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return obj
            
            json.dump(convert_to_native(results), f, indent=2)
        print(f"\nResults saved to {save_path}")
    
    return results


# Test code
if __name__ == "__main__":
    print("Testing Evaluator...")
    print("Note: This requires a trained model and dataset to run properly.")
    print("Run train.py first to generate models and datasets.")
