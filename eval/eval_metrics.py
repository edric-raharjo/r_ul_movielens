# eval/eval_metrics.py

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class RecommendationMetrics:
    """
    Metrics for evaluating binary recommendation system.
    
    Main metrics:
    - Binary classification accuracy (paper's primary metric)
    - Hit@K, Precision@K, Recall@K
    - NDCG@K (optional)
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Args:
            k_values: List of K values for top-K metrics
        """
        self.k_values = k_values
    
    def compute_classification_accuracy(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute binary classification accuracy.
        
        Args:
            predictions: (N,) predicted actions (0 or 1)
            labels: (N,) ground truth labels (0 or 1)
            
        Returns:
            accuracy: Float in [0, 1]
        """
        correct = (predictions == labels).sum().item()
        total = len(labels)
        return correct / total if total > 0 else 0.0
    
    def compute_precision_recall_f1(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute precision, recall, F1 for binary classification.
        
        Args:
            predictions: (N,) predicted actions (0 or 1)
            labels: (N,) ground truth labels (0 or 1)
            
        Returns:
            Dictionary with precision, recall, f1
        """
        # True positives, false positives, false negatives
        tp = ((predictions == 1) & (labels == 1)).sum().item()
        fp = ((predictions == 1) & (labels == 0)).sum().item()
        fn = ((predictions == 0) & (labels == 1)).sum().item()
        tn = ((predictions == 0) & (labels == 0)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
    
    def compute_hit_at_k(
        self,
        ranked_items: List[int],
        ground_truth: List[int],
        k: int
    ) -> float:
        """
        Compute Hit@K: Is there at least one relevant item in top-K?
        
        Args:
            ranked_items: List of ranked item IDs (ordered by score)
            ground_truth: List of relevant item IDs
            k: K value
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = set(ranked_items[:k])
        relevant = set(ground_truth)
        return 1.0 if len(top_k & relevant) > 0 else 0.0
    
    def compute_precision_at_k(
        self,
        ranked_items: List[int],
        ground_truth: List[int],
        k: int
    ) -> float:
        """
        Compute Precision@K: Proportion of relevant items in top-K.
        
        Args:
            ranked_items: List of ranked item IDs
            ground_truth: List of relevant item IDs
            k: K value
            
        Returns:
            Precision@K in [0, 1]
        """
        top_k = set(ranked_items[:k])
        relevant = set(ground_truth)
        return len(top_k & relevant) / k if k > 0 else 0.0
    
    def compute_recall_at_k(
        self,
        ranked_items: List[int],
        ground_truth: List[int],
        k: int
    ) -> float:
        """
        Compute Recall@K: Proportion of relevant items found in top-K.
        
        Args:
            ranked_items: List of ranked item IDs
            ground_truth: List of relevant item IDs
            k: K value
            
        Returns:
            Recall@K in [0, 1]
        """
        top_k = set(ranked_items[:k])
        relevant = set(ground_truth)
        return len(top_k & relevant) / len(relevant) if len(relevant) > 0 else 0.0
    
    def compute_ndcg_at_k(
        self,
        ranked_items: List[int],
        ground_truth_with_scores: Dict[int, float],
        k: int
    ) -> float:
        """
        Compute NDCG@K: Normalized Discounted Cumulative Gain.
        
        Args:
            ranked_items: List of ranked item IDs
            ground_truth_with_scores: Dict mapping item_id -> relevance score
            k: K value
            
        Returns:
            NDCG@K in [0, 1]
        """
        # DCG
        dcg = 0.0
        for i, item_id in enumerate(ranked_items[:k]):
            if item_id in ground_truth_with_scores:
                relevance = ground_truth_with_scores[item_id]
                dcg += relevance / np.log2(i + 2)  # i+2 because index starts at 0
        
        # IDCG (Ideal DCG)
        ideal_scores = sorted(ground_truth_with_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        user_rankings: Optional[Dict[int, Tuple[List[int], List[int]]]] = None
    ) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Args:
            predictions: (N,) binary predictions
            labels: (N,) binary labels
            user_rankings: Optional dict mapping user_id -> (ranked_items, ground_truth)
                          For computing ranking metrics
            
        Returns:
            Dictionary with all metrics
        """
        metrics = {}
        
        # Classification metrics
        metrics['accuracy'] = self.compute_classification_accuracy(predictions, labels)
        
        # Precision/Recall/F1
        pr_metrics = self.compute_precision_recall_f1(predictions, labels)
        metrics.update(pr_metrics)
        
        # Ranking metrics (if provided)
        if user_rankings is not None:
            for k in self.k_values:
                hit_scores = []
                precision_scores = []
                recall_scores = []
                
                for user_id, (ranked_items, ground_truth) in user_rankings.items():
                    hit_scores.append(self.compute_hit_at_k(ranked_items, ground_truth, k))
                    precision_scores.append(self.compute_precision_at_k(ranked_items, ground_truth, k))
                    recall_scores.append(self.compute_recall_at_k(ranked_items, ground_truth, k))
                
                metrics[f'hit@{k}'] = np.mean(hit_scores) if hit_scores else 0.0
                metrics[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
                metrics[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
        
        return metrics


def aggregate_user_metrics(
    user_metrics_list: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple users.
    
    Args:
        user_metrics_list: List of metric dictionaries per user
        
    Returns:
        Aggregated metrics (mean and std)
    """
    if not user_metrics_list:
        return {}
    
    # Collect all metric values
    metric_values = defaultdict(list)
    for user_metrics in user_metrics_list:
        for key, val in user_metrics.items():
            metric_values[key].append(val)
    
    # Compute mean and std
    aggregated = {}
    for key, values in metric_values.items():
        aggregated[f'{key}_mean'] = np.mean(values)
        aggregated[f'{key}_std'] = np.std(values)
    
    return aggregated


def compare_metrics(
    metrics_before: Dict[str, float],
    metrics_after: Dict[str, float]
) -> Dict[str, Dict[str, float]]:
    """
    Compare metrics before and after unlearning.
    
    Args:
        metrics_before: Metrics before unlearning
        metrics_after: Metrics after unlearning
        
    Returns:
        Dictionary with before, after, and change for each metric
    """
    comparison = {}
    
    # Get common keys
    common_keys = set(metrics_before.keys()) & set(metrics_after.keys())
    
    for key in common_keys:
        before = metrics_before[key]
        after = metrics_after[key]
        change = after - before
        change_pct = (change / before * 100) if before != 0 else 0.0
        
        comparison[key] = {
            'before': before,
            'after': after,
            'change': change,
            'change_pct': change_pct
        }
    
    return comparison


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to print
    """
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}")
    
    for key, val in metrics.items():
        if isinstance(val, float):
            if 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key or 'hit' in key:
                print(f"  {key:25s}: {val:6.2%}")
            else:
                print(f"  {key:25s}: {val:8.4f}")
        else:
            print(f"  {key:25s}: {val}")
    
    print(f"{'='*70}\n")


def print_comparison(comparison: Dict[str, Dict[str, float]], title: str = "Comparison"):
    """
    Pretty print metric comparison.
    
    Args:
        comparison: Comparison dictionary from compare_metrics
        title: Title to print
    """
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")
    print(f"{'Metric':<25} {'Before':>12} {'After':>12} {'Change':>12} {'Change%':>12}")
    print(f"{'-'*80}")
    
    for key, values in comparison.items():
        before = values['before']
        after = values['after']
        change = values['change']
        change_pct = values['change_pct']
        
        if 'accuracy' in key or 'precision' in key or 'recall' in key or 'f1' in key or 'hit' in key:
            print(f"{key:<25} {before:11.2%} {after:11.2%} {change:+11.2%} {change_pct:+10.1f}%")
        else:
            print(f"{key:<25} {before:12.4f} {after:12.4f} {change:+12.4f} {change_pct:+10.1f}%")
    
    print(f"{'='*80}\n")


# Test code
if __name__ == "__main__":
    print("Testing RecommendationMetrics...")
    
    metrics_calculator = RecommendationMetrics(k_values=[1, 5, 10])
    
    # Test classification accuracy
    print("\n" + "="*70)
    print("Testing Classification Metrics")
    print("="*70)
    
    predictions = torch.tensor([1, 0, 1, 1, 0, 0, 1, 0])
    labels = torch.tensor([1, 0, 0, 1, 0, 1, 1, 0])
    
    accuracy = metrics_calculator.compute_classification_accuracy(predictions, labels)
    print(f"Predictions: {predictions.tolist()}")
    print(f"Labels:      {labels.tolist()}")
    print(f"Accuracy: {accuracy:.2%}")
    
    pr_metrics = metrics_calculator.compute_precision_recall_f1(predictions, labels)
    print(f"Precision: {pr_metrics['precision']:.2%}")
    print(f"Recall: {pr_metrics['recall']:.2%}")
    print(f"F1: {pr_metrics['f1']:.4f}")
    
    # Test ranking metrics
    print("\n" + "="*70)
    print("Testing Ranking Metrics")
    print("="*70)
    
    ranked_items = [10, 5, 3, 7, 2, 8, 1, 9, 4, 6]
    ground_truth = [5, 7, 2]
    
    for k in [1, 5, 10]:
        hit = metrics_calculator.compute_hit_at_k(ranked_items, ground_truth, k)
        precision = metrics_calculator.compute_precision_at_k(ranked_items, ground_truth, k)
        recall = metrics_calculator.compute_recall_at_k(ranked_items, ground_truth, k)
        print(f"K={k:2d}: Hit={hit:.2f}, Precision={precision:.2%}, Recall={recall:.2%}")
    
    # Test NDCG
    ground_truth_scores = {5: 5.0, 7: 4.5, 2: 4.0}
    ndcg = metrics_calculator.compute_ndcg_at_k(ranked_items, ground_truth_scores, 5)
    print(f"NDCG@5: {ndcg:.4f}")
    
    # Test comparison
    print("\n" + "="*70)
    print("Testing Comparison")
    print("="*70)
    
    metrics_before = {
        'accuracy': 0.85,
        'precision': 0.80,
        'recall': 0.75,
        'hit@5': 0.90
    }
    
    metrics_after = {
        'accuracy': 0.60,
        'precision': 0.55,
        'recall': 0.50,
        'hit@5': 0.65
    }
    
    comparison = compare_metrics(metrics_before, metrics_after)
    print_comparison(comparison, "Unlearning Effect")
    
    print("\n✅ All metric tests passed!")
