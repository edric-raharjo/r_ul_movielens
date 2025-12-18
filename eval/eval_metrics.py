# eval/eval_metrics.py

import torch
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict


class RLRecommendationMetrics:
    """
    Comprehensive metrics for RL-based recommendation systems.
    Supports both per-transition and per-episode evaluation.
    """
    
    def __init__(self, k_values: List[int] = [1, 5, 10]):
        """
        Args:
            k_values: List of K values for Top-K accuracy metrics
        """
        self.k_values = sorted(k_values)
        self.reset()
    
    def reset(self):
        """Reset all metric accumulators"""
        # Per-user episode tracking
        self.user_episodes = defaultdict(lambda: {
            'transitions': [],
            'rewards': [],
            'top1_correct': 0,
            'topk_correct': {k: 0 for k in self.k_values},
            'total_transitions': 0
        })
        
    def update(
        self, 
        user_id: int,
        predicted_action: int,
        actual_action: int,
        reward: float,
        top_k_actions: Dict[int, List[int]] = None
    ):
        """
        Update metrics with a single transition.
        
        Args:
            user_id: User identifier
            predicted_action: DQN's top-1 predicted action (movie index)
            actual_action: Ground truth action (movie index)
            reward: Actual reward received (rating)
            top_k_actions: Dict mapping k -> list of top-k predicted actions
                          e.g., {5: [action1, action2, ...], 10: [...]}
        """
        episode = self.user_episodes[user_id]
        
        # Record transition
        episode['transitions'].append({
            'predicted': predicted_action,
            'actual': actual_action,
            'reward': reward
        })
        episode['rewards'].append(reward)
        episode['total_transitions'] += 1
        
        # Top-1 accuracy
        if predicted_action == actual_action:
            episode['top1_correct'] += 1
        
        # Top-K accuracy
        if top_k_actions is not None:
            for k in self.k_values:
                if k in top_k_actions and actual_action in top_k_actions[k]:
                    episode['topk_correct'][k] += 1
    
    def compute_user_metrics(self, user_id: int) -> Dict[str, float]:
        """
        Compute all metrics for a single user's episode.
        
        Returns:
            Dictionary with user-level metrics
        """
        episode = self.user_episodes[user_id]
        n_trans = episode['total_transitions']
        
        if n_trans == 0:
            return None
        
        metrics = {
            'user_id': user_id,
            'episode_length': n_trans,
            'cumulative_reward': sum(episode['rewards']),
            'average_reward': np.mean(episode['rewards']),
            'top1_accuracy': episode['top1_correct'] / n_trans,
        }
        
        # Top-K accuracies
        for k in self.k_values:
            metrics[f'top{k}_accuracy'] = episode['topk_correct'][k] / n_trans
        
        return metrics
    
    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """
        Compute aggregate metrics across all users.
        
        Returns:
            Dictionary with dataset-level metrics
        """
        if len(self.user_episodes) == 0:
            return {}
        
        # Collect per-user metrics
        user_metrics_list = []
        for user_id in self.user_episodes.keys():
            user_metrics = self.compute_user_metrics(user_id)
            if user_metrics is not None:
                user_metrics_list.append(user_metrics)
        
        if len(user_metrics_list) == 0:
            return {}
        
        # Aggregate across users
        n_users = len(user_metrics_list)
        total_transitions = sum(m['episode_length'] for m in user_metrics_list)
        
        aggregate = {
            'n_users': n_users,
            'total_transitions': total_transitions,
            'mean_episode_length': np.mean([m['episode_length'] for m in user_metrics_list]),
            'std_episode_length': np.std([m['episode_length'] for m in user_metrics_list]),
            
            # Mean cumulative reward (what the paper reports)
            'mean_cumulative_reward': np.mean([m['cumulative_reward'] for m in user_metrics_list]),
            'std_cumulative_reward': np.std([m['cumulative_reward'] for m in user_metrics_list]),
            
            # Mean average reward (more interpretable)
            'mean_avg_reward': np.mean([m['average_reward'] for m in user_metrics_list]),
            'std_avg_reward': np.std([m['average_reward'] for m in user_metrics_list]),
            
            # Top-1 accuracy (what the paper reports)
            'mean_top1_accuracy': np.mean([m['top1_accuracy'] for m in user_metrics_list]),
            'std_top1_accuracy': np.std([m['top1_accuracy'] for m in user_metrics_list]),
        }
        
        # Top-K accuracies
        for k in self.k_values:
            key = f'top{k}_accuracy'
            aggregate[f'mean_{key}'] = np.mean([m[key] for m in user_metrics_list])
            aggregate[f'std_{key}'] = np.std([m[key] for m in user_metrics_list])
        
        # Per-transition aggregate (alternative view)
        all_rewards = []
        all_top1_correct = []
        for user_id, episode in self.user_episodes.items():
            all_rewards.extend(episode['rewards'])
            all_top1_correct.extend([1 if t['predicted'] == t['actual'] else 0 
                                     for t in episode['transitions']])
        
        aggregate['global_avg_reward'] = np.mean(all_rewards)
        aggregate['global_top1_accuracy'] = np.mean(all_top1_correct)
        
        return aggregate
    
    def get_user_metrics_list(self) -> List[Dict[str, float]]:
        """Get list of per-user metrics for detailed analysis"""
        return [self.compute_user_metrics(uid) for uid in self.user_episodes.keys()]
    
    def print_summary(self, title: str = "Evaluation Results"):
        """Print formatted summary of metrics"""
        metrics = self.compute_aggregate_metrics()
        
        if not metrics:
            print(f"\n{title}: No data available")
            return
        
        print("\n" + "=" * 60)
        print(f"{title}")
        print("=" * 60)
        print(f"Users evaluated:      {metrics['n_users']}")
        print(f"Total transitions:    {metrics['total_transitions']}")
        print(f"Avg episode length:   {metrics['mean_episode_length']:.2f} ± {metrics['std_episode_length']:.2f}")
        
        print("\nPer-Transition Metrics:")
        print(f"  Top-1 Accuracy:     {metrics['mean_top1_accuracy']*100:6.2f}% ± {metrics['std_top1_accuracy']*100:.2f}%")
        for k in self.k_values:
            if k > 1:
                mean_key = f'mean_top{k}_accuracy'
                std_key = f'std_top{k}_accuracy'
                print(f"  Top-{k} Accuracy:    {metrics[mean_key]*100:6.2f}% ± {metrics[std_key]*100:.2f}%")
        print(f"  Avg Reward/Trans:   {metrics['mean_avg_reward']:.3f} ± {metrics['std_avg_reward']:.3f} stars")
        
        print("\nPer-Episode Metrics:")
        print(f"  Mean Cumulative Reward:  {metrics['mean_cumulative_reward']:.2f} ± {metrics['std_cumulative_reward']:.2f}")
        print(f"  Mean Avg Reward:         {metrics['mean_avg_reward']:.3f} ± {metrics['std_avg_reward']:.3f} stars")
        
        print("\nGlobal (All Transitions):")
        print(f"  Global Top-1 Accuracy:   {metrics['global_top1_accuracy']*100:.2f}%")
        print(f"  Global Avg Reward:       {metrics['global_avg_reward']:.3f} stars")
        print("=" * 60)
        
        return metrics


def compare_metrics(
    before_metrics: Dict[str, float],
    after_metrics: Dict[str, float],
    title: str = "Comparison: Before vs After Unlearning"
):
    """
    Compare two sets of metrics (e.g., before/after unlearning).
    
    Args:
        before_metrics: Metrics from compute_aggregate_metrics() before unlearning
        after_metrics: Metrics from compute_aggregate_metrics() after unlearning
        title: Comparison title
    """
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    
    metrics_to_compare = [
        ('mean_top1_accuracy', 'Top-1 Accuracy', '%', 100),
        ('mean_cumulative_reward', 'Mean Cumulative Reward', '', 1),
        ('mean_avg_reward', 'Mean Avg Reward', ' stars', 1),
    ]
    
    print(f"{'Metric':<30} {'Before':>12} {'After':>12} {'Change':>12} {'Drop %':>10}")
    print("-" * 70)
    
    for key, name, unit, scale in metrics_to_compare:
        before_val = before_metrics.get(key, 0) * scale
        after_val = after_metrics.get(key, 0) * scale
        change = after_val - before_val
        drop_pct = (change / before_val * 100) if before_val != 0 else 0
        
        print(f"{name:<30} {before_val:>10.2f}{unit:>2} {after_val:>10.2f}{unit:>2} "
              f"{change:>10.2f}{unit:>2} {drop_pct:>9.1f}%")
    
    print("=" * 70)


def save_metrics(metrics: Dict, user_metrics_list: List[Dict], filepath: str):
    """Save metrics to file for later analysis"""
    import json
    
    data = {
        'aggregate': metrics,
        'per_user': user_metrics_list
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Metrics saved to {filepath}")


def load_metrics(filepath: str) -> Tuple[Dict, List[Dict]]:
    """Load metrics from file"""
    import json
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data['aggregate'], data['per_user']
