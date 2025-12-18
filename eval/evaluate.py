# eval/evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Optional
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from eval.eval_metrics import RLRecommendationMetrics, compare_metrics, save_metrics
from model.dqn_model import DQN  # You'll implement this
from dataset.dataloader import MovieLensRLDataset, create_dataloaders


class DQNEvaluator:
    """
    Evaluator for DQN-based recommendation system.
    Handles evaluation on retain/forget sets with comprehensive metrics.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        k_values: List[int] = [1, 5, 10]
    ):
        """
        Args:
            model: Trained DQN model
            device: Device to run evaluation on
            k_values: K values for Top-K accuracy
        """
        self.model = model.to(device)
        self.device = device
        self.k_values = k_values
        self.model.eval()
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader: DataLoader,
        dataset: MovieLensRLDataset,
        desc: str = "Evaluating"
    ) -> RLRecommendationMetrics:
        """Evaluate DQN on a dataset with action masking"""
        metrics = RLRecommendationMetrics(k_values=self.k_values)
        
        for batch in tqdm(dataloader, desc=desc):
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            user_ids = batch['user_id']
            
            # Get Q-values
            q_values = self.model(states)  # (batch_size, num_movies)
            
            # Apply action masks for each user
            for i in range(len(user_ids)):
                user_id = user_ids[i].item() if torch.is_tensor(user_ids[i]) else user_ids[i]
                
                # Get user's action mask
                action_mask = dataset.get_user_action_mask(user_id).to(self.device)
                
                # Mask invalid actions
                masked_q = q_values[i].clone()
                masked_q[~action_mask] = float('-inf')
                
                # Get top-1 prediction
                predicted_action = masked_q.argmax().item()
                
                # Get top-K predictions
                topk_dict = {}
                for k in self.k_values:
                    # Only consider valid actions for top-k
                    valid_k = min(k, action_mask.sum().item())
                    _, topk_indices = torch.topk(masked_q, k=valid_k)
                    topk_dict[k] = topk_indices.cpu().numpy().tolist()
                
                # Update metrics
                metrics.update(
                    user_id=user_id,
                    predicted_action=predicted_action,
                    actual_action=actions[i].item(),
                    reward=rewards[i].item(),
                    top_k_actions=topk_dict
                )
        
        return metrics
    
    def evaluate_split(
        self,
        dataloaders: Dict[str, DataLoader],
        dataset: MovieLensRLDataset,
        splits: List[str] = ['retain', 'forget']
    ) -> Dict[str, RLRecommendationMetrics]:
        """
        Evaluate on multiple splits (retain/forget).
        
        Args:
            dataloaders: Dict mapping split name -> DataLoader
            dataset: Dataset object
            splits: List of split names to evaluate
            
        Returns:
            Dict mapping split name -> RLRecommendationMetrics
        """
        results = {}
        
        for split in splits:
            if split not in dataloaders:
                print(f"Warning: Split '{split}' not found in dataloaders")
                continue
            
            print(f"\nEvaluating {split.upper()} set...")
            metrics = self.evaluate_dataset(
                dataloaders[split],
                dataset,
                desc=f"Evaluating {split}"
            )
            results[split] = metrics
            metrics.print_summary(title=f"{split.upper()} Set Results")
        
        return results


def evaluate_unlearning(
    model_before: nn.Module,
    model_after: nn.Module,
    dataloaders: Dict[str, DataLoader],
    dataset: MovieLensRLDataset,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: Optional[str] = None
) -> Dict:
    """
    Comprehensive unlearning evaluation comparing before/after models.
    
    Args:
        model_before: DQN model before unlearning
        model_after: DQN model after unlearning
        dataloaders: Dict with 'retain' and 'forget' dataloaders
        dataset: Dataset object
        device: Device for evaluation
        save_dir: Directory to save results (optional)
        
    Returns:
        Dict containing all evaluation results
    """
    print("\n" + "=" * 70)
    print("UNLEARNING EVALUATION")
    print("=" * 70)
    
    # Evaluate BEFORE unlearning
    print("\n>>> BEFORE UNLEARNING <<<")
    evaluator_before = DQNEvaluator(model_before, device)
    results_before = evaluator_before.evaluate_split(dataloaders, dataset)
    
    # Evaluate AFTER unlearning
    print("\n>>> AFTER UNLEARNING <<<")
    evaluator_after = DQNEvaluator(model_after, device)
    results_after = evaluator_after.evaluate_split(dataloaders, dataset)
    
    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON ANALYSIS")
    print("=" * 70)
    
    for split in ['forget', 'retain']:
        if split in results_before and split in results_after:
            before_metrics = results_before[split].compute_aggregate_metrics()
            after_metrics = results_after[split].compute_aggregate_metrics()
            compare_metrics(
                before_metrics,
                after_metrics,
                title=f"{split.upper()} Set: Before vs After Unlearning"
            )
    
    # Save results if directory provided
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ['forget', 'retain']:
            if split in results_before:
                metrics_before = results_before[split].compute_aggregate_metrics()
                user_metrics_before = results_before[split].get_user_metrics_list()
                save_metrics(
                    metrics_before,
                    user_metrics_before,
                    str(save_dir / f'{split}_before.json')
                )
            
            if split in results_after:
                metrics_after = results_after[split].compute_aggregate_metrics()
                user_metrics_after = results_after[split].get_user_metrics_list()
                save_metrics(
                    metrics_after,
                    user_metrics_after,
                    str(save_dir / f'{split}_after.json')
                )
    
    return {
        'before': results_before,
        'after': results_after
    }


# Example usage
if __name__ == "__main__":
    # Configuration
    DATA_DIR = "E:\\Kuliah\\Kuliah\\Kuliah\\PRODI\\Semester 7\\ProSkripCode\\data_movie"
    MODEL_PATH_BEFORE = "checkpoints/dqn_before_unlearning.pt"
    MODEL_PATH_AFTER = "checkpoints/dqn_after_unlearning.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("\nLoading data...")
    dataloaders, dataset = create_dataloaders(
        data_dir=DATA_DIR,
        forget_ratio=0.1,
        use_genome=False,  # Set to True if using genome features
        state_size=50,
        batch_size=64
    )
    
    # Get dimensions for model
    state_dim = dataset.get_state_dim()
    action_dim = dataset.get_action_dim()
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    
    # Load models (you'll implement DQN model)
    # from model.dqn_model import DQN
    # model_before = DQN(state_dim, action_dim)
    # model_before.load_state_dict(torch.load(MODEL_PATH_BEFORE))
    # 
    # model_after = DQN(state_dim, action_dim)
    # model_after.load_state_dict(torch.load(MODEL_PATH_AFTER))
    
    # For now, using dummy models for testing
    print("\n[NOTE: Replace with actual trained models]")
    
    # Evaluate single model
    # evaluator = DQNEvaluator(model_before, device)
    # results = evaluator.evaluate_split(dataloaders, dataset)
    
    # Full unlearning evaluation
    # results = evaluate_unlearning(
    #     model_before,
    #     model_after,
    #     dataloaders,
    #     dataset,
    #     device=device,
    #     save_dir='results/unlearning_eval'
    # )
