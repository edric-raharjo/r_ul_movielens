# model/loss_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DQNRecommendationLoss(nn.Module):
    """
    DQN loss for binary movie recommendation with scaled inverse rewards.
    
    Q-value targets:
        Positive (rating >= 4.0): Q_target = [0.0, rating]
        Negative (rating < 4.0):  Q_target = [2*(5.0-rating), rating]
    """
    
    def __init__(
        self,
        rating_threshold: float = 4.0,
        max_rating: float = 5.0,
        negative_boost: float = 1.0
    ):
        """
        Args:
            rating_threshold: Ratings >= this are positive samples
            max_rating: Maximum possible rating
            negative_boost: Multiplier for negative sample inverse reward
        """
        super(DQNRecommendationLoss, self).__init__()
        self.rating_threshold = rating_threshold
        self.max_rating = max_rating
        self.negative_boost = negative_boost
    
    def compute_q_targets(
        self,
        ratings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-value targets based on ratings and labels.
        
        Args:
            ratings: (batch_size,) actual ratings [0.5-5.0]
            labels: (batch_size,) binary labels [0 or 1]
            
        Returns:
            q_targets: (batch_size, 2) targets for [Q(don't), Q(rec)]
        """
        batch_size = ratings.shape[0]
        q_targets = torch.zeros(batch_size, 2, device=ratings.device)
        
        # Positive samples (should recommend)
        positive_mask = labels == 1
        q_targets[positive_mask, 0] = 0.0  # Q(don't rec) = 0
        q_targets[positive_mask, 1] = ratings[positive_mask]  # Q(rec) = rating
        
        # Negative samples (should NOT recommend)
        negative_mask = labels == 0
        q_targets[negative_mask, 0] = self.negative_boost * (
            self.max_rating - ratings[negative_mask]
        )  # Q(don't rec) = 2*(5-rating)
        q_targets[negative_mask, 1] = ratings[negative_mask]  # Q(rec) = rating
        
        return q_targets
    
    def forward(
        self,
        q_values: torch.Tensor,
        ratings: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute DQN loss.
        
        Args:
            q_values: (batch_size, 2) predicted Q-values
            ratings: (batch_size,) actual ratings
            labels: (batch_size,) binary labels (0 or 1)
            
        Returns:
            loss: Scalar loss value
            loss_dict: Dictionary with loss components
        """
        # Compute targets
        q_targets = self.compute_q_targets(ratings, labels)
        
        # MSE loss
        loss = F.mse_loss(q_values, q_targets)
        
        # Separate losses for logging
        with torch.no_grad():
            # Create masks again for logging (FIX: define them here)
            positive_mask = labels == 1
            negative_mask = labels == 0
            
            # Loss for each action
            loss_q0 = F.mse_loss(q_values[:, 0], q_targets[:, 0])
            loss_q1 = F.mse_loss(q_values[:, 1], q_targets[:, 1])
            
            # Loss for positive vs negative samples
            if positive_mask.any():
                loss_positive = F.mse_loss(
                    q_values[positive_mask], 
                    q_targets[positive_mask]
                )
            else:
                loss_positive = torch.tensor(0.0)
            
            if negative_mask.any():
                loss_negative = F.mse_loss(
                    q_values[negative_mask],
                    q_targets[negative_mask]
                )
            else:
                loss_negative = torch.tensor(0.0)
            
            # Accuracy (for monitoring)
            predictions = q_values.argmax(dim=1)
            accuracy = (predictions == labels).float().mean()
        
        loss_dict = {
            'total_loss': loss.item(),
            'loss_q_dont': loss_q0.item(),
            'loss_q_rec': loss_q1.item(),
            'loss_positive': loss_positive.item(),
            'loss_negative': loss_negative.item(),
            'accuracy': accuracy.item()
        }
        
        return loss, loss_dict


class DecrementalRLLoss(nn.Module):
    """
    Decremental RL loss for unlearning.
    
    L_u = E[(s,a)~τ_u] [Q_π'(s,a)] + λ * E[(s,a)~τ_r] |Q_π'(s,a) - Q_π(s,a)|
    
    Term 1: Minimize Q-values on forget set (random policy)
    Term 2: Preserve Q-values on retain set
    """
    
    def __init__(self, lambda_weight: float = 1.0):
        """
        Args:
            lambda_weight: Weight for retain set preservation term
        """
        super(DecrementalRLLoss, self).__init__()
        self.lambda_weight = lambda_weight
    
    def forward(
        self,
        q_new_forget: torch.Tensor,
        forget_actions: torch.Tensor,
        q_new_retain: torch.Tensor,
        q_original_retain: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute decremental RL unlearning loss.
        
        Args:
            q_new_forget: (batch_forget, 2) Q-values from new network on forget set
            forget_actions: (batch_forget,) random actions (0 or 1) for forget samples
            q_new_retain: (batch_retain, 2) Q-values from new network on retain set
            q_original_retain: (batch_retain, 2) Q-values from original network on retain set
            
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary with loss components
        """
        # Term 1: Minimize Q-values for random actions on forget set
        # Gather Q-values for the randomly selected actions
        forget_q_values = q_new_forget.gather(
            1, forget_actions.unsqueeze(1)
        ).squeeze(1)
        forget_loss = forget_q_values.mean()
        
        # Term 2: Absolute difference on retain set (both Q-values)
        retain_loss = torch.abs(q_new_retain - q_original_retain).mean()
        
        # Combined loss
        total_loss = forget_loss + self.lambda_weight * retain_loss
        
        # Additional metrics
        with torch.no_grad():
            # Average Q-values
            avg_q_forget = q_new_forget.mean().item()
            avg_q_retain_new = q_new_retain.mean().item()
            avg_q_retain_original = q_original_retain.mean().item()
            
            # Q-value drift on retain set
            q_drift = (q_new_retain - q_original_retain).abs().mean().item()
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'forget_loss': forget_loss.item(),
            'retain_loss': retain_loss.item(),
            'weighted_retain_loss': (self.lambda_weight * retain_loss).item(),
            'avg_q_forget': avg_q_forget,
            'avg_q_retain_new': avg_q_retain_new,
            'avg_q_retain_original': avg_q_retain_original,
            'q_drift': q_drift
        }
        
        return total_loss, loss_dict


class DecrementalRLLossSimplified(nn.Module):
    """
    Simplified decremental RL loss (alternative).
    
    Term 1: Minimize Q(recommend) only on forget set
    Term 2: Preserve both Q-values on retain set
    """
    
    def __init__(self, lambda_weight: float = 1.0):
        super(DecrementalRLLossSimplified, self).__init__()
        self.lambda_weight = lambda_weight
    
    def forward(
        self,
        q_new_forget: torch.Tensor,
        q_new_retain: torch.Tensor,
        q_original_retain: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Simplified version: minimize Q(recommend) on forget set.
        
        Args:
            q_new_forget: (batch_forget, 2) Q-values on forget set
            q_new_retain: (batch_retain, 2) Q-values on retain set (new)
            q_original_retain: (batch_retain, 2) Q-values on retain set (original)
        """
        # Term 1: Minimize Q(recommend) on forget set
        forget_loss = q_new_forget[:, 1].mean()  # Only Q(rec)
        
        # Term 2: Preserve Q-values on retain set
        retain_loss = torch.abs(q_new_retain - q_original_retain).mean()
        
        # Combined
        total_loss = forget_loss + self.lambda_weight * retain_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'forget_loss': forget_loss.item(),
            'retain_loss': retain_loss.item(),
            'weighted_retain_loss': (self.lambda_weight * retain_loss).item()
        }
        
        return total_loss, loss_dict


def create_random_forget_samples(
    forget_dataset,
    forget_users: set,
    num_samples_per_user: int = 50,
    device: str = 'cpu'
) -> Dict[str, torch.Tensor]:
    """
    Create random action samples for forget users (for unlearning).
    
    Args:
        forget_dataset: Dataset containing forget user samples
        forget_users: Set of forget user IDs
        num_samples_per_user: Number of random samples per user
        device: Device to put tensors on
        
    Returns:
        Dictionary with:
        - 'inputs': (N, 44) concatenated user+candidate features
        - 'actions': (N,) random actions (0 or 1)
        - 'user_ids': (N,) user IDs
        - 'movie_ids': (N,) movie IDs
    """
    import numpy as np
    
    random_samples = {
        'inputs': [],
        'actions': [],
        'user_ids': [],
        'movie_ids': []
    }
    
    # Group samples by user
    user_samples = {}
    for sample in forget_dataset.samples:
        user_id = sample['user_id']
        if user_id in forget_users:
            if user_id not in user_samples:
                user_samples[user_id] = []
            user_samples[user_id].append(sample)
    
    # Sample randomly from each user
    for user_id, samples in user_samples.items():
        # Sample with replacement if needed
        n_samples = min(num_samples_per_user, len(samples))
        sampled_indices = np.random.choice(
            len(samples), 
            size=n_samples, 
            replace=(n_samples > len(samples))
        )
        
        for idx in sampled_indices:
            sample = samples[idx]
            
            # Concatenate user state + candidate
            input_vec = torch.cat([
                torch.FloatTensor(sample['user_state']),
                torch.FloatTensor(sample['candidate_features'])
            ])
            
            # Random action: 0 or 1
            random_action = np.random.randint(0, 2)
            
            random_samples['inputs'].append(input_vec)
            random_samples['actions'].append(random_action)
            random_samples['user_ids'].append(user_id)
            random_samples['movie_ids'].append(sample['movie_id'])
    
    # Convert to tensors
    if len(random_samples['inputs']) > 0:
        random_samples['inputs'] = torch.stack(random_samples['inputs']).to(device)
        random_samples['actions'] = torch.LongTensor(random_samples['actions']).to(device)
        random_samples['user_ids'] = torch.LongTensor(random_samples['user_ids']).to(device)
        random_samples['movie_ids'] = torch.LongTensor(random_samples['movie_ids']).to(device)
    
    print(f"Created {len(random_samples['inputs'])} random samples from {len(user_samples)} forget users")
    
    return random_samples


# Test code
if __name__ == "__main__":
    print("Testing Loss Functions...")
    
    batch_size = 32
    
    # Test DQN Training Loss
    print("\n" + "="*70)
    print("Testing DQNRecommendationLoss")
    print("="*70)
    
    loss_fn = DQNRecommendationLoss(rating_threshold=4.0)
    
    # Create dummy data
    q_values = torch.randn(batch_size, 2)
    ratings = torch.rand(batch_size) * 4.5 + 0.5  # [0.5, 5.0]
    labels = (ratings >= 4.0).long()
    
    print(f"\nBatch size: {batch_size}")
    print(f"Ratings: min={ratings.min():.2f}, max={ratings.max():.2f}")
    print(f"Labels: {labels.sum().item()} positive, {(batch_size - labels.sum()).item()} negative")
    
    # Compute Q-targets
    q_targets = loss_fn.compute_q_targets(ratings, labels)
    print(f"\nQ-targets shape: {q_targets.shape}")
    print(f"Sample Q-target (positive, rating=5.0): {q_targets[labels==1][0] if (labels==1).any() else 'N/A'}")
    print(f"Sample Q-target (negative, rating=2.0): {q_targets[labels==0][0] if (labels==0).any() else 'N/A'}")
    
    # Compute loss
    loss, loss_dict = loss_fn(q_values, ratings, labels)
    print(f"\nLoss: {loss.item():.4f}")
    print("Loss components:")
    for key, val in loss_dict.items():
        print(f"  {key}: {val:.4f}")
    
    # Test Decremental RL Loss
    print("\n" + "="*70)
    print("Testing DecrementalRLLoss")
    print("="*70)
    
    unlearning_loss_fn = DecrementalRLLoss(lambda_weight=1.0)
    
    # Forget set
    batch_forget = 16
    q_new_forget = torch.randn(batch_forget, 2)
    forget_actions = torch.randint(0, 2, (batch_forget,))
    
    # Retain set
    batch_retain = 32
    q_new_retain = torch.randn(batch_retain, 2)
    q_original_retain = torch.randn(batch_retain, 2)
    
    print(f"\nForget batch: {batch_forget}")
    print(f"Retain batch: {batch_retain}")
    print(f"Random actions: {forget_actions[:5]}")
    
    # Compute loss
    unlearn_loss, unlearn_dict = unlearning_loss_fn(
        q_new_forget, forget_actions,
        q_new_retain, q_original_retain
    )
    
    print(f"\nUnlearning loss: {unlearn_loss.item():.4f}")
    print("Loss components:")
    for key, val in unlearn_dict.items():
        print(f"  {key}: {val:.4f}")
    
    # Test Simplified Loss
    print("\n" + "="*70)
    print("Testing DecrementalRLLossSimplified")
    print("="*70)
    
    simplified_loss_fn = DecrementalRLLossSimplified(lambda_weight=1.0)
    
    simplified_loss, simplified_dict = simplified_loss_fn(
        q_new_forget, q_new_retain, q_original_retain
    )
    
    print(f"\nSimplified loss: {simplified_loss.item():.4f}")
    print("Loss components:")
    for key, val in simplified_dict.items():
        print(f"  {key}: {val:.4f}")
    
    # Test backward pass
    print("\n" + "="*70)
    print("Testing Backward Pass")
    print("="*70)
    
    # Training loss
    q_values_grad = torch.randn(batch_size, 2, requires_grad=True)
    loss, _ = loss_fn(q_values_grad, ratings, labels)
    loss.backward()
    print(f"Training loss backward: ✓ (grad norm: {q_values_grad.grad.norm().item():.4f})")
    
    # Unlearning loss
    q_new_forget_grad = torch.randn(batch_forget, 2, requires_grad=True)
    q_new_retain_grad = torch.randn(batch_retain, 2, requires_grad=True)
    unlearn_loss, _ = unlearning_loss_fn(
        q_new_forget_grad, forget_actions,
        q_new_retain_grad, q_original_retain
    )
    unlearn_loss.backward()
    print(f"Unlearning loss backward: ✓")
    print(f"  Forget grad norm: {q_new_forget_grad.grad.norm().item():.4f}")
    print(f"  Retain grad norm: {q_new_retain_grad.grad.norm().item():.4f}")
    
    print("\n✅ All loss function tests passed!")
