# model/loss_function.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DQNLoss(nn.Module):
    """
    Standard DQN loss using TD learning with target network.
    
    Loss = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))²]
    """
    
    def __init__(self, gamma: float = 0.99):
        """
        Args:
            gamma: Discount factor for future rewards
        """
        super(DQNLoss, self).__init__()
        self.gamma = gamma
    
    def forward(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute DQN loss for a batch of transitions.
        
        Args:
            q_network: Online Q-network
            target_network: Target Q-network (frozen)
            states: Current states, shape (batch_size, state_dim)
            actions: Actions taken, shape (batch_size,) or (batch_size, 1)
            rewards: Rewards received, shape (batch_size,) or (batch_size, 1)
            next_states: Next states, shape (batch_size, state_dim)
            dones: Terminal flags, shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Loss scalar
        """
        # Ensure correct shapes
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Current Q-values: Q(s, a)
        current_q_values = q_network(states).gather(1, actions)
        
        # Target Q-values: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = target_network(next_states).max(dim=1, keepdim=True)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # MSE loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        return loss


class DoubleDQNLoss(nn.Module):
    """
    Double DQN loss to reduce overestimation bias.
    
    Uses online network for action selection and target network for evaluation.
    Loss = E[(r + γ * Q_target(s', argmax_a' Q_online(s', a')) - Q(s, a))²]
    """
    
    def __init__(self, gamma: float = 0.99):
        super(DoubleDQNLoss, self).__init__()
        self.gamma = gamma
    
    def forward(
        self,
        q_network: nn.Module,
        target_network: nn.Module,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute Double DQN loss"""
        # Ensure correct shapes
        if actions.dim() == 1:
            actions = actions.unsqueeze(1)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # Current Q-values
        current_q_values = q_network(states).gather(1, actions)
        
        # Double DQN: select action with online network, evaluate with target
        with torch.no_grad():
            # Select best action using online network
            next_actions = q_network(next_states).argmax(dim=1, keepdim=True)
            # Evaluate that action using target network
            next_q_values = target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # MSE loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        return loss


class DecrementalRLLoss(nn.Module):
    """
    Decremental RL-based unlearning loss from the paper.
    
    L_u = E_(s,a)~τ_u [Q_π'(s,a)] + λ * E_(s,a)~τ_r |Q_π'(s,a) - Q_π(s,a)|
    
    Term 1: Minimize Q-values on forget set (make agent "forget")
    Term 2: Keep Q-values close to original on retain set (preserve performance)
    """
    
    def __init__(self, lambda_weight: float = 1.0):
        """
        Args:
            lambda_weight: Weight for retain set regularization term
        """
        super(DecrementalRLLoss, self).__init__()
        self.lambda_weight = lambda_weight
    
    def forward(
        self,
        q_network: nn.Module,
        original_q_network: nn.Module,
        forget_states: torch.Tensor,
        forget_actions: torch.Tensor,
        retain_states: torch.Tensor,
        retain_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute decremental RL unlearning loss.
        
        Args:
            q_network: Current Q-network being updated (Q_π')
            original_q_network: Original frozen Q-network (Q_π)
            forget_states: States from forget users, shape (batch_forget, state_dim)
            forget_actions: Actions from forget users, shape (batch_forget,)
            retain_states: States from retain users, shape (batch_retain, state_dim)
            retain_actions: Actions from retain users, shape (batch_retain,)
            
        Returns:
            tuple: (total_loss, loss_dict)
                - total_loss: Combined loss scalar
                - loss_dict: Dictionary with individual loss components
        """
        # Ensure correct shapes
        if forget_actions.dim() == 1:
            forget_actions = forget_actions.unsqueeze(1)
        if retain_actions.dim() == 1:
            retain_actions = retain_actions.unsqueeze(1)
        
        # Term 1: Minimize Q-values on forget set
        # E_(s,a)~τ_u [Q_π'(s,a)]
        q_forget = q_network(forget_states).gather(1, forget_actions)
        forget_loss = q_forget.mean()
        
        # Term 2: Absolute difference on retain set
        # E_(s,a)~τ_r |Q_π'(s,a) - Q_π(s,a)|
        with torch.no_grad():
            q_retain_original = original_q_network(retain_states).gather(1, retain_actions)
        
        q_retain_new = q_network(retain_states).gather(1, retain_actions)
        retain_loss = torch.abs(q_retain_new - q_retain_original).mean()
        
        # Combined loss
        total_loss = forget_loss + self.lambda_weight * retain_loss
        
        # Return loss components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'forget_loss': forget_loss.item(),
            'retain_loss': retain_loss.item(),
            'weighted_retain_loss': (self.lambda_weight * retain_loss).item()
        }
        
        return total_loss, loss_dict


class PoisoningBasedLoss(nn.Module):
    """
    Poisoning-based unlearning loss (alternative method from the paper).
    
    This is mentioned in the paper but not the primary focus.
    Includes policy divergence term.
    """
    
    def __init__(
        self,
        lambda1: float = 1.0,
        lambda2: float = 1.0
    ):
        """
        Args:
            lambda1: Weight for policy divergence term
            lambda2: Weight for performance preservation term
        """
        super(PoisoningBasedLoss, self).__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
    
    def forward(
        self,
        q_network: nn.Module,
        original_q_network: nn.Module,
        forget_states: torch.Tensor,
        retain_states: torch.Tensor,
        retain_actions: torch.Tensor,
        retain_rewards: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute poisoning-based unlearning loss.
        
        R_i = λ1 * Δ(π_i(s_i) || π_{i+1}(s_i)) + λ2 * Σ π_i(s,a)r(s,a)
        """
        # Policy divergence on forget states
        with torch.no_grad():
            old_policy = F.softmax(original_q_network(forget_states), dim=1)
        new_policy = F.softmax(q_network(forget_states), dim=1)
        
        # KL divergence
        divergence = F.kl_div(
            new_policy.log(),
            old_policy,
            reduction='batchmean'
        )
        
        # Performance preservation on retain set
        if retain_actions.dim() == 1:
            retain_actions = retain_actions.unsqueeze(1)
        if retain_rewards.dim() == 1:
            retain_rewards = retain_rewards.unsqueeze(1)
        
        q_retain = q_network(retain_states).gather(1, retain_actions)
        performance_loss = -torch.mean(q_retain * retain_rewards)
        
        # Combined loss
        total_loss = self.lambda1 * divergence + self.lambda2 * performance_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'divergence_loss': divergence.item(),
            'performance_loss': performance_loss.item()
        }
        
        return total_loss, loss_dict


# Example usage
if __name__ == "__main__":
    print("Testing loss functions...")
    
    # Create dummy data
    batch_size = 32
    state_dim = 100
    action_dim = 1000
    
    # Create networks
    from dqn_model import DQN
    q_network = DQN(state_dim, action_dim)
    target_network = DQN(state_dim, action_dim)
    target_network.copy_weights_from(q_network)
    target_network.eval()
    
    # Test standard DQN loss
    print("\n1. Testing Standard DQN Loss...")
    dqn_loss_fn = DQNLoss(gamma=0.99)
    
    states = torch.randn(batch_size, state_dim)
    actions = torch.randint(0, action_dim, (batch_size,))
    rewards = torch.rand(batch_size) * 5.0  # Ratings 0-5
    next_states = torch.randn(batch_size, state_dim)
    dones = torch.zeros(batch_size)
    
    loss = dqn_loss_fn(q_network, target_network, states, actions, rewards, next_states, dones)
    print(f"DQN Loss: {loss.item():.4f}")
    
    # Test Double DQN loss
    print("\n2. Testing Double DQN Loss...")
    double_dqn_loss_fn = DoubleDQNLoss(gamma=0.99)
    loss = double_dqn_loss_fn(q_network, target_network, states, actions, rewards, next_states, dones)
    print(f"Double DQN Loss: {loss.item():.4f}")
    
    # Test Decremental RL loss
    print("\n3. Testing Decremental RL Loss...")
    original_q_network = DQN(state_dim, action_dim)
    original_q_network.copy_weights_from(q_network)
    original_q_network.eval()
    
    unlearning_loss_fn = DecrementalRLLoss(lambda_weight=1.0)
    
    forget_states = torch.randn(16, state_dim)
    forget_actions = torch.randint(0, action_dim, (16,))
    retain_states = torch.randn(16, state_dim)
    retain_actions = torch.randint(0, action_dim, (16,))
    
    loss, loss_dict = unlearning_loss_fn(
        q_network, original_q_network,
        forget_states, forget_actions,
        retain_states, retain_actions
    )
    print(f"Unlearning Loss: {loss.item():.4f}")
    print(f"Loss components: {loss_dict}")
    
    print("\nAll tests passed!")
