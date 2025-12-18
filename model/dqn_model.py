# model/dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DQN(nn.Module):
    """
    Deep Q-Network for movie recommendation.
    
    Architecture:
        Input (state_dim) → 512 → 256 → 128 → Output (action_dim)
        With ReLU activations and optional dropout.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.2,
        use_dropout: bool = True
    ):
        """
        Args:
            state_dim: Dimension of state vector (e.g., 100 without genome, 56500 with genome)
            action_dim: Number of possible actions (number of movies)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_dropout: Whether to use dropout layers
        """
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_dropout = use_dropout
        
        # Build network layers
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
            input_dim = hidden_dim
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            Q-values tensor of shape (batch_size, action_dim)
        """
        return self.network(state)
    
    def get_q_value(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get Q-value for specific state-action pairs.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            action: Action tensor of shape (batch_size,) containing action indices
            
        Returns:
            Q-values tensor of shape (batch_size, 1)
        """
        q_values = self.forward(state)
        return q_values.gather(1, action.unsqueeze(1))
    
    def get_max_q_value(self, state: torch.Tensor) -> tuple:
        """
        Get maximum Q-value and corresponding action for each state.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            
        Returns:
            tuple: (max_q_values, best_actions)
                - max_q_values: shape (batch_size, 1)
                - best_actions: shape (batch_size, 1)
        """
        q_values = self.forward(state)
        max_q_values, best_actions = q_values.max(dim=1, keepdim=True)
        return max_q_values, best_actions
    
    def select_action(
        self,
        state: torch.Tensor,
        epsilon: float = 0.0,
        valid_actions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
            epsilon: Exploration probability (0.0 = pure exploitation)
            valid_actions: Boolean mask of shape (batch_size, action_dim) 
                          indicating valid actions (True = valid)
            
        Returns:
            Selected actions of shape (batch_size,)
        """
        batch_size = state.shape[0]
        
        # Epsilon-greedy selection
        if torch.rand(1).item() < epsilon:
            # Random action
            if valid_actions is not None:
                # Sample from valid actions only
                actions = []
                for i in range(batch_size):
                    valid_idx = torch.where(valid_actions[i])[0]
                    random_action = valid_idx[torch.randint(len(valid_idx), (1,))]
                    actions.append(random_action)
                return torch.stack(actions).squeeze()
            else:
                # Sample from all actions
                return torch.randint(0, self.action_dim, (batch_size,))
        else:
            # Greedy action (max Q-value)
            q_values = self.forward(state)
            
            if valid_actions is not None:
                # Mask invalid actions with -inf
                masked_q = q_values.clone()
                masked_q[~valid_actions] = float('-inf')
                return masked_q.argmax(dim=1)
            else:
                return q_values.argmax(dim=1)
    
    def copy_weights_from(self, source_network: 'DQN'):
        """
        Copy weights from another DQN network (for target network updates).
        
        Args:
            source_network: Source DQN to copy weights from
        """
        self.load_state_dict(source_network.state_dict())
    
    def soft_update(self, source_network: 'DQN', tau: float = 0.005):
        """
        Soft update of target network parameters using Polyak averaging.
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            source_network: Source network to update from
            tau: Interpolation parameter (0 = no update, 1 = hard update)
        """
        for target_param, source_param in zip(self.parameters(), source_network.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture (optional enhancement).
    Separates value and advantage streams.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [512, 256],
        dropout_rate: float = 0.2
    ):
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dims[1], 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using dueling architecture.
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        """
        features = self.feature_layers(state)
        
        value = self.value_stream(features)  # (batch, 1)
        advantage = self.advantage_stream(features)  # (batch, action_dim)
        
        # Combine using mean subtraction
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


# Utility function
def create_dqn(
    state_dim: int,
    action_dim: int,
    architecture: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create DQN models.
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        architecture: 'standard' or 'dueling'
        **kwargs: Additional arguments for the model
        
    Returns:
        DQN model
    """
    if architecture == 'standard':
        return DQN(state_dim, action_dim, **kwargs)
    elif architecture == 'dueling':
        return DuelingDQN(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


if __name__ == "__main__":
    # Test the model
    print("Testing DQN model...")
    
    state_dim = 100  # Without genome
    action_dim = 9742  # Example number of movies
    batch_size = 32
    
    # Create model
    model = DQN(state_dim, action_dim)
    print(f"Model created: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_state = torch.randn(batch_size, state_dim)
    q_values = model(dummy_state)
    print(f"Q-values shape: {q_values.shape}")  # Should be (32, 9742)
    
    # Test action selection
    actions = model.select_action(dummy_state, epsilon=0.1)
    print(f"Selected actions shape: {actions.shape}")  # Should be (32,)
    
    # Test with valid action mask
    valid_mask = torch.rand(batch_size, action_dim) > 0.9  # 10% valid actions
    masked_actions = model.select_action(dummy_state, epsilon=0.0, valid_actions=valid_mask)
    print(f"Masked actions shape: {masked_actions.shape}")
    
    print("\nAll tests passed!")
