# model/dqn_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class RecommendationDQN(nn.Module):
    """
    Deep Q-Network for binary movie recommendation.
    
    Architecture:
        Input(44) → Hidden(128) → Hidden(128) → Output(2)
        [Q(don't recommend), Q(recommend)]
    """
    
    def __init__(
        self,
        input_dim: int = 44,
        hidden_dims: list = [128, 128],
        dropout_rate: float = 0.2,
        output_dim: int = 2
    ):
        """
        Args:
            input_dim: Input feature dimension (22 user + 22 candidate = 44)
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            output_dim: Number of actions (2: don't rec, rec)
        """
        super(RecommendationDQN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # Build network layers
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim
        
        # Output layer (no activation - raw Q-values)
        layers.append(nn.Linear(current_dim, output_dim))
        
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Should be [user_state(22), candidate_features(22)]
            
        Returns:
            Q-values tensor of shape (batch_size, 2)
            [Q(don't recommend), Q(recommend)]
        """
        return self.network(x)
    
    def get_q_values(
        self, 
        user_state: torch.Tensor, 
        candidate_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get Q-values for user-candidate pairs.
        
        Args:
            user_state: (batch_size, 22)
            candidate_features: (batch_size, 22)
            
        Returns:
            Q-values: (batch_size, 2)
        """
        x = torch.cat([user_state, candidate_features], dim=1)
        return self.forward(x)
    
    def select_action(
        self, 
        user_state: torch.Tensor, 
        candidate_features: torch.Tensor,
        epsilon: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            user_state: (batch_size, 22)
            candidate_features: (batch_size, 22)
            epsilon: Exploration probability
            
        Returns:
            actions: (batch_size,) - Selected actions (0 or 1)
            q_values: (batch_size, 2) - Q-values for logging
        """
        batch_size = user_state.shape[0]
        
        # Get Q-values
        q_values = self.get_q_values(user_state, candidate_features)
        
        # Epsilon-greedy selection
        if torch.rand(1).item() < epsilon:
            # Random action
            actions = torch.randint(0, 2, (batch_size,), device=user_state.device)
        else:
            # Greedy action (max Q-value)
            actions = q_values.argmax(dim=1)
        
        return actions, q_values
    
    def predict_recommend(
        self,
        user_state: torch.Tensor,
        candidate_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict whether to recommend (binary classification).
        
        Args:
            user_state: (batch_size, 22)
            candidate_features: (batch_size, 22)
            
        Returns:
            predictions: (batch_size,) - 0 or 1
        """
        q_values = self.get_q_values(user_state, candidate_features)
        return q_values.argmax(dim=1)
    
    def get_q_recommend(
        self,
        user_state: torch.Tensor,
        candidate_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Get Q-value for "recommend" action only (for ranking).
        
        Args:
            user_state: (batch_size, 22)
            candidate_features: (batch_size, 22)
            
        Returns:
            Q(recommend): (batch_size,)
        """
        q_values = self.get_q_values(user_state, candidate_features)
        return q_values[:, 1]  # Return Q(recommend) only
    
    def copy_weights_from(self, source_network: 'RecommendationDQN'):
        """
        Copy weights from another network (for target network hard update).
        
        Args:
            source_network: Source DQN to copy weights from
        """
        self.load_state_dict(source_network.state_dict())
    
    def soft_update(self, source_network: 'RecommendationDQN', tau: float = 0.005):
        """
        Soft update of network parameters using Polyak averaging.
        θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            source_network: Source network to update from
            tau: Interpolation parameter (0 = no update, 1 = hard update)
        """
        for target_param, source_param in zip(self.parameters(), source_network.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )


class DQNWithTargetNetwork:
    """
    Wrapper class that manages both online and target networks.
    """
    
    def __init__(
        self,
        input_dim: int = 44,
        hidden_dims: list = [128, 128],
        dropout_rate: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        target_update_freq: int = 1000,
        use_soft_update: bool = False,
        tau: float = 0.005
    ):
        """
        Args:
            input_dim: Input dimension
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
            device: Device to use
            target_update_freq: Steps between target network updates (hard update)
            use_soft_update: Use soft update instead of hard update
            tau: Soft update interpolation parameter
        """
        self.device = device
        self.target_update_freq = target_update_freq
        self.use_soft_update = use_soft_update
        self.tau = tau
        self.step_count = 0
        
        # Create online and target networks
        self.online_network = RecommendationDQN(
            input_dim, hidden_dims, dropout_rate
        ).to(device)
        
        self.target_network = RecommendationDQN(
            input_dim, hidden_dims, dropout_rate
        ).to(device)
        
        # Initialize target network
        self.target_network.copy_weights_from(self.online_network)
        self.target_network.eval()
        
        print(f"DQN with Target Network initialized on {device}")
        print(f"Parameters: {sum(p.numel() for p in self.online_network.parameters()):,}")
    
    def update_target_network(self):
        """Update target network based on configuration"""
        if self.use_soft_update:
            # Soft update every step
            self.target_network.soft_update(self.online_network, self.tau)
        else:
            # Hard update at specified frequency
            self.step_count += 1
            if self.step_count % self.target_update_freq == 0:
                self.target_network.copy_weights_from(self.online_network)
                print(f"Target network updated at step {self.step_count}")
    
    def get_online_network(self) -> RecommendationDQN:
        """Get online network for training"""
        return self.online_network
    
    def get_target_network(self) -> RecommendationDQN:
        """Get target network for computing targets"""
        return self.target_network
    
    def train_mode(self):
        """Set online network to training mode"""
        self.online_network.train()
    
    def eval_mode(self):
        """Set online network to evaluation mode"""
        self.online_network.eval()
    
    def save(self, path: str):
        """Save both networks"""
        torch.save({
            'online_network': self.online_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'step_count': self.step_count
        }, path)
        print(f"Models saved to {path}")
    
    def load(self, path: str):
        """Load both networks"""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_network.load_state_dict(checkpoint['online_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Models loaded from {path}")


# Test code
if __name__ == "__main__":
    print("Testing RecommendationDQN...")
    
    # Test basic model
    model = RecommendationDQN(input_dim=44, hidden_dims=[128, 128])
    print(f"\nModel architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 32
    user_state = torch.randn(batch_size, 22)
    candidate = torch.randn(batch_size, 22)
    input_vec = torch.cat([user_state, candidate], dim=1)
    
    q_values = model(input_vec)
    print(f"\nInput shape: {input_vec.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Sample Q-values: {q_values[0]}")
    
    # Test get_q_values
    q_vals = model.get_q_values(user_state, candidate)
    print(f"\nget_q_values output: {q_vals.shape}")
    
    # Test action selection
    actions, q_vals = model.select_action(user_state, candidate, epsilon=0.1)
    print(f"\nSelected actions shape: {actions.shape}")
    print(f"First 5 actions: {actions[:5]}")
    
    # Test predict_recommend
    predictions = model.predict_recommend(user_state, candidate)
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Prediction distribution: {predictions.float().mean().item():.2%} recommend")
    
    # Test get_q_recommend
    q_rec = model.get_q_recommend(user_state, candidate)
    print(f"\nQ(recommend) shape: {q_rec.shape}")
    print(f"Q(recommend) range: [{q_rec.min():.3f}, {q_rec.max():.3f}]")
    
    # Test DQN with target network
    print("\n" + "="*70)
    print("Testing DQNWithTargetNetwork...")
    print("="*70)
    
    dqn_manager = DQNWithTargetNetwork(
        input_dim=44,
        hidden_dims=[128, 128],
        device='cpu',
        target_update_freq=1000
    )
    
    online = dqn_manager.get_online_network()
    target = dqn_manager.get_target_network()
    
    # Check they're initially the same
    online_output = online(input_vec)
    target_output = target(input_vec)
    diff = (online_output - target_output).abs().max()
    print(f"\nInitial difference between networks: {diff.item():.6f}")
    
    # Simulate training step
    dqn_manager.train_mode()
    online.zero_grad()
    loss = online_output.mean()
    loss.backward()
    
    # Update target
    dqn_manager.update_target_network()
    
    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        temp_path = f.name
    
    dqn_manager.save(temp_path)
    dqn_manager.load(temp_path)
    print(f"\nSave/Load test passed!")
    
    import os
    os.unlink(temp_path)
    
    print("\n✅ All tests passed!")
