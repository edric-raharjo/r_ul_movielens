# train/train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
from tqdm import tqdm
import json
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from model.dqn_model import DQN, create_dqn
from model.loss_function import DoubleDQNLoss, DecrementalRLLoss
from dataset.dataloader import MovieLensRLDataset, create_dataloaders
from eval.evaluate import DQNEvaluator, evaluate_unlearning
from eval.eval_metrics import RLRecommendationMetrics, compare_metrics


class DQNTrainer:
    """
    Trainer for DQN-based recommendation system with unlearning support.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        architecture: str = 'standard',
        hidden_dims: list = [512, 256, 128],
        dropout_rate: float = 0.2
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension (number of movies)
            device: Device to train on
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            target_update_freq: Steps between target network updates
            architecture: 'standard' or 'dueling'
            hidden_dims: Hidden layer dimensions
            dropout_rate: Dropout rate
        """
        self.device = device
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.step_count = 0
        
        # Create networks
        self.q_network = create_dqn(
            state_dim, action_dim, architecture,
            hidden_dims=hidden_dims, dropout_rate=dropout_rate
        ).to(device)
        
        self.target_network = create_dqn(
            state_dim, action_dim, architecture,
            hidden_dims=hidden_dims, dropout_rate=dropout_rate
        ).to(device)
        
        # Initialize target network
        self.target_network.copy_weights_from(self.q_network)
        self.target_network.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = DoubleDQNLoss(gamma=gamma)
        
        print(f"DQN Trainer initialized on {device}")
        print(f"Q-Network parameters: {sum(p.numel() for p in self.q_network.parameters()):,}")
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        desc: str = "Training"
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.q_network.train()
        epoch_losses = []
        
        pbar = tqdm(dataloader, desc=f"{desc} Epoch {epoch+1}")
        for batch in pbar:
            # Move batch to device
            states = batch['state'].to(self.device)
            actions = batch['action'].to(self.device)
            rewards = batch['reward'].to(self.device)
            next_states = batch['next_state'].to(self.device)
            dones = batch['done'].to(self.device)
            
            # Compute loss
            loss = self.criterion(
                self.q_network,
                self.target_network,
                states, actions, rewards, next_states, dones
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            epoch_losses.append(loss.item())
            self.step_count += 1
            
            # Update target network
            if self.step_count % self.target_update_freq == 0:
                self.target_network.copy_weights_from(self.q_network)
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {
            'loss': np.mean(epoch_losses),
            'loss_std': np.std(epoch_losses)
        }
    
    def train(
        self,
        train_dataloader: DataLoader,
        num_epochs: int,
        eval_dataloaders: Optional[Dict[str, DataLoader]] = None,
        eval_dataset: Optional[MovieLensRLDataset] = None,
        eval_freq: int = 5,
        save_dir: str = 'checkpoints'
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_dataloader: Training data
            num_epochs: Number of epochs
            eval_dataloaders: Dict with 'retain' and 'forget' dataloaders for evaluation
            eval_dataset: Dataset for evaluation
            eval_freq: Evaluate every N epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Training history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'train_loss': [],
            'eval_metrics': []
        }
        
        print(f"\n{'='*70}")
        print("STARTING DQN TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Batches per epoch: {len(train_dataloader)}")
        print(f"Total steps: {num_epochs * len(train_dataloader)}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch(train_dataloader, epoch)
            history['train_loss'].append(train_metrics['loss'])
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_metrics['loss']:.4f} ± {train_metrics['loss_std']:.4f}")
            
            # Evaluate
            if eval_dataloaders is not None and (epoch + 1) % eval_freq == 0:
                print(f"\n--- Evaluation at Epoch {epoch+1} ---")
                evaluator = DQNEvaluator(self.q_network, self.device)
                eval_results = evaluator.evaluate_split(eval_dataloaders, eval_dataset)
                history['eval_metrics'].append({
                    'epoch': epoch + 1,
                    'results': {k: v.compute_aggregate_metrics() for k, v in eval_results.items()}
                })
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
                checkpoint_path = save_dir / f'dqn_epoch_{epoch+1}.pt'
                self.save_checkpoint(checkpoint_path)
        
        # Final save
        final_path = save_dir / 'dqn_trained.pt'
        self.save_checkpoint(final_path)
        print(f"\nTraining complete! Model saved to {final_path}")
        
        return history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step_count': self.step_count
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.step_count = checkpoint.get('step_count', 0)
        print(f"Checkpoint loaded from {path}")


class DecrementalRLTrainer:
    """
    Trainer for Decremental RL-based unlearning.
    """
    
    def __init__(
        self,
        q_network: nn.Module,
        original_q_network: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        learning_rate: float = 1e-5,
        lambda_weight: float = 1.0
    ):
        """
        Args:
            q_network: Q-network to be updated (will be modified)
            original_q_network: Original frozen Q-network (Q_RF)
            device: Device to train on
            learning_rate: Learning rate (should be smaller than normal training)
            lambda_weight: Weight for retain set regularization
        """
        self.device = device
        self.q_network = q_network.to(device)
        self.original_q_network = original_q_network.to(device)
        
        # Freeze original network
        self.original_q_network.eval()
        for param in self.original_q_network.parameters():
            param.requires_grad = False
        
        # Set q_network to training mode
        self.q_network.train()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = DecrementalRLLoss(lambda_weight=lambda_weight)
        
        print(f"\nDecremental RL Trainer initialized")
        print(f"Lambda weight: {lambda_weight}")
        print(f"Learning rate: {learning_rate}")
    
    def collect_random_forget_samples(
        self,
        dataset: MovieLensRLDataset,
        forget_users: set,
        samples_per_user: int = 50
    ) -> Dict[str, torch.Tensor]:
        """
        Collect random exploration samples from forget users.
        
        Args:
            dataset: MovieLens dataset
            forget_users: Set of forget user IDs
            samples_per_user: Number of random samples per user
            
        Returns:
            Dictionary with states and actions
        """
        print(f"\nCollecting random samples from {len(forget_users)} forget users...")
        
        forget_samples = {'states': [], 'actions': []}
        
        for user_id in tqdm(list(forget_users), desc="Random exploration"):
            # Get user's episodes
            user_episodes = [ep for ep in dataset.episodes if ep['user_id'] == user_id]
            
            if len(user_episodes) == 0:
                continue
            
            # Sample random episodes
            sampled_episodes = np.random.choice(
                user_episodes,
                size=min(samples_per_user, len(user_episodes)),
                replace=True
            )
            
            for ep in sampled_episodes:
                # Get state
                state = dataset._encode_state(ep['state_movies'], ep['state_ratings'])
                
                # Random action (uniform)
                random_action = np.random.randint(0, dataset.num_movies)
                
                forget_samples['states'].append(state)
                forget_samples['actions'].append(random_action)
        
        # Convert to tensors
        forget_samples['states'] = torch.stack(forget_samples['states'])
        forget_samples['actions'] = torch.tensor(forget_samples['actions'], dtype=torch.long)
        
        print(f"Collected {len(forget_samples['states'])} random samples")
        return forget_samples
    
    def unlearn_epoch(
        self,
        forget_dataloader: DataLoader,
        retain_dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Unlearning for one epoch.
        
        Returns:
            Dictionary with unlearning metrics
        """
        self.q_network.train()
        epoch_metrics = {
            'total_loss': [],
            'forget_loss': [],
            'retain_loss': [],
            'weighted_retain_loss': []
        }
        
        # Create iterators
        forget_iter = iter(forget_dataloader)
        retain_iter = iter(retain_dataloader)
        
        num_batches = min(len(forget_dataloader), len(retain_dataloader))
        
        pbar = tqdm(range(num_batches), desc=f"Unlearning Epoch {epoch+1}")
        for _ in pbar:
            try:
                forget_batch = next(forget_iter)
                retain_batch = next(retain_iter)
            except StopIteration:
                break
            
            # Move to device
            forget_states = forget_batch['state'].to(self.device)
            forget_actions = forget_batch['action'].to(self.device)
            retain_states = retain_batch['state'].to(self.device)
            retain_actions = retain_batch['action'].to(self.device)
            
            # Compute loss
            loss, loss_dict = self.criterion(
                self.q_network,
                self.original_q_network,
                forget_states, forget_actions,
                retain_states, retain_actions
            )
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
            self.optimizer.step()
            
            # Track metrics
            for key in epoch_metrics:
                epoch_metrics[key].append(loss_dict[key])
            
            # Update progress bar
            pbar.set_postfix({
                'total': f"{loss_dict['total_loss']:.4f}",
                'forget': f"{loss_dict['forget_loss']:.4f}",
                'retain': f"{loss_dict['retain_loss']:.4f}"
            })
        
        # Average metrics
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    def unlearn(
        self,
        forget_dataloader: DataLoader,
        retain_dataloader: DataLoader,
        num_epochs: int,
        save_dir: str = 'checkpoints'
    ) -> Dict:
        """
        Full unlearning loop.
        
        Args:
            forget_dataloader: Forget set data
            retain_dataloader: Retain set data
            num_epochs: Number of unlearning epochs
            save_dir: Directory to save checkpoints
            
        Returns:
            Unlearning history
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        history = {
            'total_loss': [],
            'forget_loss': [],
            'retain_loss': []
        }
        
        print(f"\n{'='*70}")
        print("STARTING DECREMENTAL RL UNLEARNING")
        print(f"{'='*70}")
        print(f"Epochs: {num_epochs}")
        print(f"Forget batches: {len(forget_dataloader)}")
        print(f"Retain batches: {len(retain_dataloader)}")
        print(f"{'='*70}\n")
        
        for epoch in range(num_epochs):
            metrics = self.unlearn_epoch(forget_dataloader, retain_dataloader, epoch)
            
            history['total_loss'].append(metrics['total_loss'])
            history['forget_loss'].append(metrics['forget_loss'])
            history['retain_loss'].append(metrics['retain_loss'])
            
            print(f"Epoch {epoch+1}/{num_epochs} - "
                  f"Total: {metrics['total_loss']:.4f}, "
                  f"Forget: {metrics['forget_loss']:.4f}, "
                  f"Retain: {metrics['retain_loss']:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
                checkpoint_path = save_dir / f'dqn_unlearned_epoch_{epoch+1}.pt'
                torch.save(self.q_network.state_dict(), checkpoint_path)
        
        # Final save
        final_path = save_dir / 'dqn_unlearned.pt'
        torch.save(self.q_network.state_dict(), final_path)
        print(f"\nUnlearning complete! Model saved to {final_path}")
        
        return history


def print_comparison_table(results_before: Dict, results_after: Dict):
    """
    Print comparison table like the paper.
    
    Table format:
    ┌─────────────┬────────────────┬───────────────┬─────────────┐
    │ Set         │ Metric         │ Before        │ After       │
    ├─────────────┼────────────────┼───────────────┼─────────────┤
    │ Forget      │ Accuracy (%)   │ 92.07         │ 68.63       │
    │             │ Cumul. Reward  │ 41.42         │ 20.03       │
    │ Retain      │ Accuracy (%)   │ 91.85         │ 91.12       │
    │             │ Cumul. Reward  │ 40.98         │ 40.23       │
    └─────────────┴────────────────┴───────────────┴─────────────┘
    """
    print("\n" + "="*80)
    print("UNLEARNING RESULTS COMPARISON (Paper-style Table)")
    print("="*80)
    
    # Header
    print(f"\n{'Set':<15} {'Metric':<25} {'Before':<15} {'After':<15} {'Change':<15}")
    print("-"*80)
    
    for set_name in ['forget', 'retain']:
        if set_name not in results_before or set_name not in results_after:
            continue
        
        before = results_before[set_name].compute_aggregate_metrics()
        after = results_after[set_name].compute_aggregate_metrics()
        
        # Top-1 Accuracy
        acc_before = before['mean_top1_accuracy'] * 100
        acc_after = after['mean_top1_accuracy'] * 100
        acc_change = acc_after - acc_before
        
        print(f"{set_name.upper():<15} {'Top-1 Accuracy (%)':<25} "
              f"{acc_before:>10.2f}     {acc_after:>10.2f}     {acc_change:>+10.2f}")
        
        # Cumulative Reward
        reward_before = before['mean_cumulative_reward']
        reward_after = after['mean_cumulative_reward']
        reward_change = reward_after - reward_before
        
        print(f"{'':15} {'Mean Cumul. Reward':<25} "
              f"{reward_before:>10.2f}     {reward_after:>10.2f}     {reward_change:>+10.2f}")
        
        # Average Reward
        avg_reward_before = before['mean_avg_reward']
        avg_reward_after = after['mean_avg_reward']
        avg_reward_change = avg_reward_after - avg_reward_before
        
        print(f"{'':15} {'Mean Avg Reward (stars)':<25} "
              f"{avg_reward_before:>10.3f}     {avg_reward_after:>10.3f}     {avg_reward_change:>+10.3f}")
        
        print("-"*80)
    
    print("="*80)


def main(args):
    """Main training pipeline"""
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f"\n{'='*80}")
    print("DQN RECOMMENDATION SYSTEM - TRAINING WITH UNLEARNING")
    print(f"{'='*80}")
    print(f"Dataset: {args.data_dir}")
    print(f"Forget ratio: {args.forget_ratio}")
    print(f"Use genome: {args.use_genome}")
    print(f"Pilot mode: {args.pilot_users if args.pilot_users else 'Disabled'}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    # Load data
    print("Loading MovieLens data...")
    dataloaders, dataset = create_dataloaders(
        data_dir=args.data_dir,
        forget_ratio=args.forget_ratio,
        use_genome=args.use_genome,
        state_size=args.state_size,
        batch_size=args.batch_size,
        user_total=args.pilot_users,  
        seed=args.seed
    )
    
    # Get dimensions
    state_dim = dataset.get_state_dim()
    action_dim = dataset.get_action_dim()
    
    print(f"\nModel dimensions:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    # ==================== PHASE 1: NORMAL TRAINING ====================
    if not args.skip_training:
        print("\n" + "="*80)
        print("PHASE 1: NORMAL DQN TRAINING")
        print("="*80)
        
        trainer = DQNTrainer(
            state_dim=state_dim,
            action_dim=action_dim,
            device=args.device,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            target_update_freq=args.target_update_freq,
            hidden_dims=args.hidden_dims
        )
        
        training_history = trainer.train(
            train_dataloader=dataloaders['train'],
            num_epochs=args.num_epochs,
            eval_dataloaders={'retain': dataloaders['retain'], 'forget': dataloaders['forget']},
            eval_dataset=dataset,
            eval_freq=args.eval_freq,
            save_dir=args.checkpoint_dir
        )
        
        # Evaluate before unlearning
        print("\n" + "="*80)
        print("EVALUATION BEFORE UNLEARNING")
        print("="*80)
        evaluator = DQNEvaluator(trainer.q_network, args.device)
        results_before = evaluator.evaluate_split(dataloaders, dataset, splits=['retain', 'forget'])
        
        # Save original model
        model_before = trainer.q_network
    else:
        # Load pretrained model
        print(f"\nLoading pretrained model from {args.pretrained_path}")
        model_before = create_dqn(state_dim, action_dim, hidden_dims=args.hidden_dims).to(args.device)
        model_before.load_state_dict(torch.load(args.pretrained_path))
        
        evaluator = DQNEvaluator(model_before, args.device)
        results_before = evaluator.evaluate_split(dataloaders, dataset, splits=['retain', 'forget'])
    
    # ==================== PHASE 2: UNLEARNING ====================
    if not args.skip_unlearning:
        print("\n" + "="*80)
        print("PHASE 2: DECREMENTAL RL UNLEARNING")
        print("="*80)
        
        # Create copy for unlearning
        model_after = create_dqn(state_dim, action_dim, hidden_dims=args.hidden_dims).to(args.device)
        model_after.load_state_dict(model_before.state_dict())
        
        # Unlearning trainer
        unlearning_trainer = DecrementalRLTrainer(
            q_network=model_after,
            original_q_network=model_before,
            device=args.device,
            learning_rate=args.unlearning_lr,
            lambda_weight=args.lambda_weight
        )
        
        # Unlearning
        unlearning_history = unlearning_trainer.unlearn(
            forget_dataloader=dataloaders['forget'],
            retain_dataloader=dataloaders['retain'],
            num_epochs=args.unlearning_epochs,
            save_dir=args.checkpoint_dir
        )
        
        # Evaluate after unlearning
        print("\n" + "="*80)
        print("EVALUATION AFTER UNLEARNING")
        print("="*80)
        evaluator_after = DQNEvaluator(model_after, args.device)
        results_after = evaluator_after.evaluate_split(dataloaders, dataset, splits=['retain', 'forget'])
        
        # Print comparison table
        print_comparison_table(results_before, results_after)
        
        # Save results
        results_path = Path(args.checkpoint_dir) / 'unlearning_results.json'
        with open(results_path, 'w') as f:
            json.dump({
                'before': {k: v.compute_aggregate_metrics() for k, v in results_before.items()},
                'after': {k: v.compute_aggregate_metrics() for k, v in results_after.items()},
                'training_history': training_history if not args.skip_training else None,
                'unlearning_history': unlearning_history
            }, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN with Decremental RL Unlearning")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                       default='E:\\Kuliah\\Kuliah\\Kuliah\\PRODI\\Semester 7\\ProSkripCode\\data_movie',
                       help='Path to MovieLens data directory')
    parser.add_argument('--forget_ratio', type=float, default=0.1,
                       help='Ratio of users to forget')
    parser.add_argument('--use_genome', action='store_true',
                       help='Use genome features')
    parser.add_argument('--state_size', type=int, default=50,
                       help='State size (number of movies in history)')
    parser.add_argument('--pilot_users', type=int, default=None,
                       help='Number of users for pilot testing (None = all users)')
    
    # Model arguments
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 256, 128],
                       help='Hidden layer dimensions')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Learning rate for normal training')
    parser.add_argument('--target_update_freq', type=int, default=1000,
                       help='Target network update frequency (steps)')
    parser.add_argument('--eval_freq', type=int, default=5,
                       help='Evaluation frequency (epochs)')
    
    # Unlearning arguments
    parser.add_argument('--unlearning_epochs', type=int, default=10,
                       help='Number of unlearning epochs')
    parser.add_argument('--unlearning_lr', type=float, default=1e-5,
                       help='Learning rate for unlearning')
    parser.add_argument('--lambda_weight', type=float, default=1.0,
                       help='Lambda weight for retain set regularization')
    
    # System arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    
    # Pipeline control
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip normal training phase')
    parser.add_argument('--skip_unlearning', action='store_true',
                       help='Skip unlearning phase')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model (if skip_training)')
    
    args = parser.parse_args()
    
    # Run main
    main(args)
