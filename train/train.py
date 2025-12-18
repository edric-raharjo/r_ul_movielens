# train/train.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import numpy as np
import json
import copy
from tqdm import tqdm
import random

from dataset.dataloader import create_dataloaders, MovieLensRecommendationDataset
from model.dqn_model import RecommendationDQN, DQNWithTargetNetwork
from model.loss_function import (
    DQNRecommendationLoss, 
    DecrementalRLLoss,
    create_random_forget_samples
)
from eval.evaluate import RecommendationEvaluator, evaluate_before_after_unlearning
from eval.eval_metrics import print_metrics


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(
    dqn_manager: DQNWithTargetNetwork,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: DQNRecommendationLoss,
    device: str
) -> dict:
    """
    Train for one epoch.
    
    Returns:
        Dictionary with training metrics
    """
    dqn_manager.train_mode()
    online_net = dqn_manager.get_online_network()
    
    epoch_losses = []
    epoch_accuracies = []
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move to device
        inputs = batch['input_features'].to(device)
        labels = batch['label'].squeeze().to(device)
        rewards = batch['reward'].squeeze().to(device)
        
        # Forward pass
        q_values = online_net(inputs)
        
        # Compute loss
        loss, loss_dict = loss_fn(q_values, rewards, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update target network
        dqn_manager.update_target_network()
        
        # Track metrics
        epoch_losses.append(loss_dict['total_loss'])
        epoch_accuracies.append(loss_dict['accuracy'])
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_dict['total_loss']:.4f}",
            'acc': f"{loss_dict['accuracy']:.2%}"
        })
    
    return {
        'loss': np.mean(epoch_losses),
        'loss_std': np.std(epoch_losses),
        'accuracy': np.mean(epoch_accuracies),
        'accuracy_std': np.std(epoch_accuracies)
    }


def train_normal_phase(
    dqn_manager: DQNWithTargetNetwork,
    train_loader: DataLoader,
    args,
    checkpoint_dir: Path
) -> RecommendationDQN:
    """
    Normal training phase (Phase 1).
    
    Returns:
        Trained model
    """
    print("\n" + "="*80)
    print("PHASE 1: NORMAL DQN TRAINING")
    print("="*80)
    
    online_net = dqn_manager.get_online_network()
    optimizer = optim.Adam(online_net.parameters(), lr=args.learning_rate)
    loss_fn = DQNRecommendationLoss(rating_threshold=4.0)
    
    best_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch}/{args.num_epochs}")
        print(f"{'='*80}")
        
        # Train
        train_metrics = train_epoch(
            dqn_manager, train_loader, optimizer, loss_fn, args.device
        )
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Loss: {train_metrics['loss']:.4f} ± {train_metrics['loss_std']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.2%} ± {train_metrics['accuracy_std']:.2%}")
        
        # Save checkpoint
        if epoch % args.eval_freq == 0:
            checkpoint_path = checkpoint_dir / f'dqn_epoch_{epoch}.pt'
            torch.save(online_net.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            best_path = checkpoint_dir / 'dqn_best.pt'
            torch.save(online_net.state_dict(), best_path)
            print(f"  Best model saved: {best_path}")
    
    # Save final trained model
    trained_path = checkpoint_dir / 'dqn_trained.pt'
    torch.save(online_net.state_dict(), trained_path)
    print(f"\nTraining complete! Model saved to {trained_path}")
    
    return online_net


def unlearning_epoch(
    model: RecommendationDQN,
    original_model: RecommendationDQN,
    random_forget_samples: dict,
    retain_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: DecrementalRLLoss,
    device: str
) -> dict:
    """
    One epoch of decremental RL unlearning.
    
    Returns:
        Dictionary with unlearning metrics
    """
    model.train()
    
    epoch_losses = []
    epoch_forget_losses = []
    epoch_retain_losses = []
    
    # Create iterator for retain set
    retain_iterator = iter(retain_loader)
    
    # Number of batches = number of forget samples / batch_size
    num_forget_samples = len(random_forget_samples['inputs'])
    batch_size = 32  # Use smaller batch for unlearning
    num_batches = max(1, num_forget_samples // batch_size)
    
    pbar = tqdm(range(num_batches), desc="Unlearning")
    for batch_idx in pbar:
        # Get forget batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_forget_samples)
        
        forget_inputs = random_forget_samples['inputs'][start_idx:end_idx]
        forget_actions = random_forget_samples['actions'][start_idx:end_idx]
        
        # Get retain batch
        try:
            retain_batch = next(retain_iterator)
        except StopIteration:
            retain_iterator = iter(retain_loader)
            retain_batch = next(retain_iterator)
        
        retain_inputs = retain_batch['input_features'].to(device)
        
        # Forward pass on forget set
        q_forget = model(forget_inputs)
        
        # Forward pass on retain set (both models)
        q_retain_new = model(retain_inputs)
        with torch.no_grad():
            q_retain_original = original_model(retain_inputs)
        
        # Compute unlearning loss
        loss, loss_dict = loss_fn(
            q_forget, forget_actions,
            q_retain_new, q_retain_original
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        epoch_losses.append(loss_dict['total_loss'])
        epoch_forget_losses.append(loss_dict['forget_loss'])
        epoch_retain_losses.append(loss_dict['retain_loss'])
        
        # Update progress bar
        pbar.set_postfix({
            'total': f"{loss_dict['total_loss']:.4f}",
            'forget': f"{loss_dict['forget_loss']:.4f}",
            'retain': f"{loss_dict['retain_loss']:.4f}"
        })
    
    return {
        'total_loss': np.mean(epoch_losses),
        'forget_loss': np.mean(epoch_forget_losses),
        'retain_loss': np.mean(epoch_retain_losses),
        'total_loss_std': np.std(epoch_losses)
    }


def unlearning_phase(
    trained_model: RecommendationDQN,
    forget_dataset: MovieLensRecommendationDataset,
    retain_loader: DataLoader,
    forget_users: set,
    args,
    checkpoint_dir: Path
) -> RecommendationDQN:
    """
    Decremental RL unlearning phase (Phase 2).
    
    Returns:
        Unlearned model
    """
    print("\n" + "="*80)
    print("PHASE 2: DECREMENTAL RL UNLEARNING")
    print("="*80)
    
    # Create random forget samples
    print("\nGenerating random samples for forget set...")
    random_samples = create_random_forget_samples(
        forget_dataset,
        forget_users,
        num_samples_per_user=50,
        device=args.device
    )
    
    if len(random_samples['inputs']) == 0:
        print("ERROR: No forget samples generated!")
        return trained_model
    
    # Create copy of model for unlearning
    unlearning_model = copy.deepcopy(trained_model)
    original_model = copy.deepcopy(trained_model)
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False
    
    # Optimizer and loss
    optimizer = optim.Adam(unlearning_model.parameters(), lr=args.unlearning_lr)
    loss_fn = DecrementalRLLoss(lambda_weight=args.lambda_weight)
    
    print(f"\nUnlearning Configuration:")
    print(f"  Epochs: {args.unlearning_epochs}")
    print(f"  Learning rate: {args.unlearning_lr}")
    print(f"  Lambda weight: {args.lambda_weight}")
    print(f"  Forget samples: {len(random_samples['inputs'])}")
    
    # Unlearning loop
    for epoch in range(1, args.unlearning_epochs + 1):
        print(f"\n{'='*80}")
        print(f"Unlearning Epoch {epoch}/{args.unlearning_epochs}")
        print(f"{'='*80}")
        
        # Unlearn
        unlearn_metrics = unlearning_epoch(
            unlearning_model, original_model,
            random_samples, retain_loader,
            optimizer, loss_fn, args.device
        )
        
        print(f"\nEpoch {epoch} Results:")
        print(f"  Total Loss: {unlearn_metrics['total_loss']:.4f}")
        print(f"  Forget Loss: {unlearn_metrics['forget_loss']:.4f}")
        print(f"  Retain Loss: {unlearn_metrics['retain_loss']:.4f}")
        
        # Save checkpoint
        if epoch % 5 == 0:
            checkpoint_path = checkpoint_dir / f'dqn_unlearned_epoch_{epoch}.pt'
            torch.save(unlearning_model.state_dict(), checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")
    
    # Save final unlearned model
    unlearned_path = checkpoint_dir / 'dqn_unlearned.pt'
    torch.save(unlearning_model.state_dict(), unlearned_path)
    print(f"\nUnlearning complete! Model saved to {unlearned_path}")
    
    return unlearning_model


def main(args):
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("DQN RECOMMENDATION SYSTEM WITH DECREMENTAL RL UNLEARNING")
    print("="*80)
    
    # Set seed
    set_seed(args.seed)
    print(f"\nRandom seed set to {args.seed}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save arguments
    with open(checkpoint_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Configuration saved to {checkpoint_dir / 'config.json'}")
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    dataloaders, train_dataset = create_dataloaders(
        data_dir=args.data_dir,
        forget_ratio=args.forget_ratio,
        rating_threshold=4.0,
        batch_size=args.batch_size,
        user_total=args.pilot_users,
        seed=args.seed
    )
    
    train_loader = dataloaders['train']
    retain_test_loader = dataloaders['retain_test']
    forget_test_loader = dataloaders['forget_test']
    
    # Get retain and forget users
    retain_users = train_dataset.retain_users
    forget_users = train_dataset.forget_users
    
    print(f"\nData loaded successfully!")
    print(f"  Train samples: {len(train_dataset):,}")
    print(f"  Retain users: {len(retain_users)}")
    print(f"  Forget users: {len(forget_users)}")
    
    # Create test datasets for evaluation
    retain_test_dataset = MovieLensRecommendationDataset(
        data_dir=args.data_dir,
        forget_ratio=args.forget_ratio,
        rating_threshold=4.0,
        user_total=args.pilot_users,
        mode='retain',
        split='test',
        seed=args.seed
    )
    
    forget_test_dataset = MovieLensRecommendationDataset(
        data_dir=args.data_dir,
        forget_ratio=args.forget_ratio,
        rating_threshold=4.0,
        user_total=args.pilot_users,
        mode='forget',
        split='test',
        seed=args.seed
    )
    
    # Phase 1: Normal Training
    if not args.skip_training:
        # Create DQN manager
        print("\n" + "="*80)
        print("INITIALIZING MODEL")
        print("="*80)
        
        # Note: Input dim is 44 (22 user + 22 candidate) as per our design
        dqn_manager = DQNWithTargetNetwork(
            input_dim=42,  # Fixed based on our architecture
            hidden_dims=[128, 128],  # From paper
            dropout_rate=0.2,
            device=args.device,
            target_update_freq=args.target_update_freq,
            use_soft_update=False
        )
        
        trained_model = train_normal_phase(
            dqn_manager, train_loader, args, checkpoint_dir
        )
    else:
        # Load pretrained model
        print("\n" + "="*80)
        print("LOADING PRETRAINED MODEL")
        print("="*80)
        
        if args.pretrained_path is None:
            args.pretrained_path = str(checkpoint_dir / 'dqn_trained.pt')
        
        trained_model = RecommendationDQN(input_dim=44, hidden_dims=[128, 128])
        trained_model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
        trained_model.to(args.device)
        print(f"Model loaded from {args.pretrained_path}")
    
    # Evaluate before unlearning
    print("\n" + "="*80)
    print("EVALUATION BEFORE UNLEARNING")
    print("="*80)
    
    evaluator_before = RecommendationEvaluator(trained_model, args.device)
    
    print("\nEvaluating RETAIN set...")
    retain_metrics_before = evaluator_before.evaluate_full(
        retain_test_dataset, 
        batch_size=128,
        include_ranking=True
    )
    print_metrics(retain_metrics_before, "RETAIN - Before Unlearning")
    
    print("\nEvaluating FORGET set...")
    forget_metrics_before = evaluator_before.evaluate_full(
        forget_test_dataset,
        batch_size=128,
        include_ranking=True
    )
    print_metrics(forget_metrics_before, "FORGET - Before Unlearning")
    
    # Phase 2: Unlearning
    if not args.skip_unlearning:
        unlearned_model = unlearning_phase(
            trained_model,
            forget_test_dataset,
            retain_test_loader,
            forget_users,
            args,
            checkpoint_dir
        )
    else:
        print("\nSkipping unlearning phase (--skip_unlearning flag set)")
        unlearned_model = trained_model
    
    # Evaluate after unlearning
    print("\n" + "="*80)
    print("EVALUATION AFTER UNLEARNING")
    print("="*80)
    
    evaluator_after = RecommendationEvaluator(unlearned_model, args.device)
    
    print("\nEvaluating RETAIN set...")
    retain_metrics_after = evaluator_after.evaluate_full(
        retain_test_dataset,
        batch_size=128,
        include_ranking=True
    )
    print_metrics(retain_metrics_after, "RETAIN - After Unlearning")
    
    print("\nEvaluating FORGET set...")
    forget_metrics_after = evaluator_after.evaluate_full(
        forget_test_dataset,
        batch_size=128,
        include_ranking=True
    )
    print_metrics(forget_metrics_after, "FORGET - After Unlearning")
    
    # Final comparison
    from eval.eval_metrics import compare_metrics, print_comparison
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    
    retain_comparison = compare_metrics(retain_metrics_before, retain_metrics_after)
    print_comparison(retain_comparison, "RETAIN SET: Before vs After")
    
    forget_comparison = compare_metrics(forget_metrics_before, forget_metrics_after)
    print_comparison(forget_comparison, "FORGET SET: Before vs After")
    
    # Save results
    results = {
        'retain': {
            'before': retain_metrics_before,
            'after': retain_metrics_after,
            'comparison': retain_comparison
        },
        'forget': {
            'before': forget_metrics_before,
            'after': forget_metrics_after,
            'comparison': forget_comparison
        }
    }
    
    results_path = checkpoint_dir / 'unlearning_results.json'
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python
        def convert(obj):
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN with Decremental RL Unlearning")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, 
                       default='E:\\Kuliah\\Kuliah\\Kuliah\\PRODI\\Semester 7\\ProSkripCode\\data_movie',
                       help='Path to MovieLens data directory')
    parser.add_argument('--forget_ratio', type=float, default=0.1,
                       help='Ratio of users to forget')
    parser.add_argument('--pilot_users', type=int, default=None,
                       help='Number of users for pilot testing (None = all users)')
    
    # Model arguments (kept for compatibility but not used since we have fixed architecture)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 128],
                       help='Hidden layer dimensions')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor (not used in current implementation)')
    
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
