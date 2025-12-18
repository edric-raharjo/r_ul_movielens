# train/hyperparameter_sweep.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import random
from datetime import datetime
import optuna
from optuna.trial import TrialState
import argparse

from dataset.dataloader import MovieLensRecommendationDataset
from model.dqn_model import RecommendationDQN, DQNWithTargetNetwork
from model.loss_function import (
    DQNRecommendationLoss, 
    DecrementalRLLoss,
    create_random_forget_samples
)
from eval.evaluate import RecommendationEvaluator


def check_cuda_usage():
    """Check if CUDA is available and being used properly"""
    print("\n" + "="*80)
    print("CUDA STATUS CHECK")
    print("="*80)
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Device Count: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        
        # Check memory
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        
        # Test tensor allocation
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            allocated = torch.cuda.memory_allocated(0) / 1e9
            print(f"Memory Allocated (test): {allocated:.4f} GB")
            del test_tensor
            torch.cuda.empty_cache()
            print("✓ CUDA is working properly!")
            return 'cuda'
        except Exception as e:
            print(f"✗ CUDA test failed: {e}")
            print("Falling back to CPU...")
            return 'cpu'
    else:
        print("✗ CUDA not available. Using CPU.")
        return 'cpu'


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_three_way_split_dataloaders(
    data_dir: str,
    retain_ratio: float,
    forget_ratio: float,
    test_ratio: float,
    rating_threshold: float = 4.0,
    batch_size: int = 64,
    user_total: int = None,
    seed: int = 42
):
    """
    Create dataloaders with 3-way split: retain, forget, test
    
    Returns:
        dict with dataloaders and datasets for all splits
    """
    
    # Validate ratios
    assert abs(retain_ratio + forget_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0! Got {retain_ratio + forget_ratio + test_ratio}"
    
    print(f"\n3-Way Data Split:")
    print(f"  Retain: {retain_ratio*100:.1f}%")
    print(f"  Forget: {forget_ratio*100:.1f}%")
    print(f"  Test:   {test_ratio*100:.1f}%")
    
    # Load full dataset to get user list
    print("\nLoading data to determine user split...")
    full_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=0.0,  # Temporary, we'll split manually
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='train',
        split='train',
        seed=seed
    )
    
    # Get all valid users
    all_users = list(full_dataset.valid_users)
    np.random.seed(seed)
    np.random.shuffle(all_users)
    
    # Split users into 3 sets
    n_users = len(all_users)
    n_retain = int(n_users * retain_ratio)
    n_forget = int(n_users * forget_ratio)
    # n_test = remaining
    
    retain_users = set(all_users[:n_retain])
    forget_users = set(all_users[n_retain:n_retain+n_forget])
    test_users = set(all_users[n_retain+n_forget:])
    
    print(f"\nUser Split:")
    print(f"  Retain users: {len(retain_users)}")
    print(f"  Forget users: {len(forget_users)}")
    print(f"  Test users:   {len(test_users)}")
    
    # Create datasets for each split
    # We'll manually override the user sets after creation
    
    # RETAIN dataset (train split)
    retain_train_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=0.0,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='train',
        split='train',
        seed=seed
    )
    # Override users
    retain_train_dataset.retain_users = retain_users
    retain_train_dataset.forget_users = forget_users | test_users  # Everything else
    retain_train_dataset.current_users = list(retain_users)
    retain_train_dataset._precompute_user_states()
    retain_train_dataset._prepare_samples()
    
    # FORGET dataset (train split) - for training "full" model
    forget_train_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=0.0,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='train',
        split='train',
        seed=seed
    )
    forget_train_dataset.retain_users = retain_users
    forget_train_dataset.forget_users = forget_users
    forget_train_dataset.current_users = list(forget_users)
    forget_train_dataset._precompute_user_states()
    forget_train_dataset._prepare_samples()
    
    # FORGET dataset (test split) - for evaluation & unlearning
    forget_test_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=0.0,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='train',
        split='test',
        seed=seed
    )
    forget_test_dataset.retain_users = retain_users
    forget_test_dataset.forget_users = forget_users
    forget_test_dataset.current_users = list(forget_users)
    forget_test_dataset._precompute_user_states()
    forget_test_dataset._prepare_samples()
    
    # TEST dataset (test split) - NEVER used in training
    test_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=0.0,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='train',
        split='test',
        seed=seed
    )
    test_dataset.retain_users = retain_users | forget_users
    test_dataset.forget_users = test_users
    test_dataset.current_users = list(test_users)
    test_dataset._precompute_user_states()
    test_dataset._prepare_samples()
    
    print(f"\nDataset Sizes:")
    print(f"  Retain train: {len(retain_train_dataset):,} samples")
    print(f"  Forget train: {len(forget_train_dataset):,} samples")
    print(f"  Forget test:  {len(forget_test_dataset):,} samples")
    print(f"  Test set:     {len(test_dataset):,} samples")
    
    # Create dataloaders
    dataloaders = {
        'retain_train': DataLoader(retain_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'forget_train': DataLoader(forget_train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'forget_test': DataLoader(forget_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'test': DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    
    datasets = {
        'retain_train': retain_train_dataset,
        'forget_train': forget_train_dataset,
        'forget_test': forget_test_dataset,
        'test': test_dataset,
        'forget_users': forget_users,
        'test_users': test_users
    }
    
    return dataloaders, datasets



def train_epoch_fast(dqn_manager, train_loader, optimizer, loss_fn, device):
    """Fast training epoch"""
    dqn_manager.train_mode()
    online_net = dqn_manager.get_online_network()
    
    epoch_losses = []
    epoch_accuracies = []
    
    for batch in train_loader:
        inputs = batch['input_features'].to(device)
        labels = batch['label'].to(device)
        rewards = batch['reward'].to(device)
        
        # FIX: Ensure consistent shape (flatten to 1D)
        if labels.dim() > 1:
            labels = labels.squeeze()
        if rewards.dim() > 1:
            rewards = rewards.squeeze()
        
        # Additional safety check
        if labels.dim() == 0:  # scalar
            labels = labels.unsqueeze(0)
        if rewards.dim() == 0:  # scalar
            rewards = rewards.unsqueeze(0)
        
        q_values = online_net(inputs)
        loss, loss_dict = loss_fn(q_values, rewards, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=1.0)
        optimizer.step()
        
        dqn_manager.update_target_network()
        
        epoch_losses.append(loss_dict['total_loss'])
        epoch_accuracies.append(loss_dict['accuracy'])
    
    return np.mean(epoch_losses), np.mean(epoch_accuracies)



def unlearning_epoch_fast(model, original_model, random_samples, retain_loader, 
                          optimizer, loss_fn, device):
    """Fast unlearning epoch"""
    model.train()
    
    epoch_losses = []
    retain_iterator = iter(retain_loader)
    
    num_forget_samples = len(random_samples['inputs'])
    batch_size = 32
    num_batches = max(1, num_forget_samples // batch_size)
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_forget_samples)
        
        forget_inputs = random_samples['inputs'][start_idx:end_idx]
        forget_actions = random_samples['actions'][start_idx:end_idx]
        
        try:
            retain_batch = next(retain_iterator)
        except StopIteration:
            retain_iterator = iter(retain_loader)
            retain_batch = next(retain_iterator)
        
        retain_inputs = retain_batch['input_features'].to(device)
        
        q_forget = model(forget_inputs)
        q_retain_new = model(retain_inputs)
        
        with torch.no_grad():
            q_retain_original = original_model(retain_inputs)
        
        loss, loss_dict = loss_fn(
            q_forget, forget_actions,
            q_retain_new, q_retain_original
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_losses.append(loss_dict['total_loss'])
    
    return np.mean(epoch_losses)


class ObjectiveFunction:
    """Objective function for Optuna optimization"""
    
    def __init__(
        self,
        dataloaders,
        datasets,
        actual_input_dim,
        device,
        results_storage
    ):
        self.dataloaders = dataloaders
        self.datasets = datasets
        self.actual_input_dim = actual_input_dim
        self.device = device
        self.results_storage = results_storage
    
    def __call__(self, trial: optuna.Trial):
        """
        Objective function to maximize unlearning quality.
        
        Quality = forget_drop - 2 * test_drop
        
        Where:
        - forget_drop = Model2 forget acc - Model3 forget acc (should be LARGE)
        - test_drop = Model2 test acc - Model3 test acc (should be SMALL)
        """
        
        # Sample hyperparameters
        lambda_weight = trial.suggest_float('lambda_weight', 0.01, 100.0, log=True)
        unlearn_lr = trial.suggest_float('unlearn_lr', 1e-5, 1e-3, log=True)
        train_lr = trial.suggest_float('train_lr', 1e-5, 5e-3, log=True)
        train_epochs = trial.suggest_int('train_epochs', 5, 20, step=2)
        unlearn_epochs = trial.suggest_int('unlearn_epochs', 5, 20, step=2)
        
        print(f"\n{'='*80}")
        print(f"Trial {trial.number}")
        print(f"  lambda={lambda_weight:.3f}, unlearn_lr={unlearn_lr:.2e}, train_lr={train_lr:.2e}")
        print(f"  train_epochs={train_epochs}, unlearn_epochs={unlearn_epochs}")
        print(f"{'='*80}")
        
        try:
            # ==================================================================
            # MODEL 1: BASELINE (Train on RETAIN only)
            # ==================================================================
            print(f"\n[Trial {trial.number}] MODEL 1: Training baseline (retain only)...")
            
            dqn_baseline = DQNWithTargetNetwork(
                input_dim=self.actual_input_dim,
                hidden_dims=[128, 128],
                dropout_rate=0.2,
                device=self.device,
                target_update_freq=1000,
                use_soft_update=False
            )
            
            model1 = dqn_baseline.get_online_network()
            optimizer1 = optim.Adam(model1.parameters(), lr=train_lr)
            loss_fn = DQNRecommendationLoss(rating_threshold=4.0)
            
            for epoch in range(train_epochs):
                train_loss, train_acc = train_epoch_fast(
                    dqn_baseline, self.dataloaders['retain_train'], 
                    optimizer1, loss_fn, self.device
                )
                
                trial.report(train_acc, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            print(f"  Model 1 trained. Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
            
            # Evaluate Model 1
            eval1 = RecommendationEvaluator(model1, self.device)
            model1_forget = eval1.evaluate_classification(self.datasets['forget_test'], batch_size=128)
            model1_test = eval1.evaluate_classification(self.datasets['test'], batch_size=128)
            
            print(f"  Model 1 - Forget: {model1_forget['accuracy']:.2%}, Test: {model1_test['accuracy']:.2%}")
            
            # ==================================================================
            # MODEL 2: FULL (Train on RETAIN + FORGET)
            # ==================================================================
            print(f"\n[Trial {trial.number}] MODEL 2: Training full model (retain + forget)...")
            
            # Combine retain and forget data
            from torch.utils.data import ConcatDataset
            combined_train = ConcatDataset([
                self.datasets['retain_train'],
                self.datasets['forget_train']
            ])
            combined_loader = DataLoader(combined_train, batch_size=64, shuffle=True, num_workers=0)
            
            dqn_full = DQNWithTargetNetwork(
                input_dim=self.actual_input_dim,
                hidden_dims=[128, 256, 256, 128],
                dropout_rate=0.2,
                device=self.device,
                target_update_freq=1000,
                use_soft_update=False
            )
            
            model2 = dqn_full.get_online_network()
            optimizer2 = optim.Adam(model2.parameters(), lr=train_lr)
            
            for epoch in range(train_epochs):
                train_loss, train_acc = train_epoch_fast(
                    dqn_full, combined_loader,
                    optimizer2, loss_fn, self.device
                )
            
            print(f"  Model 2 trained. Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
            
            # Evaluate Model 2
            eval2 = RecommendationEvaluator(model2, self.device)
            model2_forget = eval2.evaluate_classification(self.datasets['forget_test'], batch_size=128)
            model2_test = eval2.evaluate_classification(self.datasets['test'], batch_size=128)
            
            print(f"  Model 2 - Forget: {model2_forget['accuracy']:.2%}, Test: {model2_test['accuracy']:.2%}")
            
            # ==================================================================
            # MODEL 3: UNLEARNED (Model 2 + Decremental RL on FORGET)
            # ==================================================================
            print(f"\n[Trial {trial.number}] MODEL 3: Unlearning forget set...")
            
            # Generate random samples from forget set
            random_samples = create_random_forget_samples(
                self.datasets['forget_test'], 
                self.datasets['forget_users'], 
                num_samples_per_user=50, 
                device=self.device
            )
            
            if len(random_samples['inputs']) == 0:
                print("ERROR: No forget samples!")
                raise optuna.TrialPruned()
            
            # Create unlearning model
            model3 = copy.deepcopy(model2)
            original_model = copy.deepcopy(model2)
            original_model.eval()
            for param in original_model.parameters():
                param.requires_grad = False
            
            optimizer3 = optim.Adam(model3.parameters(), lr=unlearn_lr)
            loss_fn_unlearn = DecrementalRLLoss(lambda_weight=lambda_weight)
            
            for epoch in range(unlearn_epochs):
                unlearn_loss = unlearning_epoch_fast(
                    model3, original_model,
                    random_samples, self.dataloaders['retain_train'],
                    optimizer3, loss_fn_unlearn, self.device
                )
            
            print(f"  Model 3 unlearned. Loss: {unlearn_loss:.4f}")
            
            # Evaluate Model 3
            eval3 = RecommendationEvaluator(model3, self.device)
            model3_forget = eval3.evaluate_classification(self.datasets['forget_test'], batch_size=128)
            model3_test = eval3.evaluate_classification(self.datasets['test'], batch_size=128)
            
            print(f"  Model 3 - Forget: {model3_forget['accuracy']:.2%}, Test: {model3_test['accuracy']:.2%}")
            
            # ==================================================================
            # COMPUTE QUALITY METRIC
            # ==================================================================
            
            # Drop in forget accuracy (should be LARGE)
            forget_drop = model2_forget['accuracy'] - model3_forget['accuracy']
            
            # Drop in test accuracy (should be SMALL)
            test_drop = model2_test['accuracy'] - model3_test['accuracy']
            
            # Quality metric
            quality = forget_drop - 2 * test_drop
            
            print(f"\n{'='*80}")
            print(f"Trial {trial.number} Results:")
            print(f"  Model 1 (Baseline)  - Forget: {model1_forget['accuracy']:.2%}, Test: {model1_test['accuracy']:.2%}")
            print(f"  Model 2 (Full)      - Forget: {model2_forget['accuracy']:.2%}, Test: {model2_test['accuracy']:.2%}")
            print(f"  Model 3 (Unlearned) - Forget: {model3_forget['accuracy']:.2%}, Test: {model3_test['accuracy']:.2%}")
            print(f"\n  Forget drop: {forget_drop:.4f} ({forget_drop/model2_forget['accuracy']*100:.1f}%)")
            print(f"  Test drop:   {test_drop:.4f} ({test_drop/model2_test['accuracy']*100:.1f}%)")
            print(f"  Quality:     {quality:.4f}")
            print(f"{'='*80}")
            
            # Store results
            results = {
                'trial': trial.number,
                'lambda_weight': lambda_weight,
                'unlearn_lr': unlearn_lr,
                'train_lr': train_lr,
                'train_epochs': train_epochs,
                'unlearn_epochs': unlearn_epochs,
                
                # Model 1 (Baseline)
                'model1_forget_acc': model1_forget['accuracy'],
                'model1_test_acc': model1_test['accuracy'],
                
                # Model 2 (Full)
                'model2_forget_acc': model2_forget['accuracy'],
                'model2_test_acc': model2_test['accuracy'],
                
                # Model 3 (Unlearned)
                'model3_forget_acc': model3_forget['accuracy'],
                'model3_test_acc': model3_test['accuracy'],
                
                # Drops
                'forget_drop': forget_drop,
                'test_drop': test_drop,
                'forget_drop_pct': forget_drop / model2_forget['accuracy'] * 100,
                'test_drop_pct': test_drop / model2_test['accuracy'] * 100,
                
                # Quality
                'quality': quality,
                'state': 'COMPLETE'
            }
            
            self.results_storage.append(results)
            self._save_results()
            
            return quality
        
        except optuna.TrialPruned:
            print(f"Trial {trial.number} pruned!")
            results = {
                'trial': trial.number,
                'lambda_weight': lambda_weight,
                'unlearn_lr': unlearn_lr,
                'train_lr': train_lr,
                'train_epochs': train_epochs,
                'unlearn_epochs': unlearn_epochs,
                'state': 'PRUNED'
            }
            self.results_storage.append(results)
            self._save_results()
            raise
        
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            import traceback
            traceback.print_exc()
            
            results = {
                'trial': trial.number,
                'lambda_weight': lambda_weight,
                'unlearn_lr': unlearn_lr,
                'train_lr': train_lr,
                'train_epochs': train_epochs,
                'unlearn_epochs': unlearn_epochs,
                'state': 'FAILED',
                'error': str(e)
            }
            self.results_storage.append(results)
            self._save_results()
            raise
    
    def _save_results(self):
        """Save intermediate results to CSV"""
        df = pd.DataFrame(self.results_storage)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = f"checkpoints/optuna_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Sweep for DQN Unlearning")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str,
                       default='E:\\Kuliah\\Kuliah\\Kuliah\\PRODI\\Semester 7\\ProSkripCode\\data_movie',
                       help='Path to data directory')
    parser.add_argument('--pilot_users', type=int, default=250,
                       help='Number of users for pilot')
    
    # Split ratios
    parser.add_argument('--retain_ratio', type=float, default=0.70,
                       help='Ratio of retain users')
    parser.add_argument('--forget_ratio', type=float, default=0.150,
                       help='Ratio of forget users')
    parser.add_argument('--test_ratio', type=float, default=0.150,
                       help='Ratio of test users')
    
    # Optuna arguments
    parser.add_argument('--n_trials', type=int, default=100,
                       help='Number of Optuna trials')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION FOR DQN UNLEARNING")
    print("="*80)
    
    # Check CUDA
    device = check_cuda_usage()
    
    # Validate split ratios
    total_ratio = args.retain_ratio + args.forget_ratio + args.test_ratio
    assert abs(total_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0! Got {total_ratio:.4f}"
    
    print(f"\nConfiguration:")
    print(f"  Pilot users: {args.pilot_users}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Number of trials: {args.n_trials}")
    print(f"\nSplit ratios:")
    print(f"  Retain: {args.retain_ratio*100:.1f}%")
    print(f"  Forget: {args.forget_ratio*100:.1f}%")
    print(f"  Test:   {args.test_ratio*100:.1f}%")
    
    # Set seed
    set_seed(args.seed)
    
    # Load data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    dataloaders, datasets = create_three_way_split_dataloaders(
        data_dir=args.data_dir,
        retain_ratio=args.retain_ratio,
        forget_ratio=args.forget_ratio,
        test_ratio=args.test_ratio,
        rating_threshold=4.0,
        batch_size=64,
        user_total=args.pilot_users,
        seed=args.seed
    )
    
    # Detect input dimension
    first_batch = next(iter(dataloaders['retain_train']))
    actual_input_dim = first_batch['input_features'].shape[1]
    print(f"\nInput dimension: {actual_input_dim}")
    
    # Create results storage
    results_storage = []
    
    # Create objective function
    objective = ObjectiveFunction(
        dataloaders=dataloaders,
        datasets=datasets,
        actual_input_dim=actual_input_dim,
        device=device,
        results_storage=results_storage
    )
    
    # Create Optuna study
    print("\n" + "="*80)
    print("STARTING OPTUNA OPTIMIZATION")
    print("="*80)
    print(f"Objective: Maximize (forget_drop - 2 * test_drop)")
    
    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1
        )
    )
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=args.n_trials,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"  Complete: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    print(f"  Pruned:   {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"  Failed:   {len([t for t in study.trials if t.state == TrialState.FAIL])}")
    
    if study.best_trial:
        print("\n" + "="*80)
        print("BEST TRIAL")
        print("="*80)
        
        print(f"  Trial number: {study.best_trial.number}")
        print(f"  Quality score: {study.best_trial.value:.4f}")
        print(f"\n  Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        # Find best trial in results
        best_result = [r for r in results_storage if r.get('trial') == study.best_trial.number]
        if best_result:
            best = best_result[0]
            print(f"\n  Performance:")
            print(f"    Model 1 (Baseline):")
            print(f"      Forget: {best['model1_forget_acc']:.2%}, Test: {best['model1_test_acc']:.2%}")
            print(f"    Model 2 (Full):")
            print(f"      Forget: {best['model2_forget_acc']:.2%}, Test: {best['model2_test_acc']:.2%}")
            print(f"    Model 3 (Unlearned):")
            print(f"      Forget: {best['model3_forget_acc']:.2%}, Test: {best['model3_test_acc']:.2%}")
            print(f"\n    Drops:")
            print(f"      Forget: {best['forget_drop']:.4f} ({best['forget_drop_pct']:.1f}%)")
            print(f"      Test:   {best['test_drop']:.4f} ({best['test_drop_pct']:.1f}%)")
    
    # Save final results
    df = pd.DataFrame(results_storage)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"checkpoints/optuna_final_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nFinal results saved to: {csv_path}")
    
    # Save study
    study_path = f"checkpoints/optuna_study_{timestamp}.pkl"
    import joblib
    joblib.dump(study, study_path)
    print(f"Study object saved to: {study_path}")
    
    # Print top 10 trials
    print("\n" + "="*80)
    print("TOP 10 TRIALS")
    print("="*80)
    
    complete_results = [r for r in results_storage if r.get('state') == 'COMPLETE']
    if complete_results:
        df_complete = pd.DataFrame(complete_results)
        df_sorted = df_complete.sort_values('quality', ascending=False)
        
        top10 = df_sorted.head(10)
        print(top10[[
            'trial', 'lambda_weight', 'unlearn_lr', 'train_lr',
            'train_epochs', 'unlearn_epochs',
            'forget_drop_pct', 'test_drop_pct', 'quality'
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
