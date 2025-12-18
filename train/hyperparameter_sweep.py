# train/hyperparameter_optuna.py

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

from dataset.dataloader import create_dataloaders, MovieLensRecommendationDataset
from model.dqn_model import RecommendationDQN, DQNWithTargetNetwork
from model.loss_function import (
    DQNRecommendationLoss, 
    DecrementalRLLoss,
    create_random_forget_samples
)
from eval.evaluate import RecommendationEvaluator


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_epoch_fast(dqn_manager, train_loader, optimizer, loss_fn, device):
    """Fast training epoch without verbose output"""
    dqn_manager.train_mode()
    online_net = dqn_manager.get_online_network()
    
    epoch_losses = []
    epoch_accuracies = []
    
    for batch in train_loader:
        inputs = batch['input_features'].to(device)
        labels = batch['label'].squeeze().to(device)
        rewards = batch['reward'].squeeze().to(device)
        
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
        train_loader,
        retain_test_dataset,
        forget_test_dataset,
        retain_test_loader,
        forget_users,
        actual_input_dim,
        device,
        results_storage
    ):
        self.train_loader = train_loader
        self.retain_test_dataset = retain_test_dataset
        self.forget_test_dataset = forget_test_dataset
        self.retain_test_loader = retain_test_loader
        self.forget_users = forget_users
        self.actual_input_dim = actual_input_dim
        self.device = device
        self.results_storage = results_storage
    
    def __call__(self, trial: optuna.Trial):
        """
        Objective function to maximize unlearning quality.
        Quality = forget_drop - 2 * retain_drop
        
        Higher is better (large forget drop, small retain drop)
        """
        
        # Sample hyperparameters
        lambda_weight = trial.suggest_float('lambda_weight', 0.01, 100.0, log=True)
        unlearn_lr = trial.suggest_float('unlearn_lr', 1e-5, 1e-3, log=True)
        train_lr = trial.suggest_float('train_lr', 1e-5, 5e-3, log=True)
        train_epochs = trial.suggest_int('train_epochs', 5, 20, step=2)
        unlearn_epochs = trial.suggest_int('unlearn_epochs', 5, 20, step=2)
        
        print(f"\n{'='*80}")
        print(f"Trial {trial.number}: lambda={lambda_weight:.3f}, unlearn_lr={unlearn_lr:.2e}, "
              f"train_lr={train_lr:.2e}, train_ep={train_epochs}, unlearn_ep={unlearn_epochs}")
        print(f"{'='*80}")
        
        try:
            # Phase 1: Training
            print(f"Training for {train_epochs} epochs...")
            dqn_manager = DQNWithTargetNetwork(
                input_dim=self.actual_input_dim,
                hidden_dims=[128, 256, 256, 128],
                dropout_rate=0.2,
                device=self.device,
                target_update_freq=1000,
                use_soft_update=False
            )
            
            online_net = dqn_manager.get_online_network()
            optimizer = optim.Adam(online_net.parameters(), lr=train_lr)
            loss_fn = DQNRecommendationLoss(rating_threshold=4.0)
            
            for epoch in range(train_epochs):
                train_loss, train_acc = train_epoch_fast(
                    dqn_manager, self.train_loader, optimizer, loss_fn, self.device
                )
                
                # Report intermediate value for pruning
                trial.report(train_acc, epoch)
                
                # Prune if training is not going well
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            print(f"Training done. Loss: {train_loss:.4f}, Acc: {train_acc:.2%}")
            
            # Evaluate before unlearning
            evaluator_before = RecommendationEvaluator(online_net, self.device)
            
            retain_before = evaluator_before.evaluate_classification(
                self.retain_test_dataset, batch_size=128
            )
            forget_before = evaluator_before.evaluate_classification(
                self.forget_test_dataset, batch_size=128
            )
            
            print(f"Before - Retain: {retain_before['accuracy']:.2%}, "
                  f"Forget: {forget_before['accuracy']:.2%}")
            
            # Phase 2: Unlearning
            print(f"Unlearning for {unlearn_epochs} epochs...")
            
            # Generate random samples
            random_samples = create_random_forget_samples(
                self.forget_test_dataset, 
                self.forget_users, 
                num_samples_per_user=50, 
                device=self.device
            )
            
            if len(random_samples['inputs']) == 0:
                print("ERROR: No forget samples!")
                raise optuna.TrialPruned()
            
            # Create unlearning model
            unlearning_model = copy.deepcopy(online_net)
            original_model = copy.deepcopy(online_net)
            original_model.eval()
            for param in original_model.parameters():
                param.requires_grad = False
            
            optimizer_unlearn = optim.Adam(
                unlearning_model.parameters(), 
                lr=unlearn_lr
            )
            loss_fn_unlearn = DecrementalRLLoss(lambda_weight=lambda_weight)
            
            for epoch in range(unlearn_epochs):
                unlearn_loss = unlearning_epoch_fast(
                    unlearning_model, original_model,
                    random_samples, self.retain_test_loader,
                    optimizer_unlearn, loss_fn_unlearn, self.device
                )
            
            print(f"Unlearning done. Loss: {unlearn_loss:.4f}")
            
            # Evaluate after unlearning
            evaluator_after = RecommendationEvaluator(unlearning_model, self.device)
            
            retain_after = evaluator_after.evaluate_classification(
                self.retain_test_dataset, batch_size=128
            )
            forget_after = evaluator_after.evaluate_classification(
                self.forget_test_dataset, batch_size=128
            )
            
            print(f"After - Retain: {retain_after['accuracy']:.2%}, "
                  f"Forget: {forget_after['accuracy']:.2%}")
            
            # Compute metrics
            retain_drop = retain_before['accuracy'] - retain_after['accuracy']
            forget_drop = forget_before['accuracy'] - forget_after['accuracy']
            
            # Quality metric: maximize forget drop, minimize retain drop
            quality = forget_drop - 2 * retain_drop
            
            print(f"\nResults:")
            print(f"  Retain drop: {retain_drop:.2%}")
            print(f"  Forget drop: {forget_drop:.2%}")
            print(f"  Quality: {quality:.4f}")
            
            # Store results
            results = {
                'trial': trial.number,
                'lambda_weight': lambda_weight,
                'unlearn_lr': unlearn_lr,
                'train_lr': train_lr,
                'train_epochs': train_epochs,
                'unlearn_epochs': unlearn_epochs,
                
                'retain_acc_before': retain_before['accuracy'],
                'retain_acc_after': retain_after['accuracy'],
                'forget_acc_before': forget_before['accuracy'],
                'forget_acc_after': forget_after['accuracy'],
                
                'retain_drop': retain_drop,
                'forget_drop': forget_drop,
                'retain_drop_pct': retain_drop / retain_before['accuracy'] * 100,
                'forget_drop_pct': forget_drop / forget_before['accuracy'] * 100,
                
                'quality': quality,
                'state': 'COMPLETE'
            }
            
            self.results_storage.append(results)
            
            # Save intermediate results
            self._save_results()
            
            return quality
        
        except optuna.TrialPruned:
            print(f"Trial {trial.number} pruned!")
            
            # Store pruned trial
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
            
            # Store failed trial
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
    """Main Optuna optimization"""
    
    # Configuration
    PILOT_USERS = 100
    N_TRIALS = 100  # Number of trials to run
    SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = 'E:\\Kuliah\\Kuliah\\Kuliah\\PRODI\\Semester 7\\ProSkripCode\\data_movie'
    
    print("\n" + "="*80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION FOR DQN UNLEARNING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Pilot users: {PILOT_USERS}")
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print(f"  Number of trials: {N_TRIALS}")
    
    print(f"\nHyperparameter search space:")
    print(f"  lambda_weight: [0.1, 10.0] (log scale)")
    print(f"  unlearn_lr: [1e-5, 1e-3] (log scale)")
    print(f"  train_lr: [1e-5, 5e-3] (log scale)")
    print(f"  train_epochs: [5, 25] (int)")
    print(f"  unlearn_epochs: [5, 40] (int)")
    
    print(f"\nObjective: Maximize quality = forget_drop - 2 * retain_drop")
    
    # Set seed
    set_seed(SEED)
    
    # Load data once
    print("\n" + "="*80)
    print("LOADING DATA (ONE TIME)")
    print("="*80)
    
    dataloaders, train_dataset = create_dataloaders(
        data_dir=DATA_DIR,
        forget_ratio=0.1,
        rating_threshold=4.0,
        batch_size=64,
        user_total=PILOT_USERS,
        seed=SEED
    )
    
    train_loader = dataloaders['train']
    retain_test_loader = dataloaders['retain_test']
    
    forget_users = train_dataset.forget_users
    
    # Detect input dimension
    first_batch = next(iter(train_loader))
    actual_input_dim = first_batch['input_features'].shape[1]
    
    # Create test datasets
    retain_test_dataset = MovieLensRecommendationDataset(
        data_dir=DATA_DIR,
        forget_ratio=0.1,
        rating_threshold=4.0,
        user_total=PILOT_USERS,
        mode='retain',
        split='test',
        seed=SEED
    )
    
    forget_test_dataset = MovieLensRecommendationDataset(
        data_dir=DATA_DIR,
        forget_ratio=0.1,
        rating_threshold=4.0,
        user_total=PILOT_USERS,
        mode='forget',
        split='test',
        seed=SEED
    )
    
    print(f"\nData loaded successfully!")
    print(f"  Input dimension: {actual_input_dim}")
    print(f"  Train samples: {len(train_dataset):,}")
    
    # Create results storage
    results_storage = []
    
    # Create objective function
    objective = ObjectiveFunction(
        train_loader=train_loader,
        retain_test_dataset=retain_test_dataset,
        forget_test_dataset=forget_test_dataset,
        retain_test_loader=retain_test_loader,
        forget_users=forget_users,
        actual_input_dim=actual_input_dim,
        device=DEVICE,
        results_storage=results_storage
    )
    
    # Create Optuna study
    print("\n" + "="*80)
    print("CREATING OPTUNA STUDY")
    print("="*80)
    
    study = optuna.create_study(
        direction='maximize',  # Maximize quality
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5,
            interval_steps=1
        )
    )
    
    # Run optimization
    print("\n" + "="*80)
    print("STARTING OPTIMIZATION")
    print("="*80)
    
    study.optimize(
        objective, 
        n_trials=N_TRIALS,
        show_progress_bar=True,
        catch=(Exception,)
    )
    
    # Results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == TrialState.COMPLETE])}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"Number of failed trials: {len([t for t in study.trials if t.state == TrialState.FAIL])}")
    
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
            print(f"    Retain accuracy: {best['retain_acc_before']:.2%} → {best['retain_acc_after']:.2%} "
                  f"(drop: {best['retain_drop_pct']:.1f}%)")
            print(f"    Forget accuracy: {best['forget_acc_before']:.2%} → {best['forget_acc_after']:.2%} "
                  f"(drop: {best['forget_drop_pct']:.1f}%)")
    
    # Save final results
    df = pd.DataFrame(results_storage)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"checkpoints/optuna_final_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Final results saved to: {csv_path}")
    
    # Save study
    study_path = f"checkpoints/optuna_study_{timestamp}.pkl"
    import joblib
    joblib.dump(study, study_path)
    print(f"  Study object saved to: {study_path}")
    
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
            'retain_drop_pct', 'forget_drop_pct', 'quality'
        ]].to_string(index=False))


if __name__ == "__main__":
    main()
