# dataset/dataloader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle


class MovieLensRLDataset(Dataset):
    """
    MovieLens dataset for DQN-based recommendation with unlearning support.
    
    Splits users into retain/forget sets and prepares sequential RL episodes.
    """
    
    def __init__(
        self,
        data_dir: str,
        forget_ratio: float = 0.1,
        use_genome: bool = False,
        state_size: int = 50,
        min_ratings: int = 20,
        seed: int = 42,
        mode: str = 'train',  # 'train', 'retain', or 'forget'
        eval_split: float = 0.8,
        user_total: Optional[int] = None  # <-- ADD THIS
    ):
        """
        Args:
            data_dir: Path to MovieLens data directory
            forget_ratio: Ratio of users to put in forget set (for unlearning)
            use_genome: Whether to include genome tag features
            state_size: Maximum number of movies in state history
            min_ratings: Minimum ratings per user to include
            seed: Random seed for reproducibility
            mode: 'train' (all users), 'retain' (only retain users), 'forget' (only forget users)
            eval_split: Ratio to split each user's sequence into train/eval
            user_total: Total number of users to use (for pilot testing, None = all users)  # <-- ADD THIS
        """
        self.data_dir = Path(data_dir)
        self.forget_ratio = forget_ratio
        self.use_genome = use_genome
        self.state_size = state_size
        self.min_ratings = min_ratings
        self.seed = seed
        self.mode = mode
        self.eval_split = eval_split
        self.user_total = user_total  # <-- ADD THIS
        
        np.random.seed(seed)
        
        # Load data
        print("Loading MovieLens data...")
        self._load_data()
        
        # Split users into retain/forget sets
        print(f"Splitting users (forget_ratio={forget_ratio}, user_total={user_total})...")  # <-- UPDATE THIS
        self._split_users()
        
        # Prepare sequential episodes
        print(f"Preparing {mode} episodes...")
        self._prepare_episodes()

        # After _prepare_episodes(), add:
        self._build_user_action_masks()
        
        print(f"Dataset ready: {len(self.episodes)} episodes from {len(self.user_ids)} users")

    def _build_user_action_masks(self):
        """Build action masks for each user (movies they've rated)"""
        print("Building user action masks...")
        self.user_action_masks = {}
        
        for user_id in self.user_ids:
            # Get all movies this user has rated
            user_movies = self.ratings_df[
                self.ratings_df['userId'] == user_id
            ]['movieId'].unique()
            
            # Create mask: True for movies user has rated
            mask = torch.zeros(self.num_movies, dtype=torch.bool)
            for movie_id in user_movies:
                if movie_id in self.movie_id_to_idx:
                    movie_idx = self.movie_id_to_idx[movie_id]
                    mask[movie_idx] = True
            
            self.user_action_masks[user_id] = mask
        
        print(f"Action masks built for {len(self.user_action_masks)} users")

    def get_user_action_mask(self, user_id: int) -> torch.Tensor:
        """Get action mask for a specific user"""
        return self.user_action_masks.get(user_id, torch.ones(self.num_movies, dtype=torch.bool))
    
    def _load_data(self):
        """Load all necessary CSV files"""
        # Load ratings (main data)
        self.ratings_df = pd.read_csv(self.data_dir / 'rating.csv')
        self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'])
        
        # Load movies
        self.movies_df = pd.read_csv(self.data_dir / 'movie.csv')
        
        # Create movie ID mapping (for action space)
        self.movie_ids = sorted(self.ratings_df['movieId'].unique())
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.idx_to_movie_id = {idx: mid for mid, idx in self.movie_id_to_idx.items()}
        self.num_movies = len(self.movie_ids)
        
        # Load genome features if requested
        if self.use_genome:
            print("Loading genome features...")
            genome_scores = pd.read_csv(self.data_dir / 'genome_scores.csv')
            genome_tags = pd.read_csv(self.data_dir / 'genome_tags.csv')
            
            # Create movie-tag matrix (movies x tags)
            self.genome_matrix = genome_scores.pivot(
                index='movieId', 
                columns='tagId', 
                values='relevance'
            ).fillna(0)
            
            self.num_tags = len(genome_tags)
            print(f"Genome features loaded: {self.genome_matrix.shape[0]} movies x {self.num_tags} tags")
        else:
            self.genome_matrix = None
            self.num_tags = 0
    
    def _split_users(self):
        """Split users into retain and forget sets"""
        # Filter users with minimum ratings
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings].index.tolist()
        
        # Limit total users if specified (for pilot testing)
        if self.user_total is not None:
            np.random.shuffle(valid_users)
            valid_users = valid_users[:self.user_total]
            print(f"Pilot mode: Limited to {len(valid_users)} users")
        
        # Randomly split users
        np.random.shuffle(valid_users)
        forget_size = int(len(valid_users) * self.forget_ratio)
        
        self.forget_users = set(valid_users[:forget_size])
        self.retain_users = set(valid_users[forget_size:])
        
        print(f"Users split: {len(self.retain_users)} retain, {len(self.forget_users)} forget")
        
        # Select users based on mode
        if self.mode == 'train':
            self.user_ids = valid_users  # All users
        elif self.mode == 'retain':
            self.user_ids = list(self.retain_users)
        elif self.mode == 'forget':
            self.user_ids = list(self.forget_users)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    
    def _prepare_episodes(self):
        """
        Prepare sequential RL episodes for each user.
        
        Each episode is a trajectory: (s_0, a_0, r_0, s_1), (s_1, a_1, r_1, s_2), ...
        """
        self.episodes = []
        
        for user_id in self.user_ids:
            # Get user's ratings sorted by timestamp
            user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id].sort_values('timestamp')
            
            if len(user_ratings) < self.min_ratings:
                continue
            
            # Split into train/eval based on eval_split
            split_idx = int(len(user_ratings) * self.eval_split)
            
            # For training mode, use first eval_split% of sequence
            # For evaluation, use the remaining part
            if self.mode in ['train', 'retain', 'forget']:
                user_ratings = user_ratings.iloc[:split_idx] if split_idx > 0 else user_ratings
            
            # Create episodes from sequential ratings
            # State: history of (movieId, rating) pairs
            # Action: next movie to recommend
            # Reward: rating of that movie
            
            movie_sequence = user_ratings['movieId'].values
            rating_sequence = user_ratings['rating'].values
            
            # Create sliding window episodes
            for i in range(1, len(movie_sequence)):
                # State: all movies watched before position i (up to state_size)
                start_idx = max(0, i - self.state_size)
                state_movies = movie_sequence[start_idx:i]
                state_ratings = rating_sequence[start_idx:i]
                
                # Action: movie at position i
                action_movie = movie_sequence[i]
                
                # Reward: rating at position i
                reward = rating_sequence[i]
                
                # Next state: all movies watched up to position i (up to state_size)
                next_start_idx = max(0, i + 1 - self.state_size)
                next_state_movies = movie_sequence[next_start_idx:i + 1]
                next_state_ratings = rating_sequence[next_start_idx:i + 1]
                
                # Only include action if it's in our movie vocabulary
                if action_movie in self.movie_id_to_idx:
                    self.episodes.append({
                        'user_id': user_id,
                        'state_movies': state_movies,
                        'state_ratings': state_ratings,
                        'action': self.movie_id_to_idx[action_movie],  # Convert to index
                        'reward': reward,
                        'next_state_movies': next_state_movies,
                        'next_state_ratings': next_state_ratings,
                        'done': (i == len(movie_sequence) - 1)  # Terminal state
                    })
    
    def _encode_state(self, movies, ratings):
        """
        Encode state as fixed-size vector.
        
        Returns:
            State tensor of shape (state_size * feature_dim)
            where feature_dim = 1 (movie_id) + 1 (rating) + num_tags (if genome enabled)
        """
        # Pad sequences to state_size
        padded_movies = np.zeros(self.state_size, dtype=np.int32)
        padded_ratings = np.zeros(self.state_size, dtype=np.float32)
        
        seq_len = len(movies)
        if seq_len > 0:
            padded_movies[-seq_len:] = movies
            padded_ratings[-seq_len:] = ratings
        
        # Base features: movie indices and ratings
        state_features = []
        
        for i in range(self.state_size):
            movie_id = self.idx_to_movie_id.get(padded_movies[i], 0)
            
            # Movie index (normalized)
            state_features.append(padded_movies[i] / self.num_movies)
            
            # Rating (normalized to [0, 1])
            state_features.append(padded_ratings[i] / 5.0)
            
            # Genome features if enabled
            if self.use_genome and movie_id in self.genome_matrix.index:
                genome_vec = self.genome_matrix.loc[movie_id].values
                state_features.extend(genome_vec)
            elif self.use_genome:
                # Movie not in genome matrix, use zeros
                state_features.extend(np.zeros(self.num_tags))
        
        return torch.FloatTensor(state_features)
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        """
        Returns:
            state: Encoded state tensor
            action: Action index (movie to recommend)
            reward: Reward value (rating)
            next_state: Encoded next state tensor
            done: Whether this is a terminal state
            user_id: User ID for tracking
        """
        episode = self.episodes[idx]
        
        state = self._encode_state(episode['state_movies'], episode['state_ratings'])
        next_state = self._encode_state(episode['next_state_movies'], episode['next_state_ratings'])
        
        return {
            'state': state,
            'action': episode['action'],
            'reward': torch.FloatTensor([episode['reward']]),
            'next_state': next_state,
            'done': torch.FloatTensor([1.0 if episode['done'] else 0.0]),
            'user_id': episode['user_id']
        }
    
    def get_state_dim(self):
        """Return state dimension"""
        feature_dim = 2  # movie_idx + rating
        if self.use_genome:
            feature_dim += self.num_tags
        return self.state_size * feature_dim
    
    def get_action_dim(self):
        """Return action space size (number of movies)"""
        return self.num_movies
    
    def save_split(self, filepath):
        """Save retain/forget user split for reproducibility"""
        split_data = {
            'retain_users': list(self.retain_users),
            'forget_users': list(self.forget_users),
            'movie_id_to_idx': self.movie_id_to_idx,
            'seed': self.seed
        }
        with open(filepath, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"Split saved to {filepath}")
    
    def load_split(self, filepath):
        """Load retain/forget user split"""
        with open(filepath, 'rb') as f:
            split_data = pickle.load(f)
        self.retain_users = set(split_data['retain_users'])
        self.forget_users = set(split_data['forget_users'])
        print(f"Split loaded from {filepath}")

def create_dataloaders(
    data_dir: str,
    forget_ratio: float = 0.1,
    use_genome: bool = False,
    state_size: int = 50,
    batch_size: int = 32,
    user_total: Optional[int] = None,  # <-- ADD THIS
    **kwargs
):
    """
    Convenience function to create train/retain/forget dataloaders.
    
    Args:
        user_total: Total number of users to use (None = all users)  # <-- ADD THIS
    
    Returns:
        dict with keys: 'train', 'retain', 'forget', 'eval_retain', 'eval_forget'
    """
    from torch.utils.data import DataLoader
    
    # Training datasets
    train_dataset = MovieLensRLDataset(
        data_dir, forget_ratio, use_genome, state_size, 
        mode='train', user_total=user_total, **kwargs  # <-- ADD user_total
    )
    
    retain_dataset = MovieLensRLDataset(
        data_dir, forget_ratio, use_genome, state_size, 
        mode='retain', user_total=user_total, **kwargs  # <-- ADD user_total
    )
    
    forget_dataset = MovieLensRLDataset(
        data_dir, forget_ratio, use_genome, state_size, 
        mode='forget', user_total=user_total, **kwargs  # <-- ADD user_total
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'retain': DataLoader(retain_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'forget': DataLoader(forget_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    
    return dataloaders, train_dataset


# Example usage
if __name__ == "__main__":
    data_dir = "E:\\Kuliah\\Kuliah\\Kuliah\\PRODI\\Semester 7\\ProSkripCode\\data_movie"
    
    # Test with limited users for debugging
    print("=" * 50)
    print("Testing with user_total=100 (small dataset)")
    print("=" * 50)
    dataloaders, dataset = create_dataloaders(
        data_dir=data_dir,
        forget_ratio=0.1,
        use_genome=False,
        state_size=50,
        user_total=100,  # Only use 100 users
        batch_size=32
    )
    
    print(f"\nState dimension: {dataset.get_state_dim()}")
    print(f"Action dimension: {dataset.get_action_dim()}")
    print(f"Number of users: {len(dataset.user_ids)}")
    print(f"  Retain users: {len(dataset.retain_users)}")
    print(f"  Forget users: {len(dataset.forget_users)}")
    

