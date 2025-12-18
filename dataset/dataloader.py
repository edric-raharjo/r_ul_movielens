# dataset/dataloader.py

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle
import re


class MovieLensRecommendationDataset(Dataset):
    """
    MovieLens dataset for binary recommendation with DQN.
    
    Each sample: Should we recommend this movie to this user?
    - Input: User state (22-dim) + Candidate movie (22-dim) = 44-dim
    - Output: Binary label (0=no, 1=yes) + Reward (actual rating)
    """
    
    def __init__(
        self,
        data_dir: str,
        forget_ratio: float = 0.1,
        rating_threshold: float = 4.0,
        min_ratings: int = 20,
        train_ratio: float = 0.8,  # Temporal split
        user_total: Optional[int] = None,
        mode: str = 'train',  # 'train', 'retain', 'forget'
        split: str = 'train',  # 'train' or 'test'
        seed: int = 42
    ):
        """
        Args:
            data_dir: Path to MovieLens data directory
            forget_ratio: Ratio of users to put in forget set
            rating_threshold: Ratings >= this are positive (recommend)
            min_ratings: Minimum ratings per user to include
            train_ratio: Ratio for temporal train/test split per user
            user_total: Total users to use (None = all)
            mode: 'train' (all users), 'retain' (retain only), 'forget' (forget only)
            split: 'train' or 'test' (temporal split within each user)
            seed: Random seed
        """
        self.data_dir = Path(data_dir)
        self.forget_ratio = forget_ratio
        self.rating_threshold = rating_threshold
        self.min_ratings = min_ratings
        self.train_ratio = train_ratio
        self.user_total = user_total
        self.mode = mode
        self.split = split
        self.seed = seed
        
        np.random.seed(seed)
        
        print(f"\n{'='*70}")
        print(f"Loading MovieLens Dataset")
        print(f"Mode: {mode} | Split: {split}")
        print(f"{'='*70}")
        
        # Load data
        self._load_data()
        
        # Extract genres
        self._extract_genres()
        
        # Split users
        self._split_users()
        
        # Prepare samples
        self._prepare_samples()
        
        print(f"Dataset ready: {len(self.samples)} samples from {len(self.current_users)} users")
        print(f"{'='*70}\n")
    
    def _load_data(self):
        """Load ratings and movies"""
        print("Loading ratings.csv...")
        self.ratings_df = pd.read_csv(self.data_dir / 'rating.csv')
        self.ratings_df['timestamp'] = pd.to_datetime(self.ratings_df['timestamp'])
        
        print("Loading movies.csv...")
        self.movies_df = pd.read_csv(self.data_dir / 'movie.csv')
        
        print(f"  Ratings: {len(self.ratings_df):,}")
        print(f"  Movies: {len(self.movies_df):,}")
        print(f"  Users: {self.ratings_df['userId'].nunique():,}")
    
    def _extract_genres(self):
        """Extract and encode genres"""
        print("\nExtracting genres...")
        
        # Get all unique genres
        all_genres = set()
        for genres_str in self.movies_df['genres']:
            if isinstance(genres_str, str) and genres_str != '(no genres listed)':
                all_genres.update(genres_str.split('|'))
        
        self.genre_list = sorted(list(all_genres))
        self.genre_to_idx = {g: i for i, g in enumerate(self.genre_list)}
        self.num_genres = len(self.genre_list)
        
        print(f"  Found {self.num_genres} unique genres: {self.genre_list[:5]}...")
        
        # Encode genres for each movie
        self.movies_df['genre_vector'] = self.movies_df['genres'].apply(
            self._encode_genre
        )
        
        # Extract release year from title
        print("\nExtracting release years...")
        self.movies_df['release_year'] = self.movies_df['title'].apply(
            self._extract_year
        )
        
        # Compute average rating per movie
        print("Computing average ratings per movie...")
        movie_avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean()
        self.movies_df = self.movies_df.merge(
            movie_avg_ratings.rename('avg_rating'),
            left_on='movieId',
            right_index=True,
            how='left'
        )
        self.movies_df['avg_rating'].fillna(self.movies_df['avg_rating'].mean(), inplace=True)
        
        # Create movie features lookup
        self.movie_features = {}
        for _, row in self.movies_df.iterrows():
            self.movie_features[row['movieId']] = {
                'year': row['release_year'],
                'genre_vector': row['genre_vector'],
                'avg_rating': row['avg_rating']
            }
    
    def _encode_genre(self, genre_string: str) -> np.ndarray:
        """
        Encode genre string to probability distribution.
        
        Args:
            genre_string: e.g., "Action|Romance"
            
        Returns:
            Probability distribution over genres (num_genres,)
            e.g., [0.5, 0.0, ..., 0.5, ...] if Action and Romance
        """
        if not isinstance(genre_string, str) or genre_string == '(no genres listed)':
            return np.zeros(self.num_genres)
        
        genres = genre_string.split('|')
        multi_hot = np.zeros(self.num_genres)
        
        for g in genres:
            if g in self.genre_to_idx:
                multi_hot[self.genre_to_idx[g]] = 1.0
        
        # Normalize to probability distribution
        if multi_hot.sum() > 0:
            multi_hot = multi_hot / multi_hot.sum()
        
        return multi_hot
    
    def _extract_year(self, title: str) -> float:
        """
        Extract release year from movie title.
        
        Args:
            title: e.g., "Toy Story (1995)"
            
        Returns:
            Normalized year (year - 1900) / 100
        """
        match = re.search(r'\((\d{4})\)', title)
        if match:
            year = int(match.group(1))
            return (year - 1900) / 100.0
        else:
            # Use median year if not found
            return (1995 - 1900) / 100.0  # Default to 1995
    
    def _split_users(self):
        """Split users into retain and forget sets"""
        print("\nSplitting users...")
        
        # Filter users with minimum ratings
        user_counts = self.ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings].index.tolist()
        
        # Limit total users if specified
        if self.user_total is not None:
            np.random.shuffle(valid_users)
            valid_users = valid_users[:self.user_total]
            print(f"  Pilot mode: Limited to {len(valid_users)} users")
        
        # Randomly split users
        np.random.shuffle(valid_users)
        forget_size = int(len(valid_users) * self.forget_ratio)
        
        self.forget_users = set(valid_users[:forget_size])
        self.retain_users = set(valid_users[forget_size:])
        
        print(f"  Users split: {len(self.retain_users)} retain, {len(self.forget_users)} forget")
        
        # Select users based on mode
        if self.mode == 'train':
            self.current_users = valid_users  # All users
        elif self.mode == 'retain':
            self.current_users = list(self.retain_users)
        elif self.mode == 'forget':
            self.current_users = list(self.forget_users)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        print(f"  Current mode '{self.mode}': {len(self.current_users)} users")
    
    def _prepare_samples(self):
        """
        Prepare training/test samples.
        
        Each sample: (user_state, candidate_movie, label, reward)
        """
        print(f"\nPreparing {self.split} samples...")
        
        self.samples = []
        users_with_samples = 0
        
        for user_id in self.current_users:
            # Get user's ratings sorted by timestamp (temporal)
            user_ratings = self.ratings_df[
                self.ratings_df['userId'] == user_id
            ].sort_values('timestamp')
            
            if len(user_ratings) < self.min_ratings:
                continue
            
            # Temporal split
            split_idx = int(len(user_ratings) * self.train_ratio)
            
            if self.split == 'train':
                user_data = user_ratings.iloc[:split_idx]
            else:  # test
                user_data = user_ratings.iloc[split_idx:]
            
            if len(user_data) == 0:
                continue
            
            # Compute user state from training data only (to avoid data leakage)
            user_train_data = user_ratings.iloc[:split_idx]
            user_state = self._compute_user_state(user_train_data, user_id)
            
            # Create samples for each rating in current split
            for _, row in user_data.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                
                # Skip if movie features not available
                if movie_id not in self.movie_features:
                    continue
                
                # Get candidate movie features
                candidate_features = self._get_candidate_features(movie_id)
                
                # Label: 1 if recommend, 0 if not
                label = 1 if rating >= self.rating_threshold else 0
                
                self.samples.append({
                    'user_id': user_id,
                    'user_state': user_state,
                    'candidate_features': candidate_features,
                    'movie_id': movie_id,
                    'label': label,
                    'reward': rating
                })
            
            users_with_samples += 1
        
        print(f"  Created {len(self.samples):,} samples from {users_with_samples} users")
        
        # Class distribution
        num_positive = sum(1 for s in self.samples if s['label'] == 1)
        num_negative = len(self.samples) - num_positive
        print(f"  Positive samples (rating >= {self.rating_threshold}): {num_positive:,} ({num_positive/len(self.samples)*100:.1f}%)")
        print(f"  Negative samples (rating < {self.rating_threshold}): {num_negative:,} ({num_negative/len(self.samples)*100:.1f}%)")
    
    def _compute_user_state(self, user_ratings: pd.DataFrame, user_id: int) -> np.ndarray:
        """
        Compute user state from their rating history.
        
        Returns:
            User state vector (22-dim):
            [avg_year(1), avg_rating(1), genre_distribution(20)]
        """
        if len(user_ratings) == 0:
            # Return default state
            return np.zeros(22)
        
        # 1. Average release year
        years = []
        for movie_id in user_ratings['movieId']:
            if movie_id in self.movie_features:
                years.append(self.movie_features[movie_id]['year'])
        avg_year = np.mean(years) if years else 0.5
        
        # 2. Average rating given by user
        avg_rating = user_ratings['rating'].mean() / 5.0
        
        # 3. Genre distribution
        genre_vectors = []
        for movie_id in user_ratings['movieId']:
            if movie_id in self.movie_features:
                genre_vectors.append(self.movie_features[movie_id]['genre_vector'])
        
        if genre_vectors:
            # Sum all genre vectors
            genre_sum = np.sum(genre_vectors, axis=0)
            # Normalize to probability distribution
            genre_dist = genre_sum / genre_sum.sum() if genre_sum.sum() > 0 else genre_sum
        else:
            genre_dist = np.zeros(self.num_genres)
        
        # Concatenate: [avg_year(1), avg_rating(1), genre_dist(20)]
        user_state = np.concatenate([
            [avg_year],
            [avg_rating],
            genre_dist
        ])
        
        return user_state.astype(np.float32)
    
    def _get_candidate_features(self, movie_id: int) -> np.ndarray:
        """
        Get candidate movie features.
        
        Returns:
            Candidate features (22-dim):
            [year(1), avg_rating(1), genre_distribution(20)]
        """
        features = self.movie_features[movie_id]
        
        # Normalize avg_rating
        avg_rating_norm = features['avg_rating'] / 5.0
        
        # Concatenate: [year(1), avg_rating(1), genre_dist(20)]
        candidate_features = np.concatenate([
            [features['year']],
            [avg_rating_norm],
            features['genre_vector']
        ])
        
        return candidate_features.astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            Dictionary with training sample
        """
        sample = self.samples[idx]
        
        user_state = torch.FloatTensor(sample['user_state'])
        candidate = torch.FloatTensor(sample['candidate_features'])
        
        # DEBUG: Print dimensions on first call
        if not hasattr(self, '_debug_printed'):
            print(f"\n[DEBUG] Dimensions:")
            print(f"  User state: {user_state.shape}")
            print(f"  Candidate: {candidate.shape}")
            print(f"  Num genres: {self.num_genres}")
            self._debug_printed = True
        
        # Concatenate for network input
        input_features = torch.cat([user_state, candidate])
        
        return {
            'user_id': sample['user_id'],
            'user_state': user_state,
            'candidate_features': candidate,
            'input_features': input_features,
            'movie_id': sample['movie_id'],
            'label': torch.LongTensor([sample['label']]),
            'reward': torch.FloatTensor([sample['reward']])
        }
    
    def get_all_movie_features(self) -> Dict[int, np.ndarray]:
        """
        Get features for all movies (for evaluation).
        
        Returns:
            Dict mapping movieId -> features (22-dim)
        """
        return {
            movie_id: self._get_candidate_features(movie_id)
            for movie_id in self.movie_features.keys()
        }
    
    def save_split(self, filepath: str):
        """Save user split for reproducibility"""
        split_data = {
            'retain_users': list(self.retain_users),
            'forget_users': list(self.forget_users),
            'genre_list': self.genre_list,
            'seed': self.seed
        }
        with open(filepath, 'wb') as f:
            pickle.dump(split_data, f)
        print(f"Split saved to {filepath}")


def create_dataloaders(
    data_dir: str,
    forget_ratio: float = 0.1,
    rating_threshold: float = 4.0,
    batch_size: int = 64,
    user_total: Optional[int] = None,
    **kwargs
) -> Tuple[Dict[str, DataLoader], MovieLensRecommendationDataset]:
    """
    Create train/test dataloaders for retain and forget sets.
    
    Returns:
        (dataloaders_dict, train_dataset)
    """
    # Training dataset (all users, train split)
    train_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=forget_ratio,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='train',
        split='train',
        **kwargs
    )
    
    # Test datasets
    retain_test_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=forget_ratio,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='retain',
        split='test',
        **kwargs
    )
    
    forget_test_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=forget_ratio,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='forget',
        split='test',
        **kwargs
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        'retain_test': DataLoader(retain_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
        'forget_test': DataLoader(forget_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0),
    }
    
    return dataloaders, train_dataset


# Test code
if __name__ == "__main__":
    data_dir = "E:\\Kuliah\\Kuliah\\Kuliah\\PRODI\\Semester 7\\ProSkripCode\\data_movie"
    
    print("Testing MovieLens Recommendation Dataset...")
    
    # Create dataloaders
    dataloaders, train_dataset = create_dataloaders(
        data_dir=data_dir,
        forget_ratio=0.1,
        rating_threshold=4.0,
        batch_size=32,
        user_total=50  # Pilot mode
    )
    
    print(f"\n{'='*70}")
    print("Dataset Statistics")
    print(f"{'='*70}")
    print(f"Input dimensions: {train_dataset.samples[0]['user_state'].shape[0] + train_dataset.samples[0]['candidate_features'].shape[0]}")
    print(f"Number of genres: {train_dataset.num_genres}")
    print(f"Genres: {train_dataset.genre_list}")
    
    # Test a batch
    print(f"\n{'='*70}")
    print("Testing Batch")
    print(f"{'='*70}")
    batch = next(iter(dataloaders['train']))
    print(f"Batch keys: {batch.keys()}")
    print(f"Input features shape: {batch['input_features'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
    print(f"Labels distribution: {batch['label'].float().mean().item():.2%} positive")
    
    print("\n✅ Dataloader test passed!")
