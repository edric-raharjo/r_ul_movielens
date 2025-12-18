# dataset/dataloader.py - OPTIMIZED VERSION

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import pickle
import re
from tqdm import tqdm


class MovieLensRecommendationDataset(Dataset):
    """
    MovieLens dataset for binary movie recommendation with DQN.
    OPTIMIZED for large datasets.
    """
    
    def __init__(
        self,
        data_dir: str,
        forget_ratio: float = 0.1,
        rating_threshold: float = 4.0,
        min_ratings: int = 20,
        train_ratio: float = 0.8,
        user_total: Optional[int] = None,
        mode: str = 'train',
        split: str = 'train',
        seed: int = 42
    ):
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
        print(f"Mode: {mode} | Split: {split} | Pilot users: {user_total}")
        print(f"{'='*70}")
        
        # Load and filter data FIRST
        self._load_and_filter_data()
        
        # Extract genres and movie features
        self._extract_genres()
        
        # Split users
        self._split_users()
        
        # Precompute user states (OPTIMIZATION)
        self._precompute_user_states()
        
        # Prepare samples
        self._prepare_samples()
        
        print(f"Dataset ready: {len(self.samples)} samples from {len(self.current_users)} users")
        print(f"{'='*70}\n")
    
    def _load_and_filter_data(self):
        """Load and immediately filter data to relevant users only"""
        print("Loading ratings.csv...")
        ratings_df = pd.read_csv(self.data_dir / 'rating.csv')
        ratings_df['timestamp'] = pd.to_datetime(ratings_df['timestamp'])
        
        print(f"  Total ratings: {len(ratings_df):,}")
        print(f"  Total users: {ratings_df['userId'].nunique():,}")
        
        # Filter users with minimum ratings FIRST
        print(f"\nFiltering users with >= {self.min_ratings} ratings...")
        user_counts = ratings_df['userId'].value_counts()
        valid_users = user_counts[user_counts >= self.min_ratings].index.tolist()
        print(f"  Valid users: {len(valid_users):,}")
        
        # Sample users if pilot mode
        if self.user_total is not None:
            np.random.shuffle(valid_users)
            valid_users = valid_users[:self.user_total]
            print(f"  [PILOT MODE] Sampled {len(valid_users)} users")
        
        # Filter ratings to only include selected users
        print(f"Filtering ratings to selected users...")
        self.ratings_df = ratings_df[ratings_df['userId'].isin(valid_users)].copy()
        del ratings_df  # Free memory
        
        print(f"  Filtered ratings: {len(self.ratings_df):,}")
        
        # Store valid users for later
        self.valid_users = valid_users
        
        # Load movies
        print("\nLoading movies.csv...")
        self.movies_df = pd.read_csv(self.data_dir / 'movie.csv')
        print(f"  Movies: {len(self.movies_df):,}")
    
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
        
        print(f"  Found {self.num_genres} unique genres")
        
        # Encode genres for each movie
        print("Encoding movie genres...")
        self.movies_df['genre_vector'] = self.movies_df['genres'].apply(
            self._encode_genre
        )
        
        # Extract release year
        print("Extracting release years...")
        self.movies_df['release_year'] = self.movies_df['title'].apply(
            self._extract_year
        )
        
        # Compute average rating per movie (only for filtered data)
        print("Computing average ratings...")
        movie_avg_ratings = self.ratings_df.groupby('movieId')['rating'].mean()
        self.movies_df = self.movies_df.merge(
            movie_avg_ratings.rename('avg_rating'),
            left_on='movieId',
            right_index=True,
            how='left'
        )
        # Fill NaN with global mean
        global_mean = self.ratings_df['rating'].mean()
        self.movies_df['avg_rating'].fillna(global_mean, inplace=True)
        
        # Create movie features lookup (only for movies in filtered data)
        print("Creating movie features lookup...")
        relevant_movies = self.ratings_df['movieId'].unique()
        self.movies_df_filtered = self.movies_df[
            self.movies_df['movieId'].isin(relevant_movies)
        ].copy()
        
        self.movie_features = {}
        for _, row in self.movies_df_filtered.iterrows():
            self.movie_features[row['movieId']] = {
                'year': row['release_year'],
                'genre_vector': row['genre_vector'],
                'avg_rating': row['avg_rating']
            }
        
        print(f"  Movie features cached: {len(self.movie_features)} movies")
    
    def _encode_genre(self, genre_string: str) -> np.ndarray:
        """Encode genre string to probability distribution"""
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
        """Extract release year from movie title"""
        match = re.search(r'\((\d{4})\)', title)
        if match:
            year = int(match.group(1))
            return (year - 1900) / 100.0
        else:
            return (1995 - 1900) / 100.0  # Default
    
    def _split_users(self):
        """Split users into retain and forget sets"""
        print("\nSplitting users into retain/forget sets...")
        
        # Use pre-filtered valid users
        valid_users = self.valid_users
        
        # Randomly split
        np.random.shuffle(valid_users)
        forget_size = int(len(valid_users) * self.forget_ratio)
        
        self.forget_users = set(valid_users[:forget_size])
        self.retain_users = set(valid_users[forget_size:])
        
        print(f"  Retain users: {len(self.retain_users)}")
        print(f"  Forget users: {len(self.forget_users)}")
        
        # Select users based on mode
        if self.mode == 'train':
            self.current_users = valid_users
        elif self.mode == 'retain':
            self.current_users = list(self.retain_users)
        elif self.mode == 'forget':
            self.current_users = list(self.forget_users)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
        
        print(f"  Current mode '{self.mode}': {len(self.current_users)} users")
    
    def _precompute_user_states(self):
        """
        OPTIMIZATION: Precompute user states for all users.
        This avoids recomputing the same user state for every sample.
        """
        print(f"\nPrecomputing user states for {len(self.current_users)} users...")
        
        self.user_states = {}
        
        for user_id in tqdm(self.current_users, desc="Computing user states"):
            # Get user's training data (for state computation)
            user_ratings = self.ratings_df[
                self.ratings_df['userId'] == user_id
            ].sort_values('timestamp')
            
            if len(user_ratings) < self.min_ratings:
                continue
            
            # Use only training portion for state
            split_idx = int(len(user_ratings) * self.train_ratio)
            user_train_data = user_ratings.iloc[:split_idx]
            
            if len(user_train_data) == 0:
                continue
            
            # Compute state
            user_state = self._compute_user_state(user_train_data, user_id)
            self.user_states[user_id] = user_state
        
        print(f"  Cached {len(self.user_states)} user states")
    
    def _prepare_samples(self):
        """
        Prepare training/test samples.
        OPTIMIZED: Uses pre-cached user states and vectorized operations.
        """
        print(f"\nPreparing {self.split} samples...")
        
        self.samples = []
        
        for user_id in tqdm(self.current_users, desc="Creating samples"):
            if user_id not in self.user_states:
                continue
            
            # Get user's ratings
            user_ratings = self.ratings_df[
                self.ratings_df['userId'] == user_id
            ].sort_values('timestamp')
            
            # Temporal split
            split_idx = int(len(user_ratings) * self.train_ratio)
            
            if self.split == 'train':
                user_data = user_ratings.iloc[:split_idx]
            else:
                user_data = user_ratings.iloc[split_idx:]
            
            if len(user_data) == 0:
                continue
            
            # Get pre-computed user state
            user_state = self.user_states[user_id]
            
            # Create samples (vectorized where possible)
            for _, row in user_data.iterrows():
                movie_id = row['movieId']
                rating = row['rating']
                
                if movie_id not in self.movie_features:
                    continue
                
                # Get candidate features
                candidate_features = self._get_candidate_features(movie_id)
                
                # Label
                label = 1 if rating >= self.rating_threshold else 0
                
                self.samples.append({
                    'user_id': user_id,
                    'user_state': user_state,
                    'candidate_features': candidate_features,
                    'movie_id': movie_id,
                    'label': label,
                    'reward': rating
                })
        
        # Class distribution
        num_positive = sum(1 for s in self.samples if s['label'] == 1)
        num_negative = len(self.samples) - num_positive
        print(f"\n  Total samples: {len(self.samples):,}")
        print(f"  Positive (≥{self.rating_threshold}): {num_positive:,} ({num_positive/len(self.samples)*100:.1f}%)")
        print(f"  Negative (<{self.rating_threshold}): {num_negative:,} ({num_negative/len(self.samples)*100:.1f}%)")
    
    def _compute_user_state(self, user_ratings: pd.DataFrame, user_id: int) -> np.ndarray:
        """Compute user state from rating history"""
        if len(user_ratings) == 0:
            return np.zeros(2 + self.num_genres)
        
        # Average release year
        years = []
        for movie_id in user_ratings['movieId']:
            if movie_id in self.movie_features:
                years.append(self.movie_features[movie_id]['year'])
        avg_year = np.mean(years) if years else 0.5
        
        # Average rating given
        avg_rating = user_ratings['rating'].mean() / 5.0
        
        # Genre distribution
        genre_vectors = []
        for movie_id in user_ratings['movieId']:
            if movie_id in self.movie_features:
                genre_vectors.append(self.movie_features[movie_id]['genre_vector'])
        
        if genre_vectors:
            genre_sum = np.sum(genre_vectors, axis=0)
            genre_dist = genre_sum / genre_sum.sum() if genre_sum.sum() > 0 else genre_sum
        else:
            genre_dist = np.zeros(self.num_genres)
        
        # Concatenate
        user_state = np.concatenate([
            [avg_year],
            [avg_rating],
            genre_dist
        ])
        
        return user_state.astype(np.float32)
    
    def _get_candidate_features(self, movie_id: int) -> np.ndarray:
        """Get candidate movie features"""
        features = self.movie_features[movie_id]
        
        avg_rating_norm = features['avg_rating'] / 5.0
        
        candidate_features = np.concatenate([
            [features['year']],
            [avg_rating_norm],
            features['genre_vector']
        ])
        
        return candidate_features.astype(np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Returns training sample"""
        sample = self.samples[idx]
        
        user_state = torch.FloatTensor(sample['user_state'])
        candidate = torch.FloatTensor(sample['candidate_features'])
        
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
        """Get features for all movies"""
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
    """Create train/test dataloaders"""
    
    # Training dataset
    train_dataset = MovieLensRecommendationDataset(
        data_dir=data_dir,
        forget_ratio=forget_ratio,
        rating_threshold=rating_threshold,
        user_total=user_total,
        mode='train',
        split='train',
        **kwargs
    )
    
    # Test datasets (reuse split)
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
