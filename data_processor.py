"""
Data preprocessing module for user behavior sequences
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import json
import pickle
from pathlib import Path

class UserBehaviorProcessor:
    """Processes user behavior sequences for engagement prediction"""
    
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.behavior_vocab = {}
        self.feature_columns = []
        
    def create_sample_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """Create sample user behavior data for demonstration"""
        np.random.seed(42)
        
        # Generate sample user behavior sequences
        behaviors = ['page_view', 'click', 'scroll', 'hover', 'search', 'purchase', 'add_to_cart', 'login', 'logout']
        pages = ['home', 'product', 'category', 'checkout', 'profile', 'search_results']
        devices = ['mobile', 'desktop', 'tablet']
        
        data = []
        for i in range(num_samples):
            # Generate user session
            session_length = np.random.randint(5, 50)
            user_id = f"user_{i}"
            
            # Generate behavior sequence
            behavior_sequence = []
            for j in range(session_length):
                behavior = np.random.choice(behaviors)
                page = np.random.choice(pages)
                timestamp = j * np.random.randint(1, 300)  # seconds
                
                behavior_sequence.append({
                    'behavior': behavior,
                    'page': page,
                    'timestamp': timestamp,
                    'duration': np.random.randint(1, 60)
                })
            
            # Calculate engagement features
            total_duration = sum([b['duration'] for b in behavior_sequence])
            num_purchases = sum([1 for b in behavior_sequence if b['behavior'] == 'purchase'])
            num_clicks = sum([1 for b in behavior_sequence if b['behavior'] == 'click'])
            num_pages = len(set([b['page'] for b in behavior_sequence]))
            
            # Create engagement label (binary)
            engagement_score = (
                total_duration * 0.3 + 
                num_purchases * 10 + 
                num_clicks * 0.5 + 
                num_pages * 0.2
            )
            is_engaged = 1 if engagement_score > np.percentile([self._calculate_engagement_score(behaviors, pages) for _ in range(100)], 70) else 0
            
            data.append({
                'user_id': user_id,
                'behavior_sequence': json.dumps(behavior_sequence),
                'total_duration': total_duration,
                'num_purchases': num_purchases,
                'num_clicks': num_clicks,
                'num_pages': num_pages,
                'device': np.random.choice(devices),
                'is_engaged': is_engaged
            })
        
        return pd.DataFrame(data)
    
    def _calculate_engagement_score(self, behaviors, pages):
        """Helper method to calculate engagement score"""
        return np.random.random() * 100
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the raw data"""
        # Create behavior sequence tokens
        df['behavior_tokens'] = df['behavior_sequence'].apply(self._tokenize_behavior_sequence)
        
        # Encode categorical features
        df['device_encoded'] = self.label_encoder.fit_transform(df['device'])
        
        # Normalize numerical features
        numerical_features = ['total_duration', 'num_purchases', 'num_clicks', 'num_pages']
        df[numerical_features] = self.scaler.fit_transform(df[numerical_features])
        
        return df
    
    def _tokenize_behavior_sequence(self, behavior_sequence_str: str) -> List[str]:
        """Convert behavior sequence to tokens"""
        try:
            sequence = json.loads(behavior_sequence_str)
            tokens = []
            
            for behavior in sequence:
                # Create composite tokens for behavior patterns
                behavior_token = f"{behavior['behavior']}_{behavior['page']}"
                tokens.append(behavior_token)
                
                # Add duration-based tokens
                if behavior['duration'] > 30:
                    tokens.append("long_duration")
                else:
                    tokens.append("short_duration")
            
            return tokens
        except:
            return ["unknown_behavior"]
    
    def create_behavior_vocabulary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Create vocabulary from behavior tokens"""
        all_tokens = []
        for tokens in df['behavior_tokens']:
            all_tokens.extend(tokens)
        
        unique_tokens = list(set(all_tokens))
        self.behavior_vocab = {token: idx for idx, token in enumerate(unique_tokens)}
        
        # Add special tokens
        self.behavior_vocab['[PAD]'] = len(self.behavior_vocab)
        self.behavior_vocab['[UNK]'] = len(self.behavior_vocab)
        self.behavior_vocab['[CLS]'] = len(self.behavior_vocab)
        self.behavior_vocab['[SEP]'] = len(self.behavior_vocab)
        
        return self.behavior_vocab
    
    def prepare_datasets(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        # Create behavior vocabulary
        self.create_behavior_vocabulary(df)
        
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['is_engaged'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['is_engaged'])
        
        return train_df, val_df, test_df
    
    def save_processor(self, filepath: str):
        """Save the processor state"""
        processor_state = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'behavior_vocab': self.behavior_vocab,
            'feature_columns': self.feature_columns
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(processor_state, f)
    
    def load_processor(self, filepath: str):
        """Load the processor state"""
        with open(filepath, 'rb') as f:
            processor_state = pickle.load(f)
        
        self.label_encoder = processor_state['label_encoder']
        self.scaler = processor_state['scaler']
        self.behavior_vocab = processor_state['behavior_vocab']
        self.feature_columns = processor_state['feature_columns']

def create_sample_datasets(config):
    """Create and save sample datasets"""
    processor = UserBehaviorProcessor(config)
    
    # Create sample data
    df = processor.create_sample_data(num_samples=2000)
    
    # Preprocess data
    df = processor.preprocess_data(df)
    
    # Split into train/val/test
    train_df, val_df, test_df = processor.prepare_datasets(df)
    
    # Save datasets
    train_df.to_csv(config.data.train_data_path, index=False)
    val_df.to_csv(config.data.val_data_path, index=False)
    test_df.to_csv(config.data.test_data_path, index=False)
    
    # Save processor
    processor.save_processor(f"{config.data.cache_dir}/processor.pkl")
    
    print(f"Created datasets:")
    print(f"Train: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    print(f"Behavior vocabulary size: {len(processor.behavior_vocab)}")
    
    return train_df, val_df, test_df, processor

if __name__ == "__main__":
    from config import Config
    config = Config()
    create_sample_datasets(config)
