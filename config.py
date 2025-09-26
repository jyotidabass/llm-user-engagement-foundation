"""
Configuration file for LLM-Based User Engagement Foundation Model
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration parameters"""
    model_name: str = "bert-base-uncased"  # or "roberta-base"
    max_length: int = 512
    num_labels: int = 2  # binary classification: engaged/not engaged
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100

@dataclass
class DataConfig:
    """Data configuration parameters"""
    train_data_path: str = "data/train.csv"
    val_data_path: str = "data/val.csv"
    test_data_path: str = "data/test.csv"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    max_sequence_length: int = 256

@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    
    # Device configuration
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    use_cuda: bool = True
    
    # Logging
    use_wandb: bool = False
    project_name: str = "llm-user-engagement"
    
    def __post_init__(self):
        """Create necessary directories"""
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
        os.makedirs("data", exist_ok=True)
