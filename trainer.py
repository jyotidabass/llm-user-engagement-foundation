"""
Training pipeline with CUDA support for user engagement prediction
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import os
import json
from tqdm import tqdm
import logging
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import get_linear_schedule_with_warmup, AdamW
import wandb
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UserEngagementDataset(Dataset):
    """Dataset class for user engagement data"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Tokenize behavior sequence
        encoded = self.tokenizer.encode_plus(
            row['behavior_sequence'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get labels
        label = torch.tensor(row['is_engaged'], dtype=torch.long)
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': label
        }

class UserEngagementTrainer:
    """Trainer class for user engagement prediction"""
    
    def __init__(self, config, model, tokenizer, device: str = None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # Best model tracking
        self.best_val_score = 0.0
        self.best_model_state = None
        
        # Set up logging
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging and experiment tracking"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.project_name,
                config={
                    'model_name': self.config.model.model_name,
                    'batch_size': self.config.training.batch_size,
                    'learning_rate': self.config.training.learning_rate,
                    'num_epochs': self.config.training.num_epochs,
                    'max_length': self.config.model.max_length
                }
            )
    
    def setup_optimizer_and_scheduler(self, num_training_steps: int):
        """Set up optimizer and learning rate scheduler"""
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
        
        # Scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def create_data_loaders(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation"""
        
        # Create datasets
        train_dataset = UserEngagementDataset(
            train_df, self.tokenizer, self.config.model.max_length
        )
        val_dataset = UserEngagementDataset(
            val_df, self.tokenizer, self.config.model.max_length
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            if self.config.training.gradient_accumulation_steps > 1:
                loss = loss / self.config.training.gradient_accumulation_steps
            
            loss.backward()
            
            # Gradient clipping
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
            
            # Update weights
            if (batch_idx + 1) % self.config.training.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress
            total_loss += loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            # Logging
            if (batch_idx + 1) % self.config.training.logging_steps == 0:
                logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
                
                if self.config.use_wandb:
                    wandb.log({
                        'train_loss': loss.item(),
                        'learning_rate': self.scheduler.get_last_lr()[0],
                        'epoch': epoch + 1
                    })
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        avg_loss = total_loss / len(train_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(outputs.logits, dim=-1)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_probabilities)
        except:
            auc = 0.0
        
        avg_loss = total_loss / len(val_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(train_df, val_df)
        
        # Calculate total training steps
        num_training_steps = len(train_loader) * self.config.training.num_epochs
        
        # Set up optimizer and scheduler
        self.setup_optimizer_and_scheduler(num_training_steps)
        
        # Training loop
        for epoch in range(self.config.training.num_epochs):
            logger.info(f"Epoch {epoch+1}/{self.config.training.num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, epoch)
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_accuracy'].append(train_metrics['accuracy'])
            self.training_history['val_accuracy'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics['f1'])
            self.training_history['val_f1'].append(val_metrics['f1'])
            
            # Log metrics
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
            logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'train_accuracy': train_metrics['accuracy'],
                    'val_accuracy': val_metrics['accuracy'],
                    'train_f1': train_metrics['f1'],
                    'val_f1': val_metrics['f1'],
                    'val_auc': val_metrics['auc']
                })
            
            # Save best model
            if val_metrics['f1'] > self.best_val_score:
                self.best_val_score = val_metrics['f1']
                self.best_model_state = self.model.state_dict().copy()
                self.save_model(f"{self.config.data.output_dir}/best_model.pt")
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_steps == 0:
                self.save_model(f"{self.config.data.output_dir}/checkpoint_epoch_{epoch+1}.pt")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        
        logger.info("Training completed!")
        return self.training_history
    
    def save_model(self, filepath: str):
        """Save model and tokenizer"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history,
            'best_val_score': self.best_val_score
        }, filepath)
        
        # Save tokenizer
        tokenizer_path = filepath.replace('.pt', '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model and tokenizer"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.best_val_score = checkpoint.get('best_val_score', 0.0)
        
        logger.info(f"Model loaded from {filepath}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['train_accuracy'], label='Train')
        axes[0, 1].plot(self.training_history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.training_history['train_f1'], label='Train')
        axes[1, 0].plot(self.training_history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Combined metrics
        axes[1, 1].plot(self.training_history['val_accuracy'], label='Val Accuracy')
        axes[1, 1].plot(self.training_history['val_f1'], label='Val F1')
        axes[1, 1].set_title('Validation Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

def train_model(config, model, tokenizer, train_df: pd.DataFrame, val_df: pd.DataFrame):
    """Train the model"""
    trainer = UserEngagementTrainer(config, model, tokenizer)
    
    # Train
    history = trainer.train(train_df, val_df)
    
    # Plot training history
    trainer.plot_training_history(f"{config.data.output_dir}/training_history.png")
    
    return trainer, history

if __name__ == "__main__":
    # Test trainer
    from config import Config
    from data_processor import create_sample_datasets
    from custom_tokenizer import create_tokenizer_from_processor
    from model import create_model
    
    config = Config()
    
    # Create sample data
    train_df, val_df, test_df, processor = create_sample_datasets(config)
    
    # Create tokenizer and model
    tokenizer = create_tokenizer_from_processor(processor)
    model = create_model(config)
    
    # Train model
    trainer, history = train_model(config, model, tokenizer, train_df, val_df)
    
    print("Training completed successfully!")
    print(f"Best validation F1 score: {trainer.best_val_score:.4f}")
