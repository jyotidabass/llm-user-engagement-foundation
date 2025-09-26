"""
Main execution script for LLM-Based User Engagement Foundation Model
"""

import os
import sys
import argparse
import logging
import torch
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from config import Config
from data_processor import UserBehaviorProcessor, create_sample_datasets
from custom_tokenizer import create_tokenizer_from_processor, UserBehaviorTokenizer
from model import create_model, create_baseline_model
from trainer import UserEngagementTrainer, UserEngagementDataset, train_model
from evaluator import ModelEvaluator, evaluate_model_comprehensive

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LLM-Based User Engagement Foundation Model')
    
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'evaluate', 'predict', 'demo'],
                       help='Mode to run the script')
    
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       choices=['bert-base-uncased', 'roberta-base'],
                       help='Pre-trained model to use')
    
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate for training')
    
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    
    parser.add_argument('--use_cuda', action='store_true', default=True,
                       help='Use CUDA if available')
    
    parser.add_argument('--use_wandb', action='store_true', default=False,
                       help='Use Weights & Biases for logging')
    
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to custom dataset')
    
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model')
    
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for results')
    
    parser.add_argument('--create_sample_data', action='store_true', default=True,
                       help='Create sample data if no data provided')
    
    return parser.parse_args()

def setup_config(args):
    """Setup configuration from arguments"""
    config = Config()
    
    # Update config with arguments
    config.model.model_name = args.model_name
    config.model.max_length = args.max_length
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.num_epochs = args.num_epochs
    config.data.output_dir = args.output_dir
    config.use_wandb = args.use_wandb
    config.use_cuda = args.use_cuda
    
    # Create output directory
    os.makedirs(config.data.output_dir, exist_ok=True)
    
    return config

def load_data(config, args):
    """Load or create data"""
    if args.data_path and os.path.exists(args.data_path):
        logger.info(f"Loading data from {args.data_path}")
        # Load custom data
        df = pd.read_csv(args.data_path)
        processor = UserBehaviorProcessor(config)
        df = processor.preprocess_data(df)
        train_df, val_df, test_df = processor.prepare_datasets(df)
    else:
        if args.create_sample_data:
            logger.info("Creating sample data...")
            train_df, val_df, test_df, processor = create_sample_datasets(config)
        else:
            raise ValueError("No data provided and create_sample_data is False")
    
    logger.info(f"Data loaded - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df, processor

def train_mode(config, args):
    """Training mode"""
    logger.info("Starting training mode...")
    
    # Load data
    train_df, val_df, test_df, processor = load_data(config, args)
    
    # Create tokenizer
    tokenizer = create_tokenizer_from_processor(processor)
    logger.info(f"Tokenizer created with vocabulary size: {len(tokenizer.vocab)}")
    
    # Create model
    model = create_model(config, args.model_name)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Train model
    trainer, history = train_model(config, model, tokenizer, train_df, val_df)
    
    # Save final model
    trainer.save_model(f"{config.data.output_dir}/final_model.pt")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    from torch.utils.data import DataLoader
    
    test_dataset = UserEngagementDataset(test_df, tokenizer, config.model.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    evaluator = ModelEvaluator(config)
    test_metrics = evaluator.evaluate_model(model, test_loader, "Final Model")
    
    logger.info(f"Test set results - Accuracy: {test_metrics['accuracy']:.4f}, F1: {test_metrics['f1_score']:.4f}")
    
    return trainer, model, tokenizer, test_metrics

def evaluate_mode(config, args):
    """Evaluation mode"""
    logger.info("Starting evaluation mode...")
    
    if not args.model_path:
        raise ValueError("Model path required for evaluation mode")
    
    # Load data
    train_df, val_df, test_df, processor = load_data(config, args)
    
    # Create tokenizer
    tokenizer = create_tokenizer_from_processor(processor)
    
    # Create and load model
    model = create_model(config, args.model_name)
    trainer = UserEngagementTrainer(config, model, tokenizer)
    trainer.load_model(args.model_path)
    
    # Create test loader
    from torch.utils.data import DataLoader
    test_dataset = UserEngagementDataset(test_df, tokenizer, config.model.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # Comprehensive evaluation
    results = evaluate_model_comprehensive(
        config, model, test_loader, train_df, test_df, "Loaded Model"
    )
    
    return results

def predict_mode(config, args):
    """Prediction mode"""
    logger.info("Starting prediction mode...")
    
    if not args.model_path:
        raise ValueError("Model path required for prediction mode")
    
    # Load data
    train_df, val_df, test_df, processor = load_data(config, args)
    
    # Create tokenizer
    tokenizer = create_tokenizer_from_processor(processor)
    
    # Create and load model
    model = create_model(config, args.model_name)
    trainer = UserEngagementTrainer(config, model, tokenizer)
    trainer.load_model(args.model_path)
    
    # Make predictions on test set
    model.eval()
    predictions = []
    probabilities = []
    
    with torch.no_grad():
        for _, row in test_df.iterrows():
            # Tokenize
            encoded = tokenizer.encode_plus(
                row['behavior_sequence'],
                max_length=config.model.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoded['input_ids'].to(trainer.device)
            attention_mask = encoded['attention_mask'].to(trainer.device)
            
            # Predict
            if hasattr(model, 'predict_engagement'):
                result = model.predict_engagement(input_ids, attention_mask)
                pred = result['predictions'].item()
                prob = result['probabilities'][0, 1].item()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                prob = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
            
            predictions.append(pred)
            probabilities.append(prob)
    
    # Create results DataFrame
    results_df = test_df.copy()
    results_df['predicted_engagement'] = predictions
    results_df['engagement_probability'] = probabilities
    
    # Save predictions
    results_df.to_csv(f"{config.data.output_dir}/predictions.csv", index=False)
    logger.info(f"Predictions saved to {config.data.output_dir}/predictions.csv")
    
    return results_df

def demo_mode(config, args):
    """Demo mode with sample predictions"""
    logger.info("Starting demo mode...")
    
    # Load data
    train_df, val_df, test_df, processor = load_data(config, args)
    
    # Create tokenizer
    tokenizer = create_tokenizer_from_processor(processor)
    
    # Create model (untrained for demo)
    model = create_model(config, args.model_name)
    
    # Demo predictions on a few samples
    logger.info("Making demo predictions...")
    
    demo_samples = test_df.head(5)
    
    for idx, row in demo_samples.iterrows():
        logger.info(f"\n--- Sample {idx + 1} ---")
        logger.info(f"User ID: {row['user_id']}")
        logger.info(f"Total Duration: {row['total_duration']:.2f}")
        logger.info(f"Number of Clicks: {row['num_clicks']}")
        logger.info(f"Number of Purchases: {row['num_purchases']}")
        logger.info(f"Actual Engagement: {'Engaged' if row['is_engaged'] else 'Not Engaged'}")
        
        # Make prediction
        encoded = tokenizer.encode_plus(
            row['behavior_sequence'],
            max_length=config.model.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask']
            )
            
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            probability = torch.softmax(outputs.logits, dim=-1)[0, 1].item()
            
            logger.info(f"Predicted Engagement: {'Engaged' if prediction else 'Not Engaged'}")
            logger.info(f"Engagement Probability: {probability:.4f}")
    
    return demo_samples

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup configuration
        config = setup_config(args)
        
        logger.info(f"Starting LLM-Based User Engagement Foundation Model")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Device: {config.device}")
        logger.info(f"Output directory: {config.data.output_dir}")
        
        # Run appropriate mode
        if args.mode == 'train':
            trainer, model, tokenizer, test_metrics = train_mode(config, args)
            logger.info("Training completed successfully!")
            
        elif args.mode == 'evaluate':
            results = evaluate_mode(config, args)
            logger.info("Evaluation completed successfully!")
            
        elif args.mode == 'predict':
            predictions = predict_mode(config, args)
            logger.info("Prediction completed successfully!")
            
        elif args.mode == 'demo':
            demo_results = demo_mode(config, args)
            logger.info("Demo completed successfully!")
        
        logger.info("Script completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()
