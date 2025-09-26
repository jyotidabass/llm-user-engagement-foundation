"""
Model evaluation metrics and comparison with baseline models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve,
    average_precision_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import json
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""
    
    def __init__(self, config, device: str = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        
    def evaluate_model(
        self, 
        model: nn.Module, 
        test_loader: DataLoader, 
        model_name: str = "Main Model"
    ) -> Dict[str, Any]:
        """Evaluate a single model"""
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_logits = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                if hasattr(model, 'predict_engagement'):
                    # Custom model with prediction method
                    outputs = model.predict_engagement(input_ids, attention_mask)
                    logits = outputs['logits']
                    probabilities = outputs['probabilities']
                else:
                    # Standard model
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=-1)
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Positive class probability
                all_logits.extend(logits.cpu().numpy())
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_predictions, all_probabilities)
        
        # Store results
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'logits': all_logits
        }
        
        return metrics
    
    def _calculate_metrics(self, labels: List[int], predictions: List[int], probabilities: List[float]) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        # Binary classification metrics
        precision_binary, recall_binary, f1_binary, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        # AUC metrics
        try:
            auc_roc = roc_auc_score(labels, probabilities)
        except:
            auc_roc = 0.0
        
        try:
            auc_pr = average_precision_score(labels, probabilities)
        except:
            auc_pr = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_binary': precision_binary,
            'recall_binary': recall_binary,
            'f1_score_binary': f1_binary,
            'auc_roc': auc_roc,
            'auc_pr': auc_pr,
            'specificity': specificity,
            'sensitivity': sensitivity,
            'true_positives': tp,
            'true_negatives': tn,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def create_baseline_models(self, train_df: pd.DataFrame) -> Dict[str, Any]:
        """Create and train baseline models"""
        baseline_models = {}
        
        # Prepare features for baseline models
        X_train = self._prepare_baseline_features(train_df)
        y_train = train_df['is_engaged'].values
        
        # Logistic Regression
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        lr_model.fit(X_train, y_train)
        baseline_models['Logistic Regression'] = lr_model
        
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        baseline_models['Random Forest'] = rf_model
        
        # SVM
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X_train, y_train)
        baseline_models['SVM'] = svm_model
        
        # Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(X_train, y_train)
        baseline_models['Naive Bayes'] = nb_model
        
        return baseline_models
    
    def _prepare_baseline_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare features for baseline models"""
        features = []
        
        for _, row in df.iterrows():
            # Numerical features
            numerical_features = [
                row['total_duration'],
                row['num_purchases'],
                row['num_clicks'],
                row['num_pages'],
                row['device_encoded']
            ]
            
            # Behavior sequence features
            try:
                behavior_sequence = json.loads(row['behavior_sequence'])
                sequence_features = [
                    len(behavior_sequence),  # Sequence length
                    sum(1 for b in behavior_sequence if b['behavior'] == 'purchase'),
                    sum(1 for b in behavior_sequence if b['behavior'] == 'click'),
                    sum(1 for b in behavior_sequence if b['behavior'] == 'search'),
                    len(set(b['behavior'] for b in behavior_sequence)),  # Unique behaviors
                    len(set(b['page'] for b in behavior_sequence)),  # Unique pages
                    np.mean([b.get('duration', 0) for b in behavior_sequence]),  # Avg duration
                    np.std([b.get('duration', 0) for b in behavior_sequence]),  # Duration std
                ]
            except:
                sequence_features = [0] * 8
            
            # Combine all features
            all_features = numerical_features + sequence_features
            features.append(all_features)
        
        return np.array(features)
    
    def evaluate_baseline_models(self, baseline_models: Dict[str, Any], test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Evaluate baseline models"""
        X_test = self._prepare_baseline_features(test_df)
        y_test = test_df['is_engaged'].values
        
        baseline_results = {}
        
        for name, model in baseline_models.items():
            # Get predictions
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else predictions
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test.tolist(), predictions.tolist(), probabilities.tolist())
            baseline_results[name] = metrics
            
            # Store results
            self.results[name] = {
                'metrics': metrics,
                'predictions': predictions.tolist(),
                'labels': y_test.tolist(),
                'probabilities': probabilities.tolist(),
                'logits': None
            }
        
        return baseline_results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all models and return results as DataFrame"""
        comparison_data = []
        
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1_score'],
                'AUC-ROC': metrics['auc_roc'],
                'AUC-PR': metrics['auc_pr'],
                'Specificity': metrics['specificity'],
                'Sensitivity': metrics['sensitivity']
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_confusion_matrices(self, save_path: Optional[str] = None):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (model_name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(result['labels'], result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{model_name} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide unused subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, save_path: Optional[str] = None):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            if result['probabilities'] is not None:
                fpr, tpr, _ = roc_curve(result['labels'], result['probabilities'])
                auc = result['metrics']['auc_roc']
                plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, save_path: Optional[str] = None):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for model_name, result in self.results.items():
            if result['probabilities'] is not None:
                precision, recall, _ = precision_recall_curve(result['labels'], result['probabilities'])
                auc_pr = result['metrics']['auc_pr']
                plt.plot(recall, precision, label=f'{model_name} (AUC-PR = {auc_pr:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")
        
        plt.show()
    
    def plot_metrics_comparison(self, save_path: Optional[str] = None):
        """Plot metrics comparison bar chart"""
        comparison_df = self.compare_models()
        
        # Select key metrics for comparison
        key_metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'AUC-PR']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(key_metrics):
            comparison_df.plot(x='Model', y=metric, kind='bar', ax=axes[idx])
            axes[idx].set_title(f'{metric} Comparison')
            axes[idx].set_ylabel(metric)
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics comparison saved to {save_path}")
        
        plt.show()
    
    def generate_classification_reports(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """Generate detailed classification reports for all models"""
        reports = {}
        
        for model_name, result in self.results.items():
            report = classification_report(
                result['labels'], 
                result['predictions'], 
                target_names=['Not Engaged', 'Engaged']
            )
            reports[model_name] = report
        
        if save_path:
            with open(save_path, 'w') as f:
                for model_name, report in reports.items():
                    f.write(f"\n{'='*50}\n")
                    f.write(f"Classification Report for {model_name}\n")
                    f.write(f"{'='*50}\n")
                    f.write(report)
                    f.write("\n")
            
            logger.info(f"Classification reports saved to {save_path}")
        
        return reports
    
    def save_results(self, filepath: str):
        """Save all evaluation results"""
        results_to_save = {}
        
        for model_name, result in self.results.items():
            results_to_save[model_name] = {
                'metrics': result['metrics'],
                'predictions': result['predictions'],
                'labels': result['labels'],
                'probabilities': result['probabilities']
            }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def comprehensive_evaluation(
        self, 
        model: nn.Module, 
        test_loader: DataLoader, 
        train_df: pd.DataFrame, 
        test_df: pd.DataFrame,
        model_name: str = "Main Model",
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive evaluation with baseline comparison"""
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        logger.info("Starting comprehensive evaluation...")
        
        # Evaluate main model
        logger.info(f"Evaluating {model_name}...")
        main_metrics = self.evaluate_model(model, test_loader, model_name)
        
        # Create and evaluate baseline models
        logger.info("Creating baseline models...")
        baseline_models = self.create_baseline_models(train_df)
        
        logger.info("Evaluating baseline models...")
        baseline_results = self.evaluate_baseline_models(baseline_models, test_df)
        
        # Generate comparison
        comparison_df = self.compare_models()
        
        # Generate plots
        if save_dir:
            self.plot_confusion_matrices(f"{save_dir}/confusion_matrices.png")
            self.plot_roc_curves(f"{save_dir}/roc_curves.png")
            self.plot_precision_recall_curves(f"{save_dir}/precision_recall_curves.png")
            self.plot_metrics_comparison(f"{save_dir}/metrics_comparison.png")
            
            # Save results
            comparison_df.to_csv(f"{save_dir}/model_comparison.csv", index=False)
            self.save_results(f"{save_dir}/evaluation_results.json")
            self.generate_classification_reports(f"{save_dir}/classification_reports.txt")
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        return {
            'main_model_metrics': main_metrics,
            'baseline_results': baseline_results,
            'comparison_df': comparison_df,
            'all_results': self.results
        }

def evaluate_model_comprehensive(
    config, 
    model: nn.Module, 
    test_loader: DataLoader, 
    train_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    model_name: str = "Main Model"
) -> Dict[str, Any]:
    """Convenience function for comprehensive evaluation"""
    evaluator = ModelEvaluator(config)
    return evaluator.comprehensive_evaluation(
        model, test_loader, train_df, test_df, model_name, config.data.output_dir
    )

if __name__ == "__main__":
    # Test evaluator
    from config import Config
    from data_processor import create_sample_datasets
    from custom_tokenizer import create_tokenizer_from_processor
    from model import create_model
    from trainer import UserEngagementDataset
    
    config = Config()
    
    # Create sample data
    train_df, val_df, test_df, processor = create_sample_datasets(config)
    
    # Create tokenizer and model
    tokenizer = create_tokenizer_from_processor(processor)
    model = create_model(config)
    
    # Create test dataset and loader
    test_dataset = UserEngagementDataset(test_df, tokenizer, config.model.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # Evaluate
    results = evaluate_model_comprehensive(config, model, test_loader, train_df, test_df)
    
    print("Evaluation completed successfully!")
    print(f"Main model F1 score: {results['main_model_metrics']['f1_score']:.4f}")
