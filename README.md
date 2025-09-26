# LLM-Based User Engagement Foundation Model

A comprehensive machine learning system for predicting user engagement using fine-tuned BERT/RoBERTa models with custom attention mechanisms and behavior sequence analysis.

## Features

- **Fine-tuned LLM Models**: BERT and RoBERTa with custom attention mechanisms
- **Custom Tokenization**: Specialized tokenization for user behavior sequences
- **Attention Mechanisms**: Multi-head attention to identify key engagement factors
- **Comprehensive Evaluation**: Metrics comparison with baseline models
- **CUDA Support**: GPU acceleration for training and inference
- **Scalable Architecture**: Modular design for easy extension

## Technologies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **CUDA**: GPU acceleration
- **Scikit-learn**: Baseline models and evaluation metrics
- **Pandas/NumPy**: Data processing
- **Matplotlib/Seaborn**: Visualization

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jyotidabass/llm-user-engagement-foundation.git
cd llm-user-engagement-foundation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify CUDA installation (optional):
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quick Start

### 1. Training Mode
Train a new model with sample data:
```bash
python main.py --mode train --model_name bert-base-uncased --num_epochs 5
```

### 2. Evaluation Mode
Evaluate a trained model:
```bash
python main.py --mode evaluate --model_path outputs/best_model.pt
```

### 3. Prediction Mode
Make predictions on new data:
```bash
python main.py --mode predict --model_path outputs/best_model.pt
```

### 4. Demo Mode
Run a quick demo with sample predictions:
```bash
python main.py --mode demo
```

## Usage Examples

### Basic Training
```bash
python main.py --mode train \
    --model_name bert-base-uncased \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --max_length 512
```

### Training with RoBERTa
```bash
python main.py --mode train \
    --model_name roberta-base \
    --batch_size 8 \
    --learning_rate 1e-5 \
    --num_epochs 15
```

### Custom Data Training
```bash
python main.py --mode train \
    --data_path /path/to/your/data.csv \
    --output_dir /path/to/output
```

### Evaluation with Baseline Comparison
```bash
python main.py --mode evaluate \
    --model_path outputs/best_model.pt \
    --output_dir evaluation_results
```

## Data Format

The system expects CSV data with the following columns:

- `user_id`: Unique user identifier
- `behavior_sequence`: JSON string of user behavior sequence
- `total_duration`: Total session duration
- `num_purchases`: Number of purchases
- `num_clicks`: Number of clicks
- `num_pages`: Number of unique pages visited
- `device`: Device type (mobile, desktop, tablet)
- `is_engaged`: Binary engagement label (0 or 1)

### Behavior Sequence Format
```json
[
    {
        "behavior": "page_view",
        "page": "home",
        "timestamp": 0,
        "duration": 15
    },
    {
        "behavior": "click",
        "page": "product",
        "timestamp": 15,
        "duration": 5
    }
]
```

## Model Architecture

### 1. Backbone Model
- Pre-trained BERT/RoBERTa encoder
- Fine-tuned for user engagement prediction
- Configurable layer freezing

### 2. Attention Mechanisms
- **Behavior Attention**: Identifies important behavior patterns
- **Temporal Attention**: Captures time-based engagement factors
- **Pattern Attention**: Recognizes behavior sequence patterns

### 3. Classification Head
- Multi-layer perceptron
- Dropout for regularization
- Binary classification output

## Configuration

Edit `config.py` to customize:

```python
@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    num_labels: int = 2
    dropout_rate: float = 0.1

@dataclass
class TrainingConfig:
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_steps: int = 500
```

## Evaluation Metrics

The system provides comprehensive evaluation including:

- **Accuracy**: Overall prediction accuracy
- **Precision/Recall/F1**: Classification performance
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve
- **Confusion Matrix**: Detailed classification breakdown

### Baseline Models
- Logistic Regression
- Random Forest
- Support Vector Machine
- Naive Bayes

## Output Files

After training/evaluation, the following files are generated:

```
outputs/
├── best_model.pt              # Best model checkpoint
├── final_model.pt             # Final trained model
├── training_history.png       # Training curves
├── model_comparison.csv       # Model comparison results
├── confusion_matrices.png     # Confusion matrices
├── roc_curves.png            # ROC curves
├── precision_recall_curves.png # PR curves
├── metrics_comparison.png     # Metrics comparison
├── evaluation_results.json    # Detailed results
└── classification_reports.txt # Classification reports
```

## Advanced Usage

### Custom Tokenization
```python
from custom_tokenizer import UserBehaviorTokenizer

tokenizer = UserBehaviorTokenizer(
    behavior_vocab=vocab,
    max_length=512
)
```

### Model Customization
```python
from model import UserEngagementModel

model = UserEngagementModel(
    config=model_config,
    model_name="bert-base-uncased"
)
```

### Training with Custom Data
```python
from data_processor import UserBehaviorProcessor

processor = UserBehaviorProcessor(config)
df = processor.preprocess_data(your_data)
train_df, val_df, test_df = processor.prepare_datasets(df)
```

## Performance Optimization

### GPU Acceleration
- Automatic CUDA detection
- Mixed precision training (optional)
- Gradient accumulation for large batches

### Memory Optimization
- Gradient checkpointing
- Dynamic batching
- Efficient data loading

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable gradient checkpointing

2. **Slow Training**
   - Use GPU acceleration
   - Increase batch size
   - Use mixed precision

3. **Poor Performance**
   - Increase model capacity
   - Adjust learning rate
   - Add more training data

### Debug Mode
```bash
python main.py --mode train --batch_size 4 --num_epochs 1
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{llm_user_engagement,
  title={LLM-Based User Engagement Foundation Model},
  author={Jyoti Dabass},
  year={2024},
  url={https://github.com/jyotidabass/llm-user-engagement-foundation}
}
```

## Acknowledgments

- Hugging Face for the transformers library
- PyTorch team for the deep learning framework
- The open-source community for various tools and libraries

## Support

For questions and support:
- Open an issue on [GitHub](https://github.com/jyotidabass/llm-user-engagement-foundation/issues)
- Check the documentation
- Review the examples

---

**Note**: This is a foundation model for user engagement prediction. For production use, ensure proper data validation, model monitoring, and performance evaluation.
