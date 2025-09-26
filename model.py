"""
Fine-tuned BERT/RoBERTa model with attention mechanisms for user engagement prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel, BertConfig,
    RobertaModel, RobertaConfig,
    PreTrainedModel
)
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Dict, Any
import math

class AttentionLayer(nn.Module):
    """Custom attention layer for identifying key engagement factors"""
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"
        
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        batch_size, seq_len, hidden_size = hidden_states.size()
        
        # Linear projections
        Q = self.query(hidden_states)
        K = self.key(hidden_states)
        V = self.value(hidden_states)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            # Expand mask for multi-head attention
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(attention_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, hidden_size
        )
        
        output = self.output_projection(context)
        
        return output, attention_weights

class EngagementPredictor(nn.Module):
    """Engagement prediction head with attention mechanisms"""
    
    def __init__(self, hidden_size: int, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        
        # Attention layers for different aspects
        self.behavior_attention = AttentionLayer(hidden_size, num_heads=8, dropout=dropout)
        self.temporal_attention = AttentionLayer(hidden_size, num_heads=4, dropout=dropout)
        self.pattern_attention = AttentionLayer(hidden_size, num_heads=4, dropout=dropout)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size // 2, num_labels)
        
        # Attention weights for interpretability
        self.attention_weights = None
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Apply different attention mechanisms
        behavior_output, behavior_weights = self.behavior_attention(hidden_states, attention_mask)
        temporal_output, temporal_weights = self.temporal_attention(hidden_states, attention_mask)
        pattern_output, pattern_weights = self.pattern_attention(hidden_states, attention_mask)
        
        # Global average pooling
        if attention_mask is not None:
            # Masked average pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(behavior_output)
            behavior_pooled = (behavior_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            temporal_pooled = (temporal_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            pattern_pooled = (pattern_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            behavior_pooled = behavior_output.mean(dim=1)
            temporal_pooled = temporal_output.mean(dim=1)
            pattern_pooled = pattern_output.mean(dim=1)
        
        # Concatenate features
        fused_features = torch.cat([behavior_pooled, temporal_pooled, pattern_pooled], dim=-1)
        
        # Feature fusion
        fused_output = self.feature_fusion(fused_features)
        
        # Classification
        logits = self.classifier(fused_output)
        
        # Store attention weights for interpretability
        self.attention_weights = {
            'behavior': behavior_weights,
            'temporal': temporal_weights,
            'pattern': pattern_weights
        }
        
        return logits

class UserEngagementModel(PreTrainedModel):
    """Main model for user engagement prediction"""
    
    def __init__(self, config, model_name: str = "bert-base-uncased"):
        super().__init__(config)
        self.config = config
        self.model_name = model_name
        
        # Load pre-trained model
        if "roberta" in model_name.lower():
            self.backbone = RobertaModel.from_pretrained(model_name)
            self.config.hidden_size = self.backbone.config.hidden_size
        else:
            self.backbone = BertModel.from_pretrained(model_name)
            self.config.hidden_size = self.backbone.config.hidden_size
        
        # Freeze some layers for fine-tuning
        self._freeze_backbone_layers()
        
        # Engagement predictor
        self.engagement_predictor = EngagementPredictor(
            hidden_size=self.config.hidden_size,
            num_labels=config.num_labels,
            dropout=config.dropout_rate
        )
        
        # Initialize weights
        self.init_weights()
        
    def _freeze_backbone_layers(self, freeze_layers: int = 6):
        """Freeze early layers of the backbone model"""
        if hasattr(self.backbone, 'encoder'):
            # BERT/RoBERTa structure
            for i, layer in enumerate(self.backbone.encoder.layer):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
        elif hasattr(self.backbone, 'layers'):
            # Alternative structure
            for i, layer in enumerate(self.backbone.layers):
                if i < freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> SequenceClassifierOutput:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get backbone outputs
        backbone_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get hidden states
        hidden_states = backbone_outputs.last_hidden_state
        
        # Get engagement predictions
        logits = self.engagement_predictor(hidden_states, attention_mask)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        if not return_dict:
            output = (logits,) + backbone_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=backbone_outputs.hidden_states,
            attentions=backbone_outputs.attentions,
        )
    
    def get_attention_weights(self):
        """Get attention weights for interpretability"""
        return self.engagement_predictor.attention_weights
    
    def predict_engagement(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Predict engagement with attention weights"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            
            attention_weights = self.get_attention_weights()
            
            return {
                'predictions': predictions,
                'probabilities': probabilities,
                'logits': logits,
                'attention_weights': attention_weights
            }

class BaselineModel(nn.Module):
    """Baseline model for comparison"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, num_labels: int = 2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, num_labels)
        )
    
    def forward(self, x):
        return self.classifier(x)

class EnsembleModel(nn.Module):
    """Ensemble model combining multiple approaches"""
    
    def __init__(self, models: list, weights: Optional[list] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights if weights else [1.0] * len(models)
        
    def forward(self, *args, **kwargs):
        outputs = []
        for model in self.models:
            outputs.append(model(*args, **kwargs))
        
        # Weighted average
        if isinstance(outputs[0], dict):
            # Handle dictionary outputs
            ensemble_output = {}
            for key in outputs[0].keys():
                if isinstance(outputs[0][key], torch.Tensor):
                    ensemble_output[key] = sum(
                        w * out[key] for w, out in zip(self.weights, outputs)
                    ) / sum(self.weights)
                else:
                    ensemble_output[key] = outputs[0][key]
            return ensemble_output
        else:
            # Handle tensor outputs
            return sum(w * out for w, out in zip(self.weights, outputs)) / sum(self.weights)

def create_model(config, model_name: str = "bert-base-uncased") -> UserEngagementModel:
    """Create and initialize the model"""
    
    # Create model configuration
    model_config = type('Config', (), {
        'hidden_size': 768,  # Will be updated by the model
        'num_labels': config.model.num_labels,
        'dropout_rate': config.model.dropout_rate,
        'use_return_dict': True
    })()
    
    # Create model
    model = UserEngagementModel(model_config, model_name)
    
    return model

def create_baseline_model(input_size: int, num_labels: int = 2) -> BaselineModel:
    """Create baseline model for comparison"""
    return BaselineModel(input_size, num_labels=num_labels)

if __name__ == "__main__":
    # Test model creation
    from config import Config
    
    config = Config()
    model = create_model(config)
    
    print(f"Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Test forward pass
    batch_size = 2
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"Output logits shape: {outputs.logits.shape}")
    
    # Test prediction
    predictions = model.predict_engagement(input_ids, attention_mask)
    print(f"Predictions: {predictions['predictions']}")
    print(f"Probabilities shape: {predictions['probabilities'].shape}")
