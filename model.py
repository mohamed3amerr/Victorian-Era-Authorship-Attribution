"""
Model definition for Victorian Era Authorship Attribution.
Uses BERT or RoBERTa as base model with a classification head.
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from typing import Optional


class AuthorClassifier(nn.Module):
    """
    Transformer-based author classifier.
    Uses BERT or RoBERTa as the base model with a classification head.
    """
    
    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        num_classes: int = 50,
        dropout: float = 0.3,
        freeze_base: bool = False,
        gradient_checkpointing: bool = False
    ):
        """
        Args:
            model_name: HuggingFace model name (bert-base-uncased, roberta-base, etc.)
            num_classes: Number of author classes
            dropout: Dropout rate for classification head
            freeze_base: Whether to freeze the base transformer model
            gradient_checkpointing: Use gradient checkpointing to save memory (slower but uses less memory)
        """
        super(AuthorClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pre-trained transformer model
        self.transformer = AutoModel.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.hidden_size = config.hidden_size
        
        # Enable gradient checkpointing if requested (saves memory but slightly slower)
        if gradient_checkpointing:
            try:
                # Try new API first (transformers >= 4.20.0)
                if hasattr(self.transformer, 'gradient_checkpointing_enable'):
                    self.transformer.gradient_checkpointing_enable()
                # Fallback to old API
                elif hasattr(self.transformer, 'gradient_checkpointing'):
                    self.transformer.gradient_checkpointing = True
                else:
                    print("Warning: Gradient checkpointing not supported for this model")
            except Exception as e:
                print(f"Warning: Could not enable gradient checkpointing: {e}")
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: Tokenized input IDs
            attention_mask: Attention mask
        
        Returns:
            Logits for each class
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        # Some models (like DistilBERT) don't have pooler_output, use last_hidden_state instead
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled_output = outputs.pooler_output
        else:
            # Use [CLS] token from last hidden state (first token)
            pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits
    
    def get_tokenizer(self):
        """Get the appropriate tokenizer for this model."""
        return AutoTokenizer.from_pretrained(self.model_name)


def create_model(
    model_name: str = 'bert-base-uncased',
    num_classes: int = 50,
    dropout: float = 0.3,
    freeze_base: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    gradient_checkpointing: bool = False,
    use_torch_compile: bool = False
) -> tuple:
    """
    Create model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        num_classes: Number of author classes
        dropout: Dropout rate
        freeze_base: Whether to freeze base model
        device: Device to use
        gradient_checkpointing: Use gradient checkpointing (saves memory)
        use_torch_compile: Use PyTorch 2.0 compile for faster execution (requires PyTorch >= 2.0)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Creating model: {model_name}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")
    
    model = AuthorClassifier(
        model_name=model_name,
        num_classes=num_classes,
        dropout=dropout,
        freeze_base=freeze_base,
        gradient_checkpointing=gradient_checkpointing
    )
    
    # Get tokenizer BEFORE compiling (compiled models may not expose methods)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model = model.to(device)
    
    # Count parameters before compilation
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Use PyTorch 2.0 compile for faster execution (if available)
    if use_torch_compile and hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile() for faster execution...")
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: Could not compile model: {e}")
            print("Continuing without compilation...")
    
    return model, tokenizer


