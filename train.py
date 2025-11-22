"""
Training script for Victorian Era Authorship Attribution model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scheduler: Optional = None,
    use_amp: bool = True,
    accumulation_steps: int = 1
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        scheduler: Optional learning rate scheduler
        use_amp: Use automatic mixed precision for faster training
        accumulation_steps: Number of gradient accumulation steps
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Setup mixed precision training (use new API if available, fallback to old)
    if use_amp and device == 'cuda':
        try:
            # Try new PyTorch 2.0+ API first
            scaler = torch.amp.GradScaler('cuda')
            use_new_amp = True
        except (AttributeError, TypeError):
            # Fallback to old API
            scaler = torch.cuda.amp.GradScaler()
            use_new_amp = False
    else:
        scaler = None
        use_new_amp = False
    
    progress_bar = tqdm(train_loader, desc="Training")
    optimizer.zero_grad()
    
    num_batches = len(train_loader)
    
    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        if scaler:
            if use_new_amp:
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    loss = loss / accumulation_steps
            else:
                with torch.cuda.amp.autocast():
                    logits = model(input_ids, attention_mask)
                    loss = nn.CrossEntropyLoss()(logits, labels)
                    loss = loss / accumulation_steps
        else:
            logits = model(input_ids, attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss = loss / accumulation_steps
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights every accumulation_steps or at the end of epoch
        should_update = (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches
        
        if should_update:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            optimizer.zero_grad()
        
        # Metrics (use unscaled loss for reporting)
        total_loss += loss.item() * accumulation_steps
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item() * accumulation_steps,
            'acc': correct / total
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device to use
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    # Use mixed precision for validation if on CUDA
    use_amp_val = device == 'cuda'
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            # Forward pass with mixed precision
            if use_amp_val:
                try:
                    # Try new API first
                    with torch.amp.autocast('cuda'):
                        logits = model(input_ids, attention_mask)
                        loss = nn.CrossEntropyLoss()(logits, labels)
                except (AttributeError, TypeError):
                    # Fallback to old API
                    with torch.cuda.amp.autocast():
                        logits = model(input_ids, attention_mask)
                        loss = nn.CrossEntropyLoss()(logits, labels)
            else:
                logits = model(input_ids, attention_mask)
                loss = nn.CrossEntropyLoss()(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'acc': correct / total
            })
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_dir: str = './models',
    model_name: str = 'author_classifier',
    use_amp: bool = True,
    accumulation_steps: int = 1,
    early_stopping_patience: int = None
) -> Dict:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to use
        save_dir: Directory to save models
        model_name: Name for saved model
        use_amp: Use automatic mixed precision for faster training
        accumulation_steps: Number of gradient accumulation steps (effective batch size = batch_size * accumulation_steps)
        early_stopping_patience: Stop training if validation accuracy doesn't improve for N epochs (None to disable)
    
    Returns:
        Dictionary with training history
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0.1 * total_steps,
        num_training_steps=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Total training steps: {total_steps}")
    print(f"Mixed precision (AMP): {use_amp and device == 'cuda'}")
    print(f"Gradient accumulation steps: {accumulation_steps}")
    if accumulation_steps > 1:
        print(f"Effective batch size: {train_loader.batch_size * accumulation_steps}")
    if early_stopping_patience:
        print(f"Early stopping patience: {early_stopping_patience} epochs")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, scheduler, use_amp, accumulation_steps)
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Record history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # Save best model and check early stopping
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0  # Reset patience counter
            model_path = os.path.join(save_dir, f'{model_name}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'history': history
            }, model_path)
            print(f"Saved best model with val accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if early_stopping_patience and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered! No improvement for {early_stopping_patience} epochs.")
            print(f"Best validation accuracy: {best_val_acc:.4f}")
            break
        
        # Save checkpoint (only every few epochs to reduce I/O overhead)
        if (epoch + 1) % max(1, num_epochs // 3) == 0 or epoch == num_epochs - 1:
            checkpoint_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'history': history
            }, checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(save_dir, f'{model_name}_final.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_metrics['accuracy'],
        'history': history
    }, final_model_path)
    
    # Plot training history
    plot_training_history(history, save_dir, model_name)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history


def plot_training_history(history: Dict, save_dir: str, model_name: str):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_dir: Directory to save plots
        model_name: Model name for file naming
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'{model_name}_training_history.png')
    plt.savefig(plot_path)
    print(f"Saved training history plot to {plot_path}")
    plt.close()


