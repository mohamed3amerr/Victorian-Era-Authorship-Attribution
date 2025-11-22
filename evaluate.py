"""
Evaluation script for Victorian Era Authorship Attribution model.
Computes accuracy, F1-score, and confusion matrix.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
from typing import Dict, Optional


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    id_to_author: Optional[Dict] = None
) -> Dict:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to use
        id_to_author: Optional mapping from label ID to author ID
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Evaluating")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    f1_macro = f1_score(all_labels, all_predictions, average='macro')
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(
        all_labels,
        all_predictions,
        output_dict=True
    )
    
    results = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report
    }
    
    return results


def plot_confusion_matrix(
    cm: np.ndarray,
    save_path: str,
    id_to_author: Optional[Dict] = None,
    top_n: int = 20
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save the plot
        id_to_author: Optional mapping from label ID to author ID
        top_n: Number of authors to show (if too many)
    """
    # If too many classes, show top N most frequent
    if cm.shape[0] > top_n:
        # Sum rows and columns to find most frequent classes
        row_sums = cm.sum(axis=1)
        col_sums = cm.sum(axis=0)
        total_sums = row_sums + col_sums
        top_indices = np.argsort(total_sums)[-top_n:]
        
        cm_subset = cm[np.ix_(top_indices, top_indices)]
        labels = [f"Author {i}" if id_to_author is None else f"Author {id_to_author.get(i, i)}" 
                  for i in top_indices]
    else:
        cm_subset = cm
        labels = [f"Author {i}" if id_to_author is None else f"Author {id_to_author.get(i, i)}" 
                  for i in range(cm.shape[0])]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm_subset,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Author')
    plt.xlabel('Predicted Author')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved confusion matrix to {save_path}")
    plt.close()


def print_evaluation_results(results: Dict, id_to_author: Optional[Dict] = None):
    """
    Print evaluation results.
    
    Args:
        results: Results dictionary from evaluate_model
        id_to_author: Optional mapping from label ID to author ID
    """
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score (Macro): {results['f1_macro']:.4f}")
    print(f"F1-Score (Weighted): {results['f1_weighted']:.4f}")
    print("\n" + "-"*60)
    print("Classification Report:")
    print("-"*60)
    
    # Print classification report
    report = results['classification_report']
    if 'accuracy' in report:
        print(f"Overall Accuracy: {report['accuracy']:.4f}")
    
    print(f"\n{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-"*60)
    
    for key, value in report.items():
        if isinstance(value, dict) and 'precision' in value:
            try:
             class_name = f"Author {id_to_author.get(int(key), key)}" if id_to_author else f"Author {key}"
            except ValueError:
             class_name = key
            print(f"{class_name:<15} {value['precision']:<12.4f} {value['recall']:<12.4f} "
                  f"{value['f1-score']:<12.4f} {value['support']:<10}")
    
    if 'macro avg' in report:
        print("-"*60)
        print(f"{'Macro Avg':<15} {report['macro avg']['precision']:<12.4f} "
              f"{report['macro avg']['recall']:<12.4f} {report['macro avg']['f1-score']:<12.4f} "
              f"{report['macro avg']['support']:<10}")
    
    if 'weighted avg' in report:
        print(f"{'Weighted Avg':<15} {report['weighted avg']['precision']:<12.4f} "
              f"{report['weighted avg']['recall']:<12.4f} {report['weighted avg']['f1-score']:<12.4f} "
              f"{report['weighted avg']['support']:<10}")
    
    print("="*60)


def save_evaluation_results(
    results: Dict,
    save_dir: str,
    model_name: str,
    id_to_author: Optional[Dict] = None
):
    """
    Save evaluation results to files.
    
    Args:
        results: Results dictionary
        save_dir: Directory to save results
        model_name: Model name for file naming
        id_to_author: Optional mapping from label ID to author ID
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save confusion matrix plot
    cm_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], cm_path, id_to_author)
    
    # Save metrics to text file
    metrics_path = os.path.join(save_dir, f'{model_name}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("EVALUATION METRICS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"F1-Score (Macro): {results['f1_macro']:.4f}\n")
        f.write(f"F1-Score (Weighted): {results['f1_weighted']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write("-"*60 + "\n")
        f.write(str(results['classification_report']))
    
    print(f"Saved evaluation results to {save_dir}")


