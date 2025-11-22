"""
Analysis script for Victorian Era Authorship Attribution.
Identifies which authors are easier or harder to classify.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Optional
import os


def analyze_author_difficulty(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    id_to_author: Optional[Dict] = None,
    save_dir: str = './results'
) -> pd.DataFrame:
    """
    Analyze which authors are easier or harder to classify.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        id_to_author: Optional mapping from label ID to author ID
        save_dir: Directory to save analysis results
    
    Returns:
        DataFrame with per-author metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Get classification report
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )
    
    # Extract per-class metrics
    author_metrics = []
    unique_authors = sorted(np.unique(np.concatenate([y_true, y_pred])))
    
    for author_id in unique_authors:
        author_str = str(author_id)
        if author_str in report and isinstance(report[author_str], dict):
            metrics = report[author_str]
            author_name = f"Author {author_id}" if id_to_author is None else f"Author {id_to_author.get(author_id, author_id)}"
            
            author_metrics.append({
                'author_id': author_id,
                'author_name': author_name,
                'precision': metrics.get('precision', 0.0),
                'recall': metrics.get('recall', 0.0),
                'f1_score': metrics.get('f1-score', 0.0),
                'support': metrics.get('support', 0)
            })
    
    df_metrics = pd.DataFrame(author_metrics)
    
    # Calculate difficulty score (inverse of F1-score)
    df_metrics['difficulty'] = 1.0 - df_metrics['f1_score']
    df_metrics = df_metrics.sort_values('f1_score', ascending=True)
    
    # Save to CSV
    csv_path = os.path.join(save_dir, 'author_difficulty_analysis.csv')
    df_metrics.to_csv(csv_path, index=False)
    print(f"Saved author difficulty analysis to {csv_path}")
    
    return df_metrics


def plot_author_performance(
    df_metrics: pd.DataFrame,
    save_path: str,
    top_n: int = 20
):
    """
    Plot author performance metrics.
    
    Args:
        df_metrics: DataFrame with author metrics
        save_path: Path to save the plot
        top_n: Number of authors to show
    """
    # Sort by F1-score for better visualization
    df_sorted = df_metrics.sort_values('f1_score', ascending=True)
    
    # Show top N hardest and easiest authors
    n_hardest = min(top_n, len(df_sorted))
    df_plot = pd.concat([
        df_sorted.head(n_hardest),  # Hardest
        df_sorted.tail(n_hardest)   # Easiest
    ]).drop_duplicates()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. F1-Score by Author
    ax1 = axes[0, 0]
    colors = ['red' if x < 0.5 else 'green' for x in df_plot['f1_score']]
    ax1.barh(range(len(df_plot)), df_plot['f1_score'], color=colors)
    ax1.set_yticks(range(len(df_plot)))
    ax1.set_yticklabels(df_plot['author_name'], fontsize=8)
    ax1.set_xlabel('F1-Score')
    ax1.set_title('F1-Score by Author (Red: <0.5, Green: ≥0.5)')
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax1.grid(axis='x', alpha=0.3)
    
    # 2. Precision vs Recall
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df_metrics['precision'],
        df_metrics['recall'],
        c=df_metrics['f1_score'],
        cmap='RdYlGn',
        s=100,
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.set_title('Precision vs Recall (Color = F1-Score)')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='F1-Score')
    
    # 3. Support (Number of samples) vs F1-Score
    ax3 = axes[1, 0]
    ax3.scatter(
        df_metrics['support'],
        df_metrics['f1_score'],
        alpha=0.6,
        s=100,
        edgecolors='black',
        linewidth=0.5
    )
    ax3.set_xlabel('Support (Number of Samples)')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score vs Support')
    ax3.grid(True, alpha=0.3)
    
    # 4. Difficulty Distribution
    ax4 = axes[1, 1]
    ax4.hist(df_metrics['f1_score'], bins=20, edgecolor='black', alpha=0.7)
    ax4.set_xlabel('F1-Score')
    ax4.set_ylabel('Number of Authors')
    ax4.set_title('Distribution of F1-Scores')
    ax4.axvline(x=df_metrics['f1_score'].mean(), color='red', linestyle='--', label=f'Mean: {df_metrics["f1_score"].mean():.3f}')
    ax4.axvline(x=df_metrics['f1_score'].median(), color='blue', linestyle='--', label=f'Median: {df_metrics["f1_score"].median():.3f}')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved author performance plot to {save_path}")
    plt.close()


def analyze_confusion_patterns(
    cm: np.ndarray,
    id_to_author: Optional[Dict] = None,
    save_dir: str = './results',
    top_n: int = 10
):
    """
    Analyze confusion patterns to identify commonly confused authors.
    
    Args:
        cm: Confusion matrix
        id_to_author: Optional mapping from label ID to author ID
        save_dir: Directory to save results
        top_n: Number of top confusions to show
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Find most common confusions (off-diagonal elements)
    confusions = []
    n_classes = cm.shape[0]
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                author_i = f"Author {i}" if id_to_author is None else f"Author {id_to_author.get(i, i)}"
                author_j = f"Author {j}" if id_to_author is None else f"Author {id_to_author.get(j, j)}"
                confusions.append({
                    'true_author': author_i,
                    'predicted_author': author_j,
                    'count': cm[i, j],
                    'true_id': i,
                    'predicted_id': j
                })
    
    df_confusions = pd.DataFrame(confusions)
    df_confusions = df_confusions.sort_values('count', ascending=False)
    
    # Save top confusions
    csv_path = os.path.join(save_dir, 'top_confusions.csv')
    df_confusions.head(top_n * 2).to_csv(csv_path, index=False)
    print(f"Saved top confusions to {csv_path}")
    
    # Print top confusions
    print("\n" + "="*60)
    print(f"TOP {top_n} MOST COMMON CONFUSIONS")
    print("="*60)
    print(f"{'True Author':<25} {'Predicted Author':<25} {'Count':<10}")
    print("-"*60)
    for _, row in df_confusions.head(top_n).iterrows():
        print(f"{row['true_author']:<25} {row['predicted_author']:<25} {row['count']:<10}")
    print("="*60)
    
    return df_confusions


def generate_analysis_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cm: np.ndarray,
    id_to_author: Optional[Dict] = None,
    save_dir: str = './results'
):
    """
    Generate comprehensive analysis report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        cm: Confusion matrix
        id_to_author: Optional mapping from label ID to author ID
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("="*60)
    
    # Author difficulty analysis
    df_metrics = analyze_author_difficulty(y_true, y_pred, id_to_author, save_dir)
    
    # Plot author performance
    plot_path = os.path.join(save_dir, 'author_performance_analysis.png')
    plot_author_performance(df_metrics, plot_path)
    
    # Confusion pattern analysis
    df_confusions = analyze_confusion_patterns(cm, id_to_author, save_dir)
    
    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total Authors: {len(df_metrics)}")
    print(f"Mean F1-Score: {df_metrics['f1_score'].mean():.4f}")
    print(f"Median F1-Score: {df_metrics['f1_score'].median():.4f}")
    print(f"Std F1-Score: {df_metrics['f1_score'].std():.4f}")
    print(f"Min F1-Score: {df_metrics['f1_score'].min():.4f} ({df_metrics.loc[df_metrics['f1_score'].idxmin(), 'author_name']})")
    print(f"Max F1-Score: {df_metrics['f1_score'].max():.4f} ({df_metrics.loc[df_metrics['f1_score'].idxmax(), 'author_name']})")
    
    # Authors with F1 < 0.5 (hard to classify)
    hard_authors = df_metrics[df_metrics['f1_score'] < 0.5]
    print(f"\nAuthors with F1-Score < 0.5 (Hard to Classify): {len(hard_authors)}")
    if len(hard_authors) > 0:
        print(hard_authors[['author_name', 'f1_score', 'support']].to_string(index=False))
    
    # Authors with F1 >= 0.8 (easy to classify)
    easy_authors = df_metrics[df_metrics['f1_score'] >= 0.8]
    print(f"\nAuthors with F1-Score >= 0.8 (Easy to Classify): {len(easy_authors)}")
    if len(easy_authors) > 0:
        print(easy_authors[['author_name', 'f1_score', 'support']].to_string(index=False))
    
    print("="*60)


