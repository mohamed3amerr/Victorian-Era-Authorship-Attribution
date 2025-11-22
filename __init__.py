"""
Victorian Era Authorship Attribution Package
"""

__version__ = "1.0.0"

from .data_preprocessing import (
    VictorianAuthorDataset,
    load_data,
    preprocess_data,
    create_data_loaders
)

from .model import (
    AuthorClassifier,
    create_model
)

from .train import (
    train_model,
    train_epoch,
    validate
)

from .evaluate import (
    evaluate_model,
    print_evaluation_results,
    save_evaluation_results
)

from .analyze import (
    analyze_author_difficulty,
    generate_analysis_report
)

__all__ = [
    'VictorianAuthorDataset',
    'load_data',
    'preprocess_data',
    'create_data_loaders',
    'AuthorClassifier',
    'create_model',
    'train_model',
    'train_epoch',
    'validate',
    'evaluate_model',
    'print_evaluation_results',
    'save_evaluation_results',
    'analyze_author_difficulty',
    'generate_analysis_report'
]


