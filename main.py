"""
Main script for Victorian Era Authorship Attribution project.
Orchestrates data loading, model training, evaluation, and analysis.
"""

import argparse
import os
import torch
import numpy as np
from src.data_preprocessing import preprocess_data, create_data_loaders
from src.model import create_model
from src.train import train_model
from src.evaluate import evaluate_model, print_evaluation_results, save_evaluation_results
from src.analyze import generate_analysis_report


def main():
    parser = argparse.ArgumentParser(description='Victorian Era Authorship Attribution')
    
    # Data arguments
    parser.add_argument('--train_csv', type=str, 
                       default='dataset/Gungor_2018_VictorianAuthorAttribution_data-train.csv',
                       help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str,
                       default='dataset/Gungor_2018_VictorianAuthorAttribution_data.csv',
                       help='Path to test CSV file')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Sample size for faster testing (None for full dataset)')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for validation')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='distilbert-base-uncased',
                       choices=['bert-base-uncased', 'roberta-base', 'distilbert-base-uncased'],
                       help='Base transformer model (distilbert-base-uncased is fastest, default)')
    parser.add_argument('--max_length', type=int, default=256,
                       help='Maximum sequence length (256 = 2x faster than 512, default: 256)')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--freeze_base', action='store_true',
                       help='Freeze base transformer model')
    
    # Training arguments
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'evaluate', 'analyze', 'full'],
                       help='Mode: train, evaluate, analyze, or full')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs (3-4 is usually sufficient, default: 3)')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size (use with accumulation_steps, default: 8)')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for evaluation')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision (AMP) for faster training')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Disable automatic mixed precision')
    parser.add_argument('--accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps (effective batch size = batch_size * accumulation_steps, default: 4)')
    parser.add_argument('--early_stopping_patience', type=int, default=None,
                       help='Early stopping patience (stop if no improvement for N epochs, None to disable)')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='Use gradient checkpointing to save memory (slower but uses less memory)')
    parser.add_argument('--use_torch_compile', action='store_true',
                       help='Use PyTorch 2.0 compile for faster execution (requires PyTorch >= 2.0)')
    
    # Directory arguments
    parser.add_argument('--save_dir', type=str, default='./models',
                       help='Directory to save models')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--cache_dir', type=str, default='./cache',
                       help='Directory to cache tokenized data (default: ./cache)')
    
    # Device
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu), auto-detect if None')
    
    args = parser.parse_args()
    
    # Set device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*60)
    print("VICTORIAN ERA AUTHORSHIP ATTRIBUTION")
    print("="*60)
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print("="*60)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Load and preprocess data
    if args.mode in ['train', 'full']:
        print("\nLoading and preprocessing data...")
        data = preprocess_data(
            train_csv=args.train_csv,
            test_csv=args.test_csv if args.mode == 'full' else None,
            sample_size=args.sample_size,
            test_size=args.test_size
        )
        
        num_classes = data['num_classes']
        author_to_id = data['author_to_id']
        id_to_author = data['id_to_author']
        
        print(f"\nNumber of classes: {num_classes}")
        print(f"Training samples: {len(data['X_train'])}")
        print(f"Validation samples: {len(data['X_val'])}")
        if data['X_test'] is not None:
            print(f"Test samples: {len(data['X_test'])}")
    
    # Create model and tokenizer
    if args.mode in ['train', 'full']:
        print("\nCreating model...")
        model, tokenizer = create_model(
            model_name=args.model,
            num_classes=num_classes,
            dropout=args.dropout,
            freeze_base=args.freeze_base,
            device=device,
            gradient_checkpointing=args.gradient_checkpointing,
            use_torch_compile=args.use_torch_compile
        )
        
        # Create data loaders with caching
        print("\nCreating data loaders with disk caching...")
        loaders = create_data_loaders(
            X_train=data['X_train'],
            X_val=data['X_val'],
            y_train=data['y_train'],
            y_val=data['y_val'],
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            X_test=data['X_test'],
            y_test=data['y_test'],
            cache_dir=args.cache_dir,
            pin_memory=True
        )
        
        # Train model
        print("\nStarting training...")
        model_name = f"{args.model.replace('/', '_')}_authors"
        history = train_model(
            model=model,
            train_loader=loaders['train'],
            val_loader=loaders['val'],
            num_epochs=args.epochs,
            learning_rate=args.learning_rate,
            device=device,
            save_dir=args.save_dir,
            model_name=model_name,
            use_amp=args.use_amp,
            accumulation_steps=args.accumulation_steps,
            early_stopping_patience=args.early_stopping_patience
        )
        
        # Load best model for evaluation
        best_model_path = os.path.join(args.save_dir, f'{model_name}_best.pt')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"\nLoaded best model from {best_model_path}")
    
    # Evaluate model
    if args.mode in ['evaluate', 'full']:
        # Initialize model_name (will be set in train section for 'full' mode, or here for 'evaluate' mode)
        if args.mode == 'evaluate':
            # Load model from checkpoint
            if args.model_path is None:
                # Try to find best model
                model_name = f"{args.model.replace('/', '_')}_authors"
                args.model_path = os.path.join(args.save_dir, f'{model_name}_best.pt')
            else:
                # Extract model_name from model_path if not provided
                # Try to infer from path, or use default
                model_name = f"{args.model.replace('/', '_')}_authors"
            
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model not found at {args.model_path}")
            
            # Load data for evaluation
            data = preprocess_data(
                train_csv=args.train_csv,
                test_csv=args.test_csv,
                sample_size=args.sample_size,
                test_size=args.test_size
            )
            
            num_classes = data['num_classes']
            author_to_id = data['author_to_id']
            id_to_author = data['id_to_author']
            
            # Create model
            model, tokenizer = create_model(
                model_name=args.model,
                num_classes=num_classes,
                device=device,
                gradient_checkpointing=args.gradient_checkpointing,
                use_torch_compile=args.use_torch_compile
            )
            
            # Load checkpoint
            checkpoint = torch.load(args.model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {args.model_path}")
            
            # Create data loaders with caching
            loaders = create_data_loaders(
                X_train=data['X_train'],
                X_val=data['X_val'],
                y_train=data['y_train'],
                y_val=data['y_val'],
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                max_length=args.max_length,
                X_test=data['X_test'],
                y_test=data['y_test'],
                cache_dir=args.cache_dir,
                pin_memory=True
            )
        
        print("\nEvaluating on validation set...")
        val_results = evaluate_model(
            model=model,
            data_loader=loaders['val'],
            device=device,
            id_to_author=id_to_author
        )
        
        print_evaluation_results(val_results, id_to_author)
        save_evaluation_results(
            val_results,
            args.results_dir,
            f'{model_name}_val',
            id_to_author
        )
        
        if 'test' in loaders:
            print("\nEvaluating on test set...")
            test_results = evaluate_model(
                model=model,
                data_loader=loaders['test'],
                device=device,
                id_to_author=id_to_author
            )
            
            print_evaluation_results(test_results, id_to_author)
            save_evaluation_results(
                test_results,
                args.results_dir,
                f'{model_name}_test',
                id_to_author
            )
            
            # Analysis
            if args.mode in ['analyze', 'full']:
                print("\nGenerating analysis report...")
                generate_analysis_report(
                    y_true=test_results['labels'],
                    y_pred=test_results['predictions'],
                    cm=test_results['confusion_matrix'],
                    id_to_author=id_to_author,
                    save_dir=args.results_dir
                )
        else:
            # Use validation set for analysis if no test set
            if args.mode in ['analyze', 'full']:
                print("\nGenerating analysis report...")
                generate_analysis_report(
                    y_true=val_results['labels'],
                    y_pred=val_results['predictions'],
                    cm=val_results['confusion_matrix'],
                    id_to_author=id_to_author,
                    save_dir=args.results_dir
                )
    
    # Standalone analysis mode
    if args.mode == 'analyze' and args.model_path:
        # This would require loading results from a previous run
        print("Analysis mode requires evaluation results. Please run with --mode evaluate or full first.")
    
    print("\n" + "="*60)
    print("COMPLETED!")
    print("="*60)


if __name__ == '__main__':
    main()

