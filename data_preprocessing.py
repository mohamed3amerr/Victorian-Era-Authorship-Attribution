"""
Data preprocessing module for Victorian Era Authorship Attribution dataset.
Handles loading CSV data, converting word IDs to text, and preparing data for Transformer models.
Includes offline tokenization with disk caching for faster training.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import os
import pickle
import hashlib


class VictorianAuthorDataset(Dataset):
    """PyTorch Dataset for Victorian Era author attribution with disk-cached tokenization."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenizer, 
        max_length: int = 256,
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None
    ):
        """
        Args:
            texts: List of text strings
            labels: List of author labels (integers)
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache tokenized data (None to disable caching)
            cache_key: Unique key for this dataset (used for cache filename)
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_dir = cache_dir
        self.cache_key = cache_key
        
        # Try to load from cache
        if cache_dir and cache_key:
            cache_path = os.path.join(cache_dir, f"{cache_key}_tokenized.pt")
            if os.path.exists(cache_path):
                print(f"Loading tokenized data from cache: {cache_path}")
                try:
                    cached_data = torch.load(cache_path, map_location='cpu')
                    self.input_ids = cached_data['input_ids']
                    self.attention_masks = cached_data['attention_masks']
                    self.labels = cached_data['labels']
                    self.cached = True
                    print(f"Successfully loaded {len(self.input_ids)} samples from cache!")
                    return
                except Exception as e:
                    print(f"Warning: Could not load cache: {e}. Re-tokenizing...")
        
        # Tokenize all data
        print(f"Tokenizing {len(texts)} samples (this may take a few minutes)...")
        self.input_ids = []
        self.attention_masks = []
        
        # Tokenize in batches for efficiency
        batch_size = 1000
        for i in range(0, len(texts), batch_size):
            batch_texts = [str(text) for text in texts[i:i+batch_size]]
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )
            self.input_ids.append(encodings['input_ids'])
            self.attention_masks.append(encodings['attention_mask'])
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Tokenized {min(i + batch_size, len(texts))}/{len(texts)} samples...")
        
        # Concatenate all batches
        self.input_ids = torch.cat(self.input_ids, dim=0)
        self.attention_masks = torch.cat(self.attention_masks, dim=0)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.cached = False
        
        print(f"Tokenization complete! Saving to cache...")
        
        # Save to cache if cache_dir is provided
        if cache_dir and cache_key:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{cache_key}_tokenized.pt")
            try:
                torch.save({
                    'input_ids': self.input_ids,
                    'attention_masks': self.attention_masks,
                    'labels': self.labels
                }, cache_path)
                print(f"Tokenized data saved to cache: {cache_path}")
            except Exception as e:
                print(f"Warning: Could not save cache: {e}")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }


def load_data(csv_path: str, sample_size: int = None) -> pd.DataFrame:
    """
    Load CSV data file.
    
    Args:
        csv_path: Path to CSV file
        sample_size: Optional number of rows to sample (for testing)
    
    Returns:
        DataFrame with data
    """
    print(f"Loading data from {csv_path}...")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Try different encodings for CSV files
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            if sample_size:
                df = pd.read_csv(
                    csv_path, 
                    nrows=sample_size,
                    encoding=encoding,
                    low_memory=False,
                    engine='c'
                )
            else:
                # Read in chunks for large files
                chunk_list = []
                chunk_size = 10000
                for chunk in pd.read_csv(
                    csv_path, 
                    chunksize=chunk_size,
                    encoding=encoding,
                    low_memory=False,
                    engine='c'
                ):
                    chunk_list.append(chunk)
                df = pd.concat(chunk_list, ignore_index=True)
            
            print(f"Loaded {len(df)} rows using encoding: {encoding}")
            return df
            
        except UnicodeDecodeError:
            continue
        except Exception as e:
            if encoding == encodings[-1]:  # Last encoding attempt
                raise Exception(f"Failed to load CSV file with all encodings. Last error: {str(e)}")
            continue
    
    raise Exception("Failed to load CSV file with any encoding")


def word_ids_to_text(word_ids: str, vocab_map: Dict[int, str] = None) -> str:
    """
    Convert space-separated word IDs to text.
    Since we don't have the original vocabulary mapping, we'll use placeholder tokens.
    In practice, you'd need the vocabulary mapping from the dataset.
    
    Note: For best results, you should load the vocabulary mapping from the dataset.
    The current implementation uses generic tokens which may not be optimal.
    
    Args:
        word_ids: Space-separated string of word IDs
        vocab_map: Dictionary mapping word IDs to words (if available)
    
    Returns:
        Text string
    """
    try:
        if isinstance(word_ids, str):
            # Handle empty strings
            if not word_ids or word_ids.strip() == '':
                return ""
            # Split and convert to integers
            ids = []
            for x in word_ids.split():
                x_clean = x.strip()
                if x_clean and (x_clean.replace('.', '').replace('-', '').isdigit()):
                    try:
                        id_val = int(float(x_clean))
                        if id_val > 0:  # Only include positive IDs
                            ids.append(id_val)
                    except (ValueError, OverflowError):
                        continue
        elif isinstance(word_ids, (list, np.ndarray)):
            ids = []
            for x in word_ids:
                if pd.notna(x):
                    try:
                        id_val = int(float(x))
                        if id_val > 0:
                            ids.append(id_val)
                    except (ValueError, TypeError, OverflowError):
                        continue
        else:
            return ""
        
        if not ids:
            return ""
        
        if vocab_map:
            words = [vocab_map.get(id, f"[WORD_{id}]") for id in ids]
        else:
            # Use generic word tokens that BERT tokenizer can handle
            # Format: "w1234" instead of "word_1234" to be more tokenizer-friendly
            words = [f"w{id}" for id in ids]
        
        return " ".join(words)
    
    except Exception as e:
        # Return empty string on any error to prevent crashes
        print(f"Warning: Error converting word IDs to text: {str(e)}")
        return ""


def prepare_text_from_ids(df: pd.DataFrame, text_column: str = None) -> pd.Series:
    """
    Prepare text from word ID columns in the dataframe.
    The dataset has word IDs as separate columns (1000 columns for 1000 words per fragment).
    
    Args:
        df: DataFrame with word ID columns
        text_column: Name of column containing text/word IDs (if single column format)
    
    Returns:
        Series of text strings
    """
    # Identify metadata columns to exclude
    exclude_cols = ['aid', 'bid', 'ind', 'author', 'book', 'author_id', 'book_id']
    
    # If text_column is specified, use it
    if text_column:
        if text_column in df.columns:
            text_col = df[text_column]
            if text_col.dtype == 'object':
                texts = text_col.astype(str)
            else:
                texts = text_col.apply(lambda x: word_ids_to_text(str(x)))
            return texts
        else:
            raise ValueError(f"Column {text_column} not found")
    
    # Auto-detect text columns
    # Check if there's a single text column
    possible_text_cols = [col for col in df.columns if 'text' in col.lower() or 'txt' in col.lower()]
    if possible_text_cols:
        text_col = df[possible_text_cols[0]]
        if text_col.dtype == 'object':
            texts = text_col.astype(str)
        else:
            texts = text_col.apply(lambda x: word_ids_to_text(str(x)))
        return texts
    
    # Otherwise, assume word IDs are in separate columns (1000 columns)
    # Get all columns except metadata columns
    text_cols = [col for col in df.columns if col not in exclude_cols]
    
    if not text_cols:
        raise ValueError("Could not identify text columns")
    
    print(f"Found {len(text_cols)} word ID columns")
    print(f"Sample columns: {text_cols[:5]}")
    
    # Combine word ID columns into space-separated text
    # Each row has 1000 word IDs, we'll join them with spaces
    # Format as "w{id}" to be tokenizer-friendly
    def combine_word_ids(row):
        # Get all word IDs for this row, filter out NaN and convert to int then string
        word_ids = []
        for col in text_cols:
            try:
                val = row[col]
                if pd.notna(val) and val != '':
                    try:
                        # Handle different data types (int, float, string)
                        if isinstance(val, str):
                            val_clean = val.strip()
                            if val_clean and val_clean.replace('.', '').replace('-', '').isdigit():
                                word_id = int(float(val_clean))
                            else:
                                continue
                        else:
                            word_id = int(float(val))  # Handle float representation
                        
                        if word_id > 0:  # Skip padding zeros or invalid IDs
                            # Format as "w{id}" for better tokenization
                            word_ids.append(f"w{word_id}")
                    except (ValueError, TypeError, OverflowError):
                        continue
            except (KeyError, IndexError):
                # Skip if column doesn't exist in this row
                continue
        return " ".join(word_ids) if word_ids else ""
    
    texts = df.apply(combine_word_ids, axis=1)
    
    # Filter out empty texts
    non_empty = texts.str.len() > 0
    if non_empty.sum() < len(texts):
        print(f"Warning: {len(texts) - non_empty.sum()} empty texts found")
    
    return texts


def preprocess_data(
    train_csv: str,
    test_csv: str = None,
    sample_size: int = None,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    """
    Preprocess data for training and evaluation.
    
    Args:
        train_csv: Path to training CSV
        test_csv: Optional path to test CSV
        sample_size: Optional sample size for faster testing
        test_size: Fraction of data to use for validation
        random_state: Random seed
    
    Returns:
        Dictionary with preprocessed data and metadata
    """
    # Load training data
    df_train = load_data(train_csv, sample_size)
    
    # Get author labels
    if 'aid' in df_train.columns:
        y = df_train['aid'].values
    elif 'author' in df_train.columns:
        y = df_train['author'].values
    else:
        raise ValueError("Could not find author ID column (aid or author)")
    
    # Prepare text data
    texts = prepare_text_from_ids(df_train)
    
    # Create label mapping
    unique_authors = sorted(np.unique(y))
    author_to_id = {author: idx for idx, author in enumerate(unique_authors)}
    id_to_author = {idx: author for author, idx in author_to_id.items()}
    y_encoded = np.array([author_to_id[author] for author in y])
    
    print(f"Found {len(unique_authors)} unique authors")
    print(f"Author ID range: {min(unique_authors)} to {max(unique_authors)}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        texts, y_encoded,
        test_size=test_size,
        random_state=random_state,
        stratify=y_encoded
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Load test data if provided
    X_test = None
    y_test = None
    if test_csv and os.path.exists(test_csv):
        df_test = load_data(test_csv, sample_size)
        if 'aid' in df_test.columns:
            y_test_raw = df_test['aid'].values
        elif 'author' in df_test.columns:
            y_test_raw = df_test['author'].values
        else:
            y_test_raw = None
        
        if y_test_raw is not None:
            # Map test authors to same encoding
            y_test = np.array([author_to_id.get(author, -1) for author in y_test_raw])
            X_test = prepare_text_from_ids(df_test)
            print(f"Test samples: {len(X_test)}")
            # Filter out unknown authors
            valid_mask = y_test >= 0
            X_test = X_test[valid_mask]
            y_test = y_test[valid_mask]
            print(f"Test samples after filtering: {len(X_test)}")
    
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'author_to_id': author_to_id,
        'id_to_author': id_to_author,
        'num_classes': len(unique_authors)
    }


def create_data_loaders(
    X_train: pd.Series,
    X_val: pd.Series,
    y_train: np.ndarray,
    y_val: np.ndarray,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    X_test: pd.Series = None,
    y_test: np.ndarray = None,
    cache_dir: str = './cache',
    num_workers: int = None,
    pin_memory: bool = True
):
    """
    Create PyTorch DataLoaders for training, validation, and optionally test sets.
    Uses disk caching for tokenized data.
    
    Args:
        X_train: Training texts
        X_val: Validation texts
        y_train: Training labels
        y_val: Validation labels
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size
        max_length: Maximum sequence length
        X_test: Optional test texts
        y_test: Optional test labels
        cache_dir: Directory to cache tokenized data
        num_workers: Number of workers for DataLoader (None for auto-detect)
        pin_memory: Pin memory for faster GPU transfer
    
    Returns:
        Dictionary of DataLoaders
    """
    # Generate cache keys based on data hash
    def generate_cache_key(texts, labels, split_name):
        # Create a simple hash from first few samples
        sample_text = "".join([str(t) for t in texts[:10]]) if len(texts) > 0 else ""
        sample_labels = "".join([str(l) for l in labels[:10]]) if len(labels) > 0 else ""
        hash_input = f"{split_name}_{len(texts)}_{sample_text}_{sample_labels}_{max_length}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    # Determine num_workers
    if num_workers is None:
        import platform
        if platform.system() == 'Windows':
            num_workers = 0  # Windows has issues with multiprocessing
        else:
            num_workers = min(4, os.cpu_count() or 1)  # Use up to 4 workers on Linux/Mac
    
    # Create cache directory
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    
    # Create datasets with caching
    train_cache_key = generate_cache_key(X_train.tolist(), y_train.tolist(), 'train')
    train_dataset = VictorianAuthorDataset(
        X_train.tolist() if isinstance(X_train, pd.Series) else X_train,
        y_train.tolist(),
        tokenizer,
        max_length,
        cache_dir=cache_dir,
        cache_key=train_cache_key
    )
    
    val_cache_key = generate_cache_key(X_val.tolist(), y_val.tolist(), 'val')
    val_dataset = VictorianAuthorDataset(
        X_val.tolist() if isinstance(X_val, pd.Series) else X_val,
        y_val.tolist(),
        tokenizer,
        max_length,
        cache_dir=cache_dir,
        cache_key=val_cache_key
    )
    
    # Create data loaders with optimizations
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0  # Keep workers alive between epochs
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0
    )
    
    loaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    if X_test is not None and y_test is not None:
        test_cache_key = generate_cache_key(X_test.tolist(), y_test.tolist(), 'test')
        test_dataset = VictorianAuthorDataset(
            X_test.tolist() if isinstance(X_test, pd.Series) else X_test,
            y_test.tolist(),
            tokenizer,
            max_length,
            cache_dir=cache_dir,
            cache_key=test_cache_key
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            persistent_workers=num_workers > 0
        )
        loaders['test'] = test_loader
    
    return loaders
