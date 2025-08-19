import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.utils import resample
from utils.progress_display import create_progress_display


class SimpleUndersamplerWrapper:
    """Simple Undersampling wrapper using sklearn's resample.
    
    This is a fast and simple undersampling method that randomly reduces
    the majority class to balance the dataset. Much faster than neighbor-based methods.
    """
    
    def __init__(self, 
                 balance_ratio: float = 1.0,
                 random_state: int = 42):
        """Initialize the Simple Under Sampler.
        
        Args:
            balance_ratio (float): Ratio of majority to minority class after sampling (default: 1.0 for perfect balance)
            random_state (int): Random seed for reproducibility
        """
        self.balance_ratio = balance_ratio
        self.random_state = random_state
        self.progress = create_progress_display()
        
    def balance_dataset(self, x_train: pd.DataFrame, x_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series,
                       fault_type_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Balance training dataset using simple undersampling.
        
        Args:
            x_train (pd.DataFrame): Training features
            x_test (pd.DataFrame): Test features  
            y_train (pd.Series): Training labels
            y_test (pd.Series): Test labels
            fault_type_col (str): Optional fault type column (not used)
            
        Returns:
            Tuple containing balanced training data and original test data
        """
        with self.progress.sampling_status() as status:
            status.update("Initializing Simple Under Sampler...")
            
            # Check if we have any positive samples in training data
            positive_count = y_train.sum()
            negative_count = (~y_train).sum()
            
            if positive_count == 0:
                self.progress.display_warning("No positive samples in training data")
                return x_train, x_test, y_train, y_test
            
            if negative_count == 0:
                self.progress.display_warning("No negative samples in training data")
                return x_train, x_test, y_train, y_test
            
            status.update("Applying simple undersampling to training data...")
            
            try:
                # Calculate target size for majority class
                target_majority_size = int(positive_count * self.balance_ratio)
                
                if target_majority_size >= negative_count:
                    # No need to undersample
                    self.progress.display_warning("Dataset already balanced or target ratio too high")
                    return x_train, x_test, y_train, y_test
                
                # Separate positive and negative samples
                positive_mask = y_train == 1
                negative_mask = y_train == 0
                
                # Get positive samples (keep all)
                x_positive = x_train[positive_mask]
                y_positive = y_train[positive_mask]
                
                # Get negative samples and undersample
                x_negative = x_train[negative_mask]
                y_negative = y_train[negative_mask]
                
                # Randomly sample from majority class
                x_negative_sampled, y_negative_sampled = resample(
                    x_negative, y_negative,
                    n_samples=target_majority_size,
                    random_state=self.random_state,
                    replace=False
                )
                
                # Combine balanced samples
                x_train_balanced = pd.concat([x_positive, x_negative_sampled], ignore_index=True)
                y_train_balanced = pd.concat([y_positive, y_negative_sampled], ignore_index=True)
                
                # Shuffle the combined dataset
                combined_indices = np.arange(len(x_train_balanced))
                np.random.seed(self.random_state)
                np.random.shuffle(combined_indices)
                
                x_train_balanced = x_train_balanced.iloc[combined_indices].reset_index(drop=True)
                y_train_balanced = y_train_balanced.iloc[combined_indices].reset_index(drop=True)
                
                # Test data remains unchanged
                x_test_balanced = x_test.copy()
                y_test_balanced = y_test.copy()
                
                status.complete("Simple undersampling completed")
                
                # Display statistics
                self._display_sampling_statistics(x_train, y_train, x_train_balanced, y_train_balanced)
                self._display_combined_balance_stats(
                    train_original=y_train,
                    train_balanced=y_train_balanced,
                    test_original=y_test,
                    test_balanced=y_test_balanced
                )
                
                return x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced
                
            except Exception as e:
                self.progress.display_warning(f"Simple undersampling failed: {str(e)}")
                return x_train, x_test, y_train, y_test
    
    def _display_sampling_statistics(self, x_train_orig, y_train_orig, x_train_bal, y_train_bal):
        """Display simple undersampling statistics."""
        stats_data = {
            "Sampling Method": "Simple Undersampling",
            "Balance Ratio": self.balance_ratio,
            "Original Training Size": len(x_train_orig),
            "Resampled Training Size": len(x_train_bal),
            "Size Reduction": f"{((len(x_train_orig) - len(x_train_bal)) / len(x_train_orig) * 100):.1f}%"
        }
        
        self.progress.display_results_table("Simple Undersampling Statistics", stats_data)
    
    def _display_combined_balance_stats(self, train_original, train_balanced,
                                      test_original, test_balanced):
        """Display combined statistics about the balancing operation."""
        # Calculate statistics
        train_orig_stats = {
            'total': len(train_original),
            'positive': train_original.sum(),
            'negative': (~train_original).sum(),
            'ratio': (~train_original).sum() / max(train_original.sum(), 1)
        }

        train_bal_stats = {
            'total': len(train_balanced),
            'positive': train_balanced.sum(),
            'negative': (~train_balanced).sum(),
            'ratio': (~train_balanced).sum() / max(train_balanced.sum(), 1)
        }

        test_orig_stats = {
            'total': len(test_original),
            'positive': test_original.sum(),
            'negative': (~test_original).sum(),
            'ratio': (~test_original).sum() / max(test_original.sum(), 1)
        }

        test_bal_stats = {
            'total': len(test_balanced),
            'positive': test_balanced.sum(),
            'negative': (~test_balanced).sum(),
            'ratio': (~test_balanced).sum() / max(test_balanced.sum(), 1)
        }

        # Create balance statistics
        balance_data = {
            "Training Original - Total": train_orig_stats['total'],
            "Training Original - Positive": train_orig_stats['positive'],
            "Training Original - Negative": train_orig_stats['negative'],
            "Training Original - Neg/Pos Ratio": f"{train_orig_stats['ratio']:.2f}",
            "Training Balanced - Total": train_bal_stats['total'],
            "Training Balanced - Positive": train_bal_stats['positive'],
            "Training Balanced - Negative": train_bal_stats['negative'],
            "Training Balanced - Neg/Pos Ratio": f"{train_bal_stats['ratio']:.2f}",
            "Test Original - Total": test_orig_stats['total'],
            "Test Original - Positive": test_orig_stats['positive'],
            "Test Original - Negative": test_orig_stats['negative'],
            "Test Original - Neg/Pos Ratio": f"{test_orig_stats['ratio']:.2f}",
            "Test Balanced - Total": test_bal_stats['total'],
            "Test Balanced - Positive": test_bal_stats['positive'],
            "Test Balanced - Negative": test_bal_stats['negative'],
            "Test Balanced - Neg/Pos Ratio": f"{test_bal_stats['ratio']:.2f}"
        }

        self.progress.display_results_table("Dataset Balance Statistics", balance_data)