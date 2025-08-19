import numpy as np
import pandas as pd
from typing import Tuple
from imblearn.under_sampling import NearMiss
from utils.progress_display import create_progress_display


class NearMissSampler:
    """NearMiss Under Sampler wrapper.
    
    NearMiss is a classic distance-based undersampling method that selects
    samples from the majority class based on their distance to minority samples.
    It's more intelligent than random sampling while being much faster than
    complex methods like SMOTEENN.
    """
    
    def __init__(self, 
                 version: int = 1,
                 n_neighbors: int = 3,
                 sampling_strategy: str = 'auto',
                 n_jobs: int = 1):
        """Initialize the NearMiss sampler.
        
        Args:
            version (int): NearMiss version (1, 2, or 3) - default: 1
            n_neighbors (int): Number of neighbors to consider - default: 3
            sampling_strategy (str): Sampling strategy (default: 'auto')
            n_jobs (int): Number of parallel jobs (default: 1)
        """
        self.version = version
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.n_jobs = n_jobs
        self.progress = create_progress_display()
        
    def balance_dataset(self, x_train: pd.DataFrame, x_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series,
                       fault_type_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Balance training dataset using NearMiss.
        
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
            status.update("Initializing NearMiss sampler...")
            
            # Check if we have any positive samples in training data
            if y_train.sum() == 0:
                self.progress.display_warning("No positive samples in training data")
                return x_train, x_test, y_train, y_test
            
            # Check if we have any negative samples in training data
            if (~y_train).sum() == 0:
                self.progress.display_warning("No negative samples in training data")
                return x_train, x_test, y_train, y_test
            
            status.update("Applying NearMiss undersampling...")
            
            try:
                # Initialize NearMiss sampler
                nearmiss = NearMiss(
                    version=self.version,
                    n_neighbors=self.n_neighbors,
                    sampling_strategy=self.sampling_strategy,
                    n_jobs=self.n_jobs
                )
                
                # Apply NearMiss to training data
                x_train_resampled, y_train_resampled = nearmiss.fit_resample(x_train, y_train)
                
                # Convert back to pandas format
                x_train_balanced = pd.DataFrame(
                    x_train_resampled, 
                    columns=x_train.columns
                )
                y_train_balanced = pd.Series(y_train_resampled, name=y_train.name)
                
                # Test data remains unchanged
                x_test_balanced = x_test.copy()
                y_test_balanced = y_test.copy()
                
                status.complete("NearMiss undersampling completed")
                
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
                self.progress.display_warning(f"NearMiss undersampling failed, falling back to random sampling: {str(e)}")
                # Fall back to simple random undersampling
                from .simple_undersampler import SimpleUndersamplerWrapper
                simple_sampler = SimpleUndersamplerWrapper(
                    balance_ratio=1.0,
                    random_state=42
                )
                return simple_sampler.balance_dataset(x_train, x_test, y_train, y_test, fault_type_col)
    
    def _display_sampling_statistics(self, x_train_orig, y_train_orig, x_train_bal, y_train_bal):
        """Display NearMiss sampling statistics."""
        removed_samples = len(x_train_orig) - len(x_train_bal)
        stats_data = {
            "Sampling Method": "NearMiss",
            "Version": self.version,
            "N Neighbors": self.n_neighbors,
            "Sampling Strategy": self.sampling_strategy,
            "Original Training Size": len(x_train_orig),
            "Resampled Training Size": len(x_train_bal),
            "Removed Samples": removed_samples,
            "Reduction Rate": f"{(removed_samples / len(x_train_orig) * 100):.1f}%"
        }
        
        self.progress.display_results_table("NearMiss Sampling Statistics", stats_data)
    
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