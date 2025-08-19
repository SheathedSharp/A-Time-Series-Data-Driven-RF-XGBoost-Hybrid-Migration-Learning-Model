import numpy as np
import pandas as pd
from typing import Tuple
from imblearn.under_sampling import NeighbourhoodCleaningRule
from utils.progress_display import create_progress_display


class NCRSampler:
    """Neighbourhood Cleaning Rule (NCR) Under Sampler wrapper.
    
    NCR is a sophisticated undersampling method that removes samples based on 
    neighborhood analysis. It removes samples that are either:
    1. Majority class samples whose neighbors are mostly minority class
    2. Minority class samples that are misclassified by their neighbors
    
    This creates cleaner decision boundaries compared to random sampling.
    """
    
    def __init__(self, 
                 n_neighbors: int = 3,
                 threshold_cleaning: float = 0.5,
                 sampling_strategy: str = 'auto',
                 random_state: int = 42,
                 n_jobs: int = 1):
        """Initialize the NCR sampler.
        
        Args:
            n_neighbors (int): Number of neighbors to use for cleaning (default: 3)
            threshold_cleaning (float): Threshold for cleaning (default: 0.5)
            sampling_strategy (str): Sampling strategy (default: 'auto')
            random_state (int): Random seed for reproducibility
            n_jobs (int): Number of parallel jobs (default: 1)
        """
        self.n_neighbors = n_neighbors
        self.threshold_cleaning = threshold_cleaning
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.progress = create_progress_display()
        
    def balance_dataset(self, x_train: pd.DataFrame, x_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series,
                       fault_type_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Balance training dataset using Neighbourhood Cleaning Rule.
        
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
            status.update("Initializing Neighbourhood Cleaning Rule sampler...")
            
            # Check if we have any positive samples in training data
            if y_train.sum() == 0:
                self.progress.display_warning("No positive samples in training data")
                return x_train, x_test, y_train, y_test
            
            # Check if we have any negative samples in training data
            if (~y_train).sum() == 0:
                self.progress.display_warning("No negative samples in training data")
                return x_train, x_test, y_train, y_test
            
            # Check dataset size for performance
            if len(x_train) > 100000:
                self.progress.display_warning(f"Large dataset ({len(x_train)} samples), NCR may be slow...")
                # Use smaller n_neighbors for large datasets
                n_neighbors = min(self.n_neighbors, 3)
            else:
                n_neighbors = self.n_neighbors
            
            ncr = NeighbourhoodCleaningRule(
                n_neighbors=n_neighbors,
                threshold_cleaning=self.threshold_cleaning,
                sampling_strategy=self.sampling_strategy,
                n_jobs=self.n_jobs
            )
            
            status.update("Applying Neighbourhood Cleaning Rule to training data...")
            
            try:
                # Apply NCR to training data
                x_train_resampled, y_train_resampled = ncr.fit_resample(x_train, y_train)
                
                # Convert back to pandas format
                x_train_balanced = pd.DataFrame(
                    x_train_resampled, 
                    columns=x_train.columns
                )
                y_train_balanced = pd.Series(y_train_resampled, name=y_train.name)
                
                # Test data remains unchanged
                x_test_balanced = x_test.copy()
                y_test_balanced = y_test.copy()
                
                status.complete("Neighbourhood Cleaning Rule sampling completed")
                
                # Display statistics
                self._display_sampling_statistics(x_train, y_train, x_train_balanced, y_train_balanced, n_neighbors)
                self._display_combined_balance_stats(
                    train_original=y_train,
                    train_balanced=y_train_balanced,
                    test_original=y_test,
                    test_balanced=y_test_balanced
                )
                
                return x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced
                
            except Exception as e:
                self.progress.display_warning(f"Neighbourhood Cleaning Rule failed, falling back to random sampling: {str(e)}")
                # Fall back to simple random undersampling
                from .simple_undersampler import SimpleUndersamplerWrapper
                simple_sampler = SimpleUndersamplerWrapper(
                    balance_ratio=1.0,
                    random_state=self.random_state
                )
                return simple_sampler.balance_dataset(x_train, x_test, y_train, y_test, fault_type_col)
    
    def _display_sampling_statistics(self, x_train_orig, y_train_orig, x_train_bal, y_train_bal, actual_neighbors):
        """Display NCR sampling statistics."""
        removed_samples = len(x_train_orig) - len(x_train_bal)
        stats_data = {
            "Sampling Method": "Neighbourhood Cleaning Rule",
            "N Neighbors": actual_neighbors,
            "Threshold Cleaning": self.threshold_cleaning,
            "Sampling Strategy": self.sampling_strategy,
            "Original Training Size": len(x_train_orig),
            "Resampled Training Size": len(x_train_bal),
            "Removed Samples": removed_samples,
            "Removal Rate": f"{(removed_samples / len(x_train_orig) * 100):.1f}%"
        }
        
        self.progress.display_results_table("Neighbourhood Cleaning Rule Sampling Statistics", stats_data)
    
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