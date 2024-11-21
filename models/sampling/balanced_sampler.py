import pandas as pd
from typing import List
from tabulate import tabulate


class BalancedSampler:
    """Custom balanced sampler for temporal data.
    
    This sampler maintains the temporal characteristics of the data
    while balancing positive and negative samples. It identifies continuous
    positive samples and creates balanced subsets with surrounding negative samples.
    """
    
    def __init__(self, negative_positive_ratio=10.0):
        """Initialize the balanced sampler.
        
        Args:
            negative_positive_ratio (float): Desired ratio of negative to positive samples
        """
        self.negative_positive_ratio = negative_positive_ratio
        
    def create_balanced_subsets(self, df):
        """Create balanced subsets from the input DataFrame.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'label' column
            
        Returns:
            List[pd.DataFrame]: List of balanced subsets
        """
        balanced_subsets = []
        remaining_df = df.copy()

        while remaining_df['label'].sum() > 0:
            # Find continuous positive samples
            positive_indices = remaining_df[remaining_df['label']].index
            first_positive_idx = positive_indices[0]
            
            # Find the end of continuous positive sequence
            last_positive_idx = first_positive_idx
            while (last_positive_idx < remaining_df.index.max() and 
                   remaining_df['label'][last_positive_idx + 1]):
                last_positive_idx += 1
                
            # Calculate sequence length
            sequence_length = last_positive_idx - first_positive_idx + 1
            
            # Create balanced subset
            subset_start = first_positive_idx - int(self.negative_positive_ratio * sequence_length)
            subset = remaining_df.loc[subset_start:last_positive_idx].copy()
            balanced_subsets.append(subset)
            
            # Remove processed samples
            remaining_df = remaining_df.drop(subset.index)
            
        return balanced_subsets
    
    def balance_dataset(self, x_train, x_test, y_train, y_test):
        """Balance training and test datasets while preserving temporal characteristics.
        
        Args:
            x_train (pd.DataFrame): Training features
            x_test (pd.DataFrame): Test features
            y_train (pd.Series): Training labels
            y_test (pd.Series): Test labels
            
        Returns:
            Tuple containing:
                - Balanced training features
                - Balanced test features
                - Balanced training labels
                - Balanced test labels
        """
            
        # Combine features and labels
        train_data = x_train.copy()
        test_data = x_test.copy()
        train_data['label'] = y_train
        test_data['label'] = y_test
        
        # Create balanced subsets
        train_subsets = self.create_balanced_subsets(train_data)
        test_subsets = self.create_balanced_subsets(test_data)
        
        # Combine subsets
        balanced_train = pd.concat(train_subsets)
        balanced_test = pd.concat(test_subsets)
        
        # Split features and labels
        x_train_balanced = balanced_train.drop('label', axis=1)
        y_train_balanced = balanced_train['label']
        x_test_balanced = balanced_test.drop('label', axis=1)
        y_test_balanced = balanced_test['label']
        
        # Print balancing results
        self._print_combined_balance_stats(
            train_original=y_train,
            train_balanced=y_train_balanced,
            test_original=y_test,
            test_balanced=y_test_balanced
        )
        
        return x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced

    @staticmethod
    def _print_combined_balance_stats(train_original, train_balanced,
                                      test_original, test_balanced):
        """Print combined statistics about the balancing operation for both datasets.

        Args:
            train_original (pd.Series): Original training labels
            train_balanced (pd.Series): Balanced training labels
            test_original (pd.Series): Original test labels
            test_balanced (pd.Series): Balanced test labels
        """
        headers = ["Metric", "Training Original", "Training Balanced",
                   "Test Original", "Test Balanced"]

        # Calculate statistics
        train_orig_stats = {
            'total': len(train_original),
            'positive': train_original.sum(),
            'negative': (~train_original).sum(),
            'ratio': (~train_original).sum() / train_original.sum()
        }

        train_bal_stats = {
            'total': len(train_balanced),
            'positive': train_balanced.sum(),
            'negative': (~train_balanced).sum(),
            'ratio': (~train_balanced).sum() / train_balanced.sum()
        }

        test_orig_stats = {
            'total': len(test_original),
            'positive': test_original.sum(),
            'negative': (~test_original).sum(),
            'ratio': (~test_original).sum() / test_original.sum()
        }

        test_bal_stats = {
            'total': len(test_balanced),
            'positive': test_balanced.sum(),
            'negative': (~test_balanced).sum(),
            'ratio': (~test_balanced).sum() / test_balanced.sum()
        }

        # Create table data
        table_data = [
            ["Total Samples",
             train_orig_stats['total'],
             train_bal_stats['total'],
             test_orig_stats['total'],
             test_bal_stats['total']],
            ["Positive Samples",
             train_orig_stats['positive'],
             train_bal_stats['positive'],
             test_orig_stats['positive'],
             test_bal_stats['positive']],
            ["Negative Samples",
             train_orig_stats['negative'],
             train_bal_stats['negative'],
             test_orig_stats['negative'],
             test_bal_stats['negative']],
            ["Neg/Pos Ratio",
             f"{train_orig_stats['ratio']:.2f}",
             f"{train_bal_stats['ratio']:.2f}",
             f"{test_orig_stats['ratio']:.2f}",
             f"{test_bal_stats['ratio']:.2f}"]
        ]

        print("\nDataset Balance Statistics:")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))