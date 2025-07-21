import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tabulate import tabulate
from utils.progress_display import create_progress_display


class ContinuousBalancedSliceSampler:
    """Continuous Balanced Slice Sampling (CBSS) for temporal fault data.
    
    This sampler implements a temporal-aware resampling technique based on 
    fault precursor theory (Goto et al., 2009) and industrial maintenance research.
    It calculates adaptive precursor windows using fault duration statistics.
    
    The precursor window length is calculated as:
    T_precursor,i = k × μ_i + α × σ_i × √(N_i/(N_i + β))
    
    where μ_i and σ_i are mean and std of fault duration for fault type i,
    N_i is the number of fault occurrences, and k, α, β are empirically
    determined coefficients.
    """
    
    def __init__(self, 
                 k: float = 4.0, 
                 alpha: float = 1.96, 
                 beta: float = 10.0,
                 min_precursor_length: int = 60,
                 max_precursor_length: int = 1800):
        """Initialize the CBSS sampler.
        
        Args:
            k (float): Main coefficient for mean duration scaling (default: 4.0)
                      Based on industrial fault analysis suggesting 3-6 times mean duration
            alpha (float): Variance adjustment coefficient (default: 1.96)
                          Corresponds to 95% confidence interval
            beta (float): Sample size correction factor (default: 10.0)
                         Empirically determined to adjust for small sample bias
            min_precursor_length (int): Minimum precursor window length in seconds (default: 60)
            max_precursor_length (int): Maximum precursor window length in seconds (default: 1800)
        """
        self.k = k
        self.alpha = alpha
        self.beta = beta
        self.min_precursor_length = min_precursor_length
        self.max_precursor_length = max_precursor_length
        self.fault_statistics = {}
        self.progress = create_progress_display()
    
    def analyze_fault_statistics(self, df: pd.DataFrame, fault_type_col: str = None) -> Dict:
        """Analyze fault duration statistics for adaptive precursor window calculation.
        
        Args:
            df (pd.DataFrame): DataFrame with fault data
            fault_type_col (str): Column name containing fault type information
            
        Returns:
            Dict: Fault statistics including mean, std, count for each fault type
        """
        fault_stats = {}
        
        if fault_type_col and fault_type_col in df.columns:
            # Analyze by fault type
            fault_types = df[df['label'] == 1][fault_type_col].unique()
            for fault_type in fault_types:
                fault_data = df[df[fault_type_col] == fault_type]
                stats = self._calculate_fault_duration_stats(fault_data)
                fault_stats[fault_type] = stats
        else:
            # Global analysis for all faults
            fault_data = df[df['label'] == 1]
            stats = self._calculate_fault_duration_stats(fault_data)
            fault_stats['global'] = stats
            
        self.fault_statistics = fault_stats
        return fault_stats
    
    def _calculate_fault_duration_stats(self, fault_df: pd.DataFrame) -> Dict:
        """Calculate fault duration statistics.
        
        Args:
            fault_df (pd.DataFrame): DataFrame containing fault events
            
        Returns:
            Dict: Statistics including mean, std, count, and precursor window length
        """
        if len(fault_df) == 0:
            return {
                'mean_duration': 0,
                'std_duration': 0,
                'count': 0,
                'precursor_length': self.min_precursor_length
            }
        
        # Find continuous fault sequences
        fault_sequences = self._find_continuous_sequences(fault_df)
        durations = [seq['duration'] for seq in fault_sequences]
        
        if len(durations) == 0:
            return {
                'mean_duration': 0,
                'std_duration': 0,
                'count': 0,
                'precursor_length': self.min_precursor_length
            }
        
        mean_duration = np.mean(durations)
        std_duration = np.std(durations, ddof=1) if len(durations) > 1 else 0
        count = len(durations)
        
        # Calculate precursor window length using our formula
        precursor_length = self._calculate_precursor_window(
            mean_duration, std_duration, count
        )
        
        return {
            'mean_duration': mean_duration,
            'std_duration': std_duration,
            'count': count,
            'precursor_length': precursor_length,
            'sequences': fault_sequences
        }
    
    def _calculate_precursor_window(self, mean_duration: float, 
                                  std_duration: float, count: int) -> int:
        """Calculate precursor window length using statistical formula.
        
        T_precursor = k × μ + α × σ × √(N/(N + β))
        
        Args:
            mean_duration (float): Mean fault duration
            std_duration (float): Standard deviation of fault duration
            count (int): Number of fault occurrences
            
        Returns:
            int: Precursor window length in time steps
        """
        if count == 0:
            return self.min_precursor_length
        
        # Apply the statistical formula
        sample_correction = np.sqrt(count / (count + self.beta))
        precursor_length = (self.k * mean_duration + 
                          self.alpha * std_duration * sample_correction)
        
        # Apply bounds
        precursor_length = max(self.min_precursor_length, 
                             min(self.max_precursor_length, int(precursor_length)))
        
        return precursor_length
    
    def _find_continuous_sequences(self, df: pd.DataFrame) -> List[Dict]:
        """Find continuous fault sequences in the data.
        
        Args:
            df (pd.DataFrame): DataFrame with fault data
            
        Returns:
            List[Dict]: List of fault sequences with start, end, and duration
        """
        sequences = []
        fault_indices = df[df['label'] == 1].index.tolist()
        
        if not fault_indices:
            return sequences
        
        # Group continuous indices
        current_start = fault_indices[0]
        current_end = fault_indices[0]
        
        for i in range(1, len(fault_indices)):
            if fault_indices[i] == current_end + 1:
                current_end = fault_indices[i]
            else:
                # End of continuous sequence
                sequences.append({
                    'start': current_start,
                    'end': current_end,
                    'duration': current_end - current_start + 1
                })
                current_start = fault_indices[i]
                current_end = fault_indices[i]
        
        # Add the last sequence
        sequences.append({
            'start': current_start,
            'end': current_end,
            'duration': current_end - current_start + 1
        })
        
        return sequences
    
    def create_balanced_subsets(self, df: pd.DataFrame, 
                              fault_type_col: str = None) -> List[pd.DataFrame]:
        """Create balanced subsets using adaptive precursor windows.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'label' column
            fault_type_col (str): Optional column for fault type analysis
            
        Returns:
            List[pd.DataFrame]: List of balanced subsets
        """
        # Analyze fault statistics
        self.analyze_fault_statistics(df, fault_type_col)
        
        balanced_subsets = []
        remaining_df = df.copy()
        
        while remaining_df['label'].sum() > 0:
            # Find continuous positive samples
            positive_indices = remaining_df[remaining_df['label'] == 1].index
            if len(positive_indices) == 0:
                break
                
            first_positive_idx = positive_indices[0]
            
            # Find the end of continuous positive sequence
            last_positive_idx = first_positive_idx
            while (last_positive_idx < remaining_df.index.max() and 
                   last_positive_idx + 1 in remaining_df.index and
                   remaining_df.loc[last_positive_idx + 1, 'label'] == 1):
                last_positive_idx += 1
                
            # Calculate sequence length
            sequence_length = last_positive_idx - first_positive_idx + 1
            
            # Determine precursor window length
            if fault_type_col and fault_type_col in df.columns:
                fault_type = remaining_df.loc[first_positive_idx, fault_type_col]
                if fault_type in self.fault_statistics:
                    precursor_length = self.fault_statistics[fault_type]['precursor_length']
                else:
                    precursor_length = self.fault_statistics.get('global', {}).get('precursor_length', self.min_precursor_length)
            else:
                precursor_length = self.fault_statistics.get('global', {}).get('precursor_length', self.min_precursor_length)
            
            # Create balanced subset with adaptive precursor window
            subset_start = max(remaining_df.index.min(), 
                             first_positive_idx - precursor_length)
            subset_end = last_positive_idx
            
            # Extract subset
            subset_indices = remaining_df.index[
                (remaining_df.index >= subset_start) & 
                (remaining_df.index <= subset_end)
            ]
            subset = remaining_df.loc[subset_indices].copy()
            
            if len(subset) > 0:
                balanced_subsets.append(subset)
            
            # Remove processed positive samples
            processed_indices = remaining_df.index[
                (remaining_df.index >= first_positive_idx) & 
                (remaining_df.index <= last_positive_idx)
            ]
            remaining_df = remaining_df.drop(processed_indices)
            
        return balanced_subsets
    
    def balance_dataset(self, x_train: pd.DataFrame, x_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series,
                       fault_type_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Balance training and test datasets using CBSS.
        
        Args:
            x_train (pd.DataFrame): Training features
            x_test (pd.DataFrame): Test features
            y_train (pd.Series): Training labels
            y_test (pd.Series): Test labels
            fault_type_col (str): Optional fault type column name
            
        Returns:
            Tuple containing balanced training and test data
        """
        with self.progress.sampling_status() as status:
            status.update("Analyzing fault patterns...")
            # Combine features and labels
            train_data = x_train.copy()
            test_data = x_test.copy()
            train_data['label'] = y_train
            test_data['label'] = y_test
            
            status.update("Creating balanced subsets...")
            # Create balanced subsets
            train_subsets = self.create_balanced_subsets(train_data, fault_type_col)
            test_subsets = self.create_balanced_subsets(test_data, fault_type_col)
            
            # Handle empty subsets
            if not train_subsets:
                self.progress.display_warning("No balanced training subsets created")
                return x_train, x_test, y_train, y_test
            
            if not test_subsets:
                self.progress.display_warning("No balanced test subsets created")
                return x_train, x_test, y_train, y_test
            
            status.update("Combining balanced subsets...")
            # Combine subsets and sort by index to maintain temporal order
            balanced_train = pd.concat(train_subsets).sort_index()
            balanced_test = pd.concat(test_subsets).sort_index()
            
            # Split features and labels
            x_train_balanced = balanced_train.drop('label', axis=1)
            y_train_balanced = balanced_train['label']
            x_test_balanced = balanced_test.drop('label', axis=1)
            y_test_balanced = balanced_test['label']
            
            status.complete("Dataset balancing completed")
            
            # Display statistics
            self._display_sampling_statistics()
            self._display_combined_balance_stats(
                train_original=y_train,
                train_balanced=y_train_balanced,
                test_original=y_test,
                test_balanced=y_test_balanced
            )
            
        return x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced
    
    def _display_sampling_statistics(self):
        """Display fault statistics and precursor window calculations."""
        if not self.fault_statistics:
            return
        
        # Create statistics summary
        stats_data = {
            "Parameters": f"k={self.k}, α={self.alpha}, β={self.beta}",
            "Formula": "T_precursor = k×μ + α×σ×√(N/(N+β))"
        }
        
        for fault_type, stats in self.fault_statistics.items():
            stats_data[f"{fault_type} - Count"] = stats['count']
            stats_data[f"{fault_type} - Mean Duration"] = f"{stats['mean_duration']:.2f}s"
            stats_data[f"{fault_type} - Std Duration"] = f"{stats['std_duration']:.2f}s"
            stats_data[f"{fault_type} - Precursor Window"] = f"{stats['precursor_length']}s"
        
        self.progress.display_results_table("CBSS Sampling Statistics", stats_data)

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