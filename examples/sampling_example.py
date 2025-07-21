"""
Example of using Continuous Balanced Slice Sampling (CBSS)
"""
from models.sampling.balanced_sampler import ContinuousBalancedSliceSampler
import pandas as pd
import os
from config import TEMPORAL_DATA_DIR, FAULT_DESCRIPTIONS
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from utils.progress_display import create_progress_display

if __name__ == "__main__":
    progress = create_progress_display()
    
    with progress.data_loading_status() as status:
        # Load data
        status.update("Loading temporal data...")
        data_url = os.path.join(TEMPORAL_DATA_DIR, 'production_line_1.csv')
        data = pd.read_csv(data_url, low_memory=False)
        train_data, test_data = split_train_test_datasets(data, 1001)

        y_train = train_data[FAULT_DESCRIPTIONS[1001]] == 1001
        x_train, _ = remove_irrelevant_features(train_data)

        y_test = test_data[FAULT_DESCRIPTIONS[1001]] == 1001
        x_test, _ = remove_irrelevant_features(test_data)

        status.update("Initializing CBSS sampler...")
        # Initialize CBSS sampler with adaptive parameters
        # k=4.0: 4 times mean fault duration for precursor window
        # alpha=1.96: 95% confidence interval
        # beta=10.0: small sample correction factor
        sampler = ContinuousBalancedSliceSampler(
            k=4.0, 
            alpha=1.96, 
            beta=10.0,
            min_precursor_length=60,
            max_precursor_length=1800
        )

        status.update("Balancing datasets using adaptive precursor windows...")
        # Balance datasets using adaptive precursor windows
        X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = \
            sampler.balance_dataset(x_train, x_test, y_train, y_test)

    # Display results
    summary_data = {
        "Original Train Shape": str(x_train.shape),
        "Original Test Shape": str(x_test.shape),
        "Balanced Train Shape": str(X_train_balanced.shape),
        "Balanced Test Shape": str(X_test_balanced.shape),
        "Original Positive Samples": str(y_train.sum()),
        "Balanced Positive Samples": str(y_train_balanced.sum())
    }
    progress.display_results_table("CBSS Sampling Results", summary_data)