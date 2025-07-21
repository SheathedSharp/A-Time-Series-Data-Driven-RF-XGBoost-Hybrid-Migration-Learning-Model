from models.feature_engineering.feature_selector import FeatureSelector
from config import FAULT_DESCRIPTIONS
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from models.sampling.balanced_sampler import ContinuousBalancedSliceSampler
from utils.data_loader import DataLoader
from utils.progress_display import create_progress_display
import pandas as pd


if __name__ == '__main__':
    progress = create_progress_display()
    
    # load data
    with progress.data_loading_status() as status:
        status.update("Loading production line data...")
        data_loader = DataLoader()
        data = data_loader.prepare_data(production_line_code=1, temporal=True)
        train_data, test_data = split_train_test_datasets(data, 1001)

        y_train = train_data[FAULT_DESCRIPTIONS[1001]] == 1001
        x_train, _ = remove_irrelevant_features(train_data)

        y_test = test_data[FAULT_DESCRIPTIONS[1001]] == 1001
        x_test, _ = remove_irrelevant_features(test_data)

        status.update("Balancing dataset...")
        sampler = ContinuousBalancedSliceSampler(
            k=4.0, 
            alpha=1.96, 
            beta=10.0,
            min_precursor_length=60,
            max_precursor_length=1800
        )

        # Balance datasets
        X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = \
            sampler.balance_dataset(x_train, x_test, y_train, y_test)

        train_data = pd.concat([X_train_balanced, y_train_balanced], axis=1)
        test_data = pd.concat([X_test_balanced, y_test_balanced], axis=1)

        status.complete("Data loading and balancing completed")

    # Display data shapes
    summary_data = {
        "Original Training Shape": str(train_data.shape),
        "Original Testing Shape": str(test_data.shape)
    }
    progress.display_results_table("Dataset Information", summary_data)

    with progress.feature_selection_status() as status:
        status.update("Initializing feature selector...")
        feature_selector = FeatureSelector()
        
        status.update("Performing feature selection...")
        X_train_selected, X_test_selected = feature_selector.select_important_features(
            train_data=train_data,
            test_data=test_data,
            fault_code=1001,
        )
        
        status.complete("Feature selection completed")

    # Display results
    results_data = {
        "After Feature Selection - Train Shape": str(X_train_selected.shape),
        "After Feature Selection - Test Shape": str(X_test_selected.shape),
        "Features Selected": str(X_train_selected.shape[1])
    }
    progress.display_results_table("Feature Selection Results", results_data)


