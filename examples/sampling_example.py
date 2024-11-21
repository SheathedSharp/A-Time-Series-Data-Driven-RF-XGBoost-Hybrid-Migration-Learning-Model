from old_files import BalancedSampler
import pandas as pd
import os
from config import TEMPORAL_DATA_DIR, FAULT_DESCRIPTIONS
from utils.data_process import split_train_test_datasets, remove_irrelevant_features

if __name__ == "__main__":
    # load data
    data_url = os.path.join(TEMPORAL_DATA_DIR, 'production_line_1.csv')
    data = pd.read_csv(data_url, low_memory=False)
    train_data, test_data = split_train_test_datasets(data, 1001)

    y_train = train_data[FAULT_DESCRIPTIONS[1001]] == 1001
    x_train, _ = remove_irrelevant_features(train_data)

    y_test = test_data[FAULT_DESCRIPTIONS[1001]] == 1001
    x_test, _ = remove_irrelevant_features(test_data)

    # Initialize sampler with desired negative/positive ratio
    sampler = BalancedSampler(negative_positive_ratio=10.0)

    # Balance datasets
    X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = \
        sampler.balance_dataset(x_train, x_test, y_train, y_test)

    print(f"Original training data x shape: {x_train.shape}")
    print(f"Balanced training data x shape: {X_train_balanced.shape}")
    print(f"Original test data x shape: {x_test.shape}")
    print(f"Balanced test data x shape: {X_test_balanced.shape}")