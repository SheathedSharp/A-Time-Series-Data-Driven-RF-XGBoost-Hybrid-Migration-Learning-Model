from old_files import FeatureSelector
from config import FAULT_DESCRIPTIONS
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from old_files import BalancedSampler
from utils.data_loader import DataLoader
import pandas as pd


if __name__ == '__main__':
    # load data
    data_loader = DataLoader()
    data = data_loader.prepare_data(production_line_code=1, temporal=True)
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

    train_data = pd.concat([X_train_balanced, y_train_balanced], axis=1)
    test_data = pd.concat([X_test_balanced, y_test_balanced], axis=1)

    print(f"Original training data shape: {train_data.shape}")
    print(f"Original testing data shape: {test_data.shape}")

    feature_selector = FeatureSelector()
    X_train_selected, X_test_selected = feature_selector.select_important_features(
        train_data=train_data,
        test_data=test_data,
        fault_code=1001,
        model_exist=False
    )

    print("After feature selecting train data shape:", X_train_selected.shape)
    print("After feature selecting test data shape:", X_test_selected.shape)


