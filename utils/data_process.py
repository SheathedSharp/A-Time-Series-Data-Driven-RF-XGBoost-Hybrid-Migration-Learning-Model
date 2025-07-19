import numpy as np
from sklearn.model_selection import train_test_split
from config import EXCLUDE_COLUMNS, FAULT_DESCRIPTIONS

def remove_irrelevant_features(data):
    """Remove irrelevant features from the dataset

    Args:
        data: includes all the features and labels

    Returns:
        data: the dataset with irrelevant features removed
    """
    feature_columns = [col for col in data.columns if col not in EXCLUDE_COLUMNS]
    return data[feature_columns], feature_columns

def get_irrelevant_features_data(data):
    """Get irrelevant features data from the dataset

    Args:
        data: includes all the features and labels

    Returns:
        data: the dataset with irrelevant data
    """
    irrelevant_feature_columns = [col for col in data.columns if col in EXCLUDE_COLUMNS]
    return data[irrelevant_feature_columns]


def merge_irrelevant_features(processed_data, irrelevant_data):
    """Merge irrelevant features back to processed data.

    Args:
        processed_data: The processed data without irrelevant features
        irrelevant_data: The original data containing irrelevant features

    Returns:
        pd.DataFrame: Merged dataframe containing both processed and irrelevant features
    """
    final_data = processed_data.copy()
    for col in EXCLUDE_COLUMNS:
        final_data[col] = irrelevant_data[col].values
    return final_data

def split_train_test_datasets(data, fault_code):
    all_dates = data['Date'].unique()

    train_dates, test_dates = train_test_split(
        all_dates, test_size=0.2, random_state=42)

    train_data = data[data['Date'].isin(train_dates)]
    test_data = data[data['Date'].isin(test_dates)]

    fault_description = FAULT_DESCRIPTIONS[fault_code]

    # Check if the fault code is in the train and test datasets
    if np.sum(train_data[f'{fault_description}'] == fault_code) == 0 or np.sum(
            test_data[f'{fault_description}'] == fault_code) == 0:
        train_data, test_data = split_train_test_datasets(data, fault_code)

    return train_data, test_data