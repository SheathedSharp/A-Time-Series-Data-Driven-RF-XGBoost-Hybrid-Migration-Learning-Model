import os
import numpy as np
import pandas as pd

from models.prediction.xgboost_predictor import XGBoostPredictor
from models.feature_engineering.feature_selector import FeatureSelector
from models.sampling.balanced_sampler import ContinuousBalancedSliceSampler
from utils.data_loader import DataLoader
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from config import MODEL_DIR, FAULT_DESCRIPTIONS, get_feature_path
import argparse

def xgboost_predict(production_line_code, fault_code, temporal, use_rf=True, rf_threshold=0.9, use_balance=True, parameter_optimization=False, random_state=42):
    """
    Train and evaluate XGBoost model with optional CBSS balancing and RF feature selection.
    
    Processing pipeline:
    1. Load and split data
    2. Apply CBSS balancing if use_balance=True (affects both RF and XGBoost)
    3. Apply RF feature selection if use_rf=True (on balanced/imbalanced data based on step 2)
    4. Train XGBoost on the final processed data
    
    Args:
        production_line_code (int): Production line identifier
        fault_code (int): Fault code to predict
        temporal (bool): Whether to use temporal features
        use_rf (bool): Whether to use Random Forest for feature selection
        rf_threshold (float): Feature importance threshold for RF selection
        use_balance (bool): Whether to apply CBSS balancing to datasets
        parameter_optimization (bool): Whether to use parameter optimization for XGBoost
        random_state (int): Random seed for reproducibility
    """
    
    # Set global random seeds for reproducibility
    np.random.seed(random_state)

    # Load data
    data_loader = DataLoader()
    data = data_loader.prepare_data(production_line_code, temporal)
    data['label'] = (data[FAULT_DESCRIPTIONS[fault_code]] == fault_code)

    train_data, test_data = split_train_test_datasets(data, fault_code)
    train_data, _ = remove_irrelevant_features(train_data)
    test_data, _ = remove_irrelevant_features(test_data)

    y_train = train_data['label']
    x_train = train_data.drop('label', axis=1)
    y_test = test_data['label']
    x_test = test_data.drop('label', axis=1)

    # Apply CBSS balancing if requested  
    if use_balance:
        sampler = ContinuousBalancedSliceSampler(
            k=4.0, 
            alpha=1.96, 
            beta=10.0,
            min_precursor_length=60,
            max_precursor_length=1800
        )
        x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced = sampler.balance_dataset(
            x_train, x_test, y_train, y_test
        )
        
        x_train_work, x_test_work, y_train_work, y_test_work = x_train_balanced, x_test_balanced, y_train_balanced, y_test_balanced
        train_data_work = x_train_work.copy()
        train_data_work['label'] = y_train_work
        test_data_work = x_test_work.copy()
        test_data_work['label'] = y_test_work
    else:
        # Use original imbalanced data
        x_train_work, x_test_work, y_train_work, y_test_work = x_train, x_test, y_train, y_test
        train_data_work, test_data_work = train_data, test_data
    
    # Apply RF feature selection if requested
    if use_rf:
        # Check if pre-selected features exist
        features_path = get_feature_path(fault_code)
        model_exist = os.path.exists(features_path)

        # Use FeatureSelector on the working dataset
        feature_selector = FeatureSelector(random_state=random_state)
        x_train_work, x_test_work = feature_selector.select_important_features(
            train_data_work, test_data_work, fault_code, rf_threshold, model_exist=model_exist
        )

    # Final datasets for XGBoost training
    x_train_final, x_test_final, y_train_final, y_test_final = x_train_work, x_test_work, y_train_work, y_test_work

    # Initialize XGBoost predictor with integrated progress display
    predictor = XGBoostPredictor(random_state=random_state, show_progress=True)

    # Train the model (progress is handled internally)
    model = predictor.train(
        x_train=x_train_final,
        y_train=y_train_final,
        x_test=x_test_final,
        y_test=y_test_final,
        parameter_optimization=parameter_optimization
    )

    # Evaluate and save the model (progress is handled internally)
    model_path = os.path.join(MODEL_DIR, f'xgboost_production_line_{production_line_code}_fault_{fault_code}.pkl')
    predictor.evaluate_and_save(x_test_final, y_test_final, model_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Use XGBoost model to predict production line fault')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')
    parser.add_argument('--fault_code', type=int, required=True, help='Fault code')

    parser.add_argument('--no-temporal', action='store_false', dest='temporal', help='Do not use Temporal data')

    parser.add_argument('--no-rf', action='store_false', dest='use_rf', help='Do not use Random Forest for feature selection')
    parser.add_argument('--rf-threshold', type=float, default=0.9, help='Threshold for feature selection')
    parser.add_argument('--no-balance', action='store_false', dest='use_balance', help='Do not balance dataset using CBSS')
    parser.add_argument('--parameter-opt', action='store_true', dest='parameter_optimization', help='Use parameter optimization')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')

    parser.set_defaults(temporal=True, use_rf=True, use_balance=True, parameter_optimization=False)

    # Parse command line arguments
    args = parser.parse_args()
    
    xgboost_predict(
        args.production_line,
        args.fault_code,
        args.temporal,
        args.use_rf,
        args.rf_threshold,
        args.use_balance,
        args.parameter_optimization,
        args.random_state
    )