import os
import numpy as np
import pandas as pd

from models.prediction.mlp_predictor import MLPPredictor
from utils.data_loader import DataLoader
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from config import MODEL_DIR, FAULT_DESCRIPTIONS
import argparse

def mlp_predict(production_line_code, fault_code, random_state=42):
    """
    Train and evaluate MLP BASELINE model using raw data only.
    
    Baseline processing pipeline:
    1. Load raw data (NO temporal features)
    2. Split train/test (NO CBSS balancing)
    3. Train MLP directly (NO RF feature selection)
    
    This serves as a fair baseline comparison for the hybrid RF-XGBoost method.
    
    Args:
        production_line_code (int): Production line identifier
        fault_code (int): Fault code to predict
        random_state (int): Random seed for reproducibility
    """
    
    # Set global random seeds for reproducibility
    np.random.seed(random_state)

    # Load RAW data
    data_loader = DataLoader()
    data = data_loader.prepare_data(production_line_code, temporal=False) 
    data['label'] = (data[FAULT_DESCRIPTIONS[fault_code]] == fault_code)

    train_data, test_data = split_train_test_datasets(data, fault_code)
    train_data, _ = remove_irrelevant_features(train_data)
    test_data, _ = remove_irrelevant_features(test_data)

    y_train = train_data['label']
    x_train = train_data.drop('label', axis=1)
    y_test = test_data['label']
    x_test = test_data.drop('label', axis=1)

    # Initialize MLP baseline predictor with integrated progress display
    predictor = MLPPredictor(random_state=random_state, show_progress=True)

    # Train the baseline model
    model = predictor.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    # Evaluate and save the baseline model
    model_path = os.path.join(MODEL_DIR, f'mlp_production_line_{production_line_code}_fault_{fault_code}.pkl')
    predictor.evaluate_and_save(x_test, y_test, model_path)
    
    print(f"Baseline MLP model saved to: {model_path}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train MLP baseline model for comparison with hybrid RF-XGBoost method')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')
    parser.add_argument('--fault_code', type=int, required=True, help='Fault code')

    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')

    # Parse command line arguments
    args = parser.parse_args()
    
    mlp_predict(
        args.production_line,
        args.fault_code,
        args.random_state
    )