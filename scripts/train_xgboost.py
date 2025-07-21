import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from models.prediction.xgboost_predictor import XGBoostPredictor
from utils.data_loader import DataLoader
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from utils.model_evaluation import evaluate_model
from utils.progress_display import create_progress_display
from config import MODEL_DIR, FAULT_DESCRIPTIONS, get_feature_path
import argparse

def get_selected_features(production_line_code, fault_code, rf_threshold, rf_balance, random_state=42, status=None):
    """Get or generate selected features for the specified fault code."""
    features_path = get_feature_path(fault_code)
    progress = create_progress_display()
    
    if os.path.exists(features_path):
        if status:
            status.update("Loading existing feature selection...")

        selected_features = pd.read_csv(features_path)['feature_name'].tolist()
    else:
        if status:
            status.update("No existing feature selection found. Running feature selection...")

        try:
            cmd = [
                'python', 'scripts/select_features.py',
                '--production_line', str(production_line_code),
                '--fault_code', str(fault_code),
                '--threshold', str(rf_threshold),
                '--random-state', str(random_state)  # Pass random seed
            ]

            if not rf_balance:  # Only add --no-balance parameter if balance is not needed
                cmd.append('--no-balance')

            if status:
                status.update("Running Random Forest feature selection script...")
            
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            
            # Check if file was generated
            if not os.path.exists(features_path):
                raise FileNotFoundError("Feature selection file was not generated successfully")
                
            selected_features = pd.read_csv(features_path)['feature_name'].tolist()
            
        except subprocess.CalledProcessError as e:
            progress.display_error("Failed to perform feature selection", 
                                f"Exit code: {e.returncode}\nError output: {e.stderr}")
            raise RuntimeError("Failed to perform feature selection")
            
        except Exception as e:
            progress.display_error("Unexpected error during feature selection", str(e))
            raise
            
    return selected_features

def xgboost_predict(production_line_code, fault_code, temporal, use_rf=True, rf_threshold=0.9, rf_balance=True, parameter_optimization=False, random_state=42):
    """Train and evaluate XGBoost model with optional RF feature selection."""
    
    # Set global random seeds for reproducibility
    np.random.seed(random_state)

    # Create progress display
    progress = create_progress_display()

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

    if use_rf:
        with progress.feature_selection_status() as status:
            status.update("Using Random Forest for feature selection...")
            
            selected_features = get_selected_features(
                production_line_code, 
                fault_code, 
                rf_threshold, 
                rf_balance,
                random_state,
                status  
            )
            x_train = x_train[selected_features]
            x_test = x_test[selected_features]
            
            
            status.complete(f"Feature selection completed, using {len(selected_features)} features")

    with progress.model_training_status() as status:
        status.update("Initializing XGBoost predictor...")
        predictor = XGBoostPredictor(random_state=random_state)

        status.update("Training XGBoost model...")
        model = predictor.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            parameter_optimization=parameter_optimization
        )

        status.update("Evaluating model performance...")
        scaler = StandardScaler()
        x_test_scaled = scaler.fit_transform(x_test)
        evaluate_model(model, x_test_scaled, y_test)

        status.update("Saving trained model...")
        model_path = os.path.join(MODEL_DIR, f'xgboost_production_line_{production_line_code}_fault_{fault_code}.pkl')
        predictor.save_model(model_path)
        
        status.complete("XGBoost model training completed successfully")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Use XGBoost model to predict production line fault')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')
    parser.add_argument('--fault_code', type=int, required=True, help='Fault code')

    parser.add_argument('--no-temporal', action='store_false', dest='temporal', help='Do not use Temporal data')

    parser.add_argument('--no-rf', action='store_false', dest='use_rf', help='Do not use Random Forest for feature selection')
    parser.add_argument('--rf-threshold', type=float, default=0.9, help='Threshold for feature selection')
    parser.add_argument('--no-balance', action='store_false', dest='rf_balance', help='Do not balance dataset')
    parser.add_argument('--parameter-opt', action='store_true', dest='parameter_optimization', help='Use parameter optimization')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')

    parser.set_defaults(temporal=True, use_rf=True, rf_balance=True, parameter_optimization=False)

    # Parse command line arguments
    args = parser.parse_args()
    
    xgboost_predict(
        args.production_line,
        args.fault_code,
        args.temporal,
        args.use_rf,
        args.rf_threshold,
        args.rf_balance,
        args.parameter_optimization,
        args.random_state
    )