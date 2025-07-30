import os
import numpy as np
import argparse

from models.prediction.lstm_predictor import LSTMPredictor
from models.feature_engineering.feature_selector import FeatureSelector
from utils.data_loader import DataLoader
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from utils.model_evaluation import evaluate_model
from utils.progress_display import create_progress_display
from config import MODEL_DIR, FAULT_DESCRIPTIONS, get_feature_path


def lstm_predict(production_line_code, fault_code, temporal, use_rf=True, rf_threshold=0.9,
                 rf_balance=True, parameter_optimization=False, sequence_length=10, 
                 fast_mode=False, random_state=42):
    """Train and evaluate LSTM model with optional RF feature selection."""
    
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
        # Check if pre-selected features exist
        features_path = get_feature_path(fault_code)
        model_exist = os.path.exists(features_path)

        # Use FeatureSelector to select features
        feature_selector = FeatureSelector(random_state=random_state)
        x_train, x_test = feature_selector.select_important_features(
            train_data, test_data, fault_code, rf_threshold, model_exist=model_exist
        )

    with progress.model_training_status() as status:
        status.update("Initializing LSTM predictor...")
        predictor = LSTMPredictor(random_state=random_state, sequence_length=sequence_length)
        
        # Override parameters for fast mode
        if fast_mode:
            predictor.initial_param_space = {
                'lstm_units': [32],
                'dropout_rate': [0.2],
                'learning_rate': [0.001],
                'batch_size': [32],
                'epochs': [10],
                'dense_units': [16],
                'lstm_layers': [1]
            }

        status.update("Training LSTM model...")
        model = predictor.train(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            parameter_optimization=parameter_optimization and not fast_mode
        )

        status.update("Evaluating model performance...")
        # Create a wrapper class for LSTM model to work with evaluate_model
        class LSTMModelWrapper:
            def __init__(self, predictor, x_test, y_test):
                self.predictor = predictor
                self.x_test = x_test
                self.y_test = y_test
            
            def predict(self, x_input):
                # Get predictions from LSTM predictor
                y_pred, _ = self.predictor.predict(x_input)
                
                # Ensure predictions and test labels have the same length
                min_length = min(len(y_pred), len(self.y_test))
                y_pred_adjusted = y_pred[:min_length]
                
                return y_pred_adjusted
        
        # Create wrapper and evaluate
        model_wrapper = LSTMModelWrapper(predictor, x_test, y_test)
        
        # Adjust y_test to match prediction length
        y_pred_temp, _ = predictor.predict(x_test)
        min_length = min(len(y_pred_temp), len(y_test))
        y_test_adjusted = y_test.iloc[:min_length] if hasattr(y_test, 'iloc') else y_test[:min_length]
        
        evaluate_model(model_wrapper, x_test, y_test_adjusted)

        status.update("Saving trained model...")
        model_path = os.path.join(MODEL_DIR, f'lstm_production_line_{production_line_code}_fault_{fault_code}.pkl')
        predictor.save_model(model_path)
        
        status.complete("LSTM model training completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use LSTM model to predict production line fault')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')
    parser.add_argument('--fault_code', type=int, required=True, help='Fault code')
    
    parser.add_argument('--no-temporal', action='store_false', dest='temporal', help='Do not use Temporal data')
    
    parser.add_argument('--no-rf', action='store_false', dest='use_rf',
                        help='Do not use Random Forest for feature selection')
    
    parser.add_argument('--rf-threshold', type=float, default=0.9, help='Threshold for feature selection')
    parser.add_argument('--no-balance', action='store_false', dest='rf_balance', help='Do not balance dataset')
    parser.add_argument('--parameter-opt', action='store_true', dest='parameter_optimization',
                        help='Use parameter optimization')
    parser.add_argument('--sequence-length', type=int, default=10, help='Sequence length for LSTM input')
    parser.add_argument('--fast-mode', action='store_true', help='Use fast training mode with minimal parameters')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')

    parser.set_defaults(temporal=True, use_rf=True, rf_balance=True, parameter_optimization=False, fast_mode=False)

    # Parse command line arguments
    args = parser.parse_args()

    lstm_predict(
        args.production_line,
        args.fault_code,
        args.temporal,
        args.use_rf,
        args.rf_threshold,
        args.rf_balance,
        args.parameter_optimization,
        args.sequence_length,
        args.fast_mode,
        args.random_state
    )