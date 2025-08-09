import os
import numpy as np
import argparse

from models.prediction.lstm_predictor import LSTMPredictor
from utils.data_loader import DataLoader
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from utils.model_evaluation import evaluate_model
from config import MODEL_DIR, FAULT_DESCRIPTIONS


def lstm_predict(production_line_code, fault_code, sequence_length=10, random_state=42):
    """Train and evaluate LSTM model using raw data only.
    
    This approach uses:
    - Raw data only (no temporal features)
    - No CBSS balanced sampling
    - No RF feature selection
    - Default LSTM parameters (no optimization)
    """
    
    # Set global random seeds for reproducibility
    np.random.seed(random_state)

    # Load raw data (temporal=False for baseline)
    data_loader = DataLoader()
    data = data_loader.prepare_data(production_line_code, temporal=False)
    data['label'] = (data[FAULT_DESCRIPTIONS[fault_code]] == fault_code)

    # Split data without any preprocessing
    train_data, test_data = split_train_test_datasets(data, fault_code)
    train_data, _ = remove_irrelevant_features(train_data)
    test_data, _ = remove_irrelevant_features(test_data)

    y_train = train_data['label']
    x_train = train_data.drop('label', axis=1)
    y_test = test_data['label']
    x_test = test_data.drop('label', axis=1)

    # Initialize LSTM predictor
    predictor = LSTMPredictor(
        random_state=random_state, 
        sequence_length=sequence_length,
        show_progress=True
    )
    
    # Train model
    model = predictor.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

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

    # Save model
    model_path = os.path.join(MODEL_DIR, f'lstm_production_line_{production_line_code}_fault_{fault_code}.pkl')
    predictor.save_model(model_path)
    
    print(f"\nLSTM model saved to: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use LSTM model to predict production line fault')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')
    parser.add_argument('--fault_code', type=int, required=True, help='Fault code')
    parser.add_argument('--sequence-length', type=int, default=10, help='Sequence length for LSTM input')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')

    # Parse command line arguments
    args = parser.parse_args()

    lstm_predict(
        args.production_line,
        args.fault_code,
        args.sequence_length,
        args.random_state
    )