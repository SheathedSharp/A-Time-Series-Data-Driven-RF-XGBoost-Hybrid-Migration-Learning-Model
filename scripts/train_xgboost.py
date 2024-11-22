import os
import subprocess
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.prediction.xgboost_predictor import XGBoostPredictor
from utils.data_loader import DataLoader
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from utils.model_evaluation import evaluate_model
from config import MODEL_DIR, FAULT_DESCRIPTIONS, get_feature_path
import argparse

def get_selected_features(production_line_code, fault_code, rf_ratio, rf_threshold, rf_balance):
    """Get or generate selected features for the specified fault code."""
    features_path = get_feature_path(fault_code)
    
    if os.path.exists(features_path):
        print("\nLoading existing feature selection...")
        selected_features = pd.read_csv(features_path)['feature_name'].tolist()
    else:
        print("\nNo existing feature selection found. Running feature selection...")
        try:
            cmd = [
                'python', 'scripts/select_features.py',
                '--production_line', str(production_line_code),
                '--fault_code', str(fault_code),
                '--ratio', str(rf_ratio),
                '--threshold', str(rf_threshold)
            ]

            if not rf_balance:  # 如果不需要平衡，才添加 --no-balance 参数
                cmd.append('--no-balance')

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print("Feature selection completed successfully!")
            
            # 检查文件是否生成
            if not os.path.exists(features_path):
                raise FileNotFoundError("Feature selection file was not generated successfully")
                
            selected_features = pd.read_csv(features_path)['feature_name'].tolist()
            
        except subprocess.CalledProcessError as e:
            print("\nError during feature selection:")
            print(f"Exit code: {e.returncode}")
            print(f"Error output: {e.stderr}")
            raise RuntimeError("Failed to perform feature selection")
            
        except Exception as e:
            print(f"\nUnexpected error during feature selection: {str(e)}")
            raise
            
    return selected_features

def xgboost_predict(production_line_code, fault_code, temporal, use_rf=True, rf_ratio=10, rf_threshold=0.9, rf_balance=True, parameter_optimization=False):
    """Train and evaluate XGBoost model with optional RF feature selection."""

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
        print("\nUsing Random Forest for feature selection...")
        selected_features = get_selected_features(
            production_line_code, 
            fault_code, 
            rf_ratio, 
            rf_threshold, 
            rf_balance,
        )
        x_train = x_train[selected_features]
        x_test = x_test[selected_features]


    predictor = XGBoostPredictor()

    model = predictor.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        parameter_optimization=parameter_optimization
    )

    scaler = StandardScaler()
    x_test_scaled = scaler.fit_transform(x_test)
    evaluate_model(model,x_test_scaled,y_test)

    model_path = os.path.join(MODEL_DIR, f'xgboost_production_line_{production_line_code}_fault_{fault_code}.pkl')
    predictor.save_model(model_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Use XGBoost model to predict production line fault')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')
    parser.add_argument('--fault_code', type=int, required=True, help='Fault code')

    parser.add_argument('--no-temporal', action='store_false', dest='temporal', help='Do not use Temporal data')

    parser.add_argument('--no-rf', action='store_false', dest='use_rf', help='Do not use Random Forest for feature selection')
    parser.add_argument('--rf-ratio', type=float, default=10.0, help='Negative/positive ratio for balanced sampling')
    parser.add_argument('--rf-threshold', type=float, default=0.9, help='Threshold for feature selection')
    parser.add_argument('--no-balance', action='store_false', dest='rf_balance', help='Do not balance dataset')
    parser.add_argument('--parameter-opt', action='store_true', dest='parameter_optimization', help='Use parameter optimization')

    parser.set_defaults(temporal=True, ues_rf=True, rf_balance=True, parameter_optimization=False)

    # Parse command line arguments
    args = parser.parse_args()
    
    xgboost_predict(
        args.production_line,
        args.fault_code,
        args.temporal,
        args.use_rf,
        args.rf_ratio,
        args.rf_threshold,
        args.rf_balance,
        args.parameter_optimization
    )