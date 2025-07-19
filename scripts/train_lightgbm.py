import os
from sklearn.preprocessing import StandardScaler

from models.feature_engineering.feature_selector import FeatureSelector
from models.prediction.lightgbm_predictor import LightGBMPredictor
from utils.data_loader import DataLoader
from utils.data_process import split_train_test_datasets, remove_irrelevant_features
from utils.model_evaluation import evaluate_model
from config import MODEL_DIR, FAULT_DESCRIPTIONS
import argparse


def lightgbm_predict(production_line_code, fault_code, temporal, use_rf=True, rf_threshold=0.9,
                     rf_balance=True, parameter_optimization=False):
    """Train and evaluate LightGBM model with optional RF feature selection."""

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
        feature_selector = FeatureSelector()
        x_train, x_test = feature_selector.select_features(
            x_train,
            x_test,
            production_line_code,
            fault_code,
            rf_threshold,
            rf_balance
        )

    predictor = LightGBMPredictor()

    lightgbm_model = predictor.train(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        parameter_optimization=parameter_optimization
    )

    scaler = StandardScaler()
    x_test_scaled = scaler.fit_transform(x_test)
    evaluate_model(lightgbm_model, x_test_scaled, y_test)

    model_path = os.path.join(MODEL_DIR, f'lightgbm_production_line_{production_line_code}_fault_{fault_code}.pkl')
    predictor.save_model(model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use LightGBM model to predict production line fault')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')
    parser.add_argument('--fault_code', type=int, required=True, help='Fault code')

    parser.add_argument('--no-temporal', action='store_false', dest='temporal', help='Do not use Temporal data')

    parser.add_argument('--no-rf', action='store_false', dest='use_rf',
                        help='Do not use Random Forest for feature selection')
    
    parser.add_argument('--rf-threshold', type=float, default=0.9, help='Threshold for feature selection')
    parser.add_argument('--no-balance', action='store_false', dest='rf_balance', help='Do not balance dataset')
    parser.add_argument('--parameter-opt', action='store_true', dest='parameter_optimization',
                        help='Use parameter optimization')

    parser.set_defaults(temporal=True, use_rf=True, rf_balance=True, parameter_optimization=False)

    # Parse command line arguments
    args = parser.parse_args()

    lightgbm_predict(
        args.production_line,
        args.fault_code,
        args.temporal,
        args.use_rf,
        args.rf_threshold,
        args.rf_balance,
        args.parameter_optimization
    )