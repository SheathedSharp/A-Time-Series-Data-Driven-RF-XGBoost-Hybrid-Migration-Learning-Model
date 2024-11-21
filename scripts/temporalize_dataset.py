import os
from models.feature_engineering.temporal_processor import TemporalFeatureProcessor
from utils.data_loader import DataLoader
from utils.data_process import remove_irrelevant_features, get_irrelevant_features_data, merge_irrelevant_features
from utils.base_utils import get_feature_columns
from config import TEMPORAL_DATA_DIR


def generate_and_save_temporal_features(production_line_code):
    """Generate and save temporal features for a specific fault code

    Args:
        production_line_code (int): Production line code
    """
    # Prepare data
    data_loader = DataLoader()
    data = data_loader.prepare_data(production_line_code, temporal=False)

    # Remove and backup irrelevant features
    irrelevant_features = get_irrelevant_features_data(data)
    data, data_feature_columns = remove_irrelevant_features(data)

    # Process data with temporal
    processor = TemporalFeatureProcessor()
    feature_columns = get_feature_columns(data_feature_columns)
    processed_data = processor.process(data, base_feature_columns=feature_columns, need_temporal_features=True)

    # Merge irrelevant features back to processed data
    processed_data = merge_irrelevant_features(processed_data, irrelevant_features)

    # Save processed data
    output_path = os.path.join(TEMPORAL_DATA_DIR, f'production_line_{production_line_code}.csv')

    processed_data.to_csv(output_path, index=False)

    print(f"Generated time series features:")
    print(f"data set url: {output_path}")
    print(f"Original data set shape: {data.shape}")
    print(f"Processing data set shape: {processed_data.shape}")


if __name__ == "__main__":
    import argparse

    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Generate temporal feature dataset')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')

    # Parse command line arguments
    args = parser.parse_args()

    # Call function with parsed arguments
    generate_and_save_temporal_features(args.production_line)