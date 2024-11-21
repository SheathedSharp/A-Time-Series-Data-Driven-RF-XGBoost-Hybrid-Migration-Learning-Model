from old_files import TemporalFeatureProcessor
from utils.data_loader import DataLoader
from utils.data_process import remove_irrelevant_features, get_irrelevant_features_data, merge_irrelevant_features
from utils.base_utils import get_feature_columns

if __name__ == "__main__":
    data_loader = DataLoader()
    pipeline_code, fault_code, fault_description, data = data_loader.prepare_data(1, temporal=False)
    print(f"Original data shape: {data.shape}")

    # Remove and backup irrelevant features
    irrelevant_features = get_irrelevant_features_data(data)
    data, data_feature_columns = remove_irrelevant_features(data)

    # Process data with temporal
    processor = TemporalFeatureProcessor()
    feature_columns = get_feature_columns(data_feature_columns)
    processed_data = processor.process(data, base_feature_columns=feature_columns, need_temporal_features=True)

    # Merge irrelevant features back to processed data
    processed_data = merge_irrelevant_features(processed_data, irrelevant_features)

    print(f"Processed data shape: {processed_data.shape}")