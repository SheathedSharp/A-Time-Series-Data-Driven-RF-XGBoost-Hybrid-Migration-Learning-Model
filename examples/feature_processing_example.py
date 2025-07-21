from old_files import TemporalFeatureProcessor
from utils.data_loader import DataLoader
from utils.data_process import remove_irrelevant_features, get_irrelevant_features_data, merge_irrelevant_features
from utils.base_utils import get_feature_columns
from utils.progress_display import create_progress_display

if __name__ == "__main__":
    progress = create_progress_display()
    
    with progress.temporal_processing_status() as status:
        status.update("Loading raw data...")
        data_loader = DataLoader()
        pipeline_code, fault_code, fault_description, data = data_loader.prepare_data(1, temporal=False)
        
        # Display original data shape
        progress.display_data_summary(data, "Original Data Summary")

        # Remove and backup irrelevant features
        status.update("Processing feature columns...")
        irrelevant_features = get_irrelevant_features_data(data)
        data, data_feature_columns = remove_irrelevant_features(data)

        # Process data with temporal
        status.update("Generating temporal features...")
        processor = TemporalFeatureProcessor()
        feature_columns = get_feature_columns(data_feature_columns)
        processed_data = processor.process(data, base_feature_columns=feature_columns, need_temporal_features=True)

        # Merge irrelevant features back to processed data
        status.update("Merging features...")
        processed_data = merge_irrelevant_features(processed_data, irrelevant_features)

        status.complete("Temporal feature processing completed")

    # Display processed data summary
    progress.display_data_summary(processed_data, "Processed Data Summary")