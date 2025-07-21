import os
from models.feature_engineering.temporal_processor import TemporalFeatureProcessor
from utils.data_loader import DataLoader
from utils.data_process import remove_irrelevant_features, get_irrelevant_features_data, merge_irrelevant_features
from utils.base_utils import get_feature_columns
from utils.progress_display import create_progress_display
from config import TEMPORAL_DATA_DIR


def generate_and_save_temporal_features(production_line_code):
    """Generate and save temporal features for a specific fault code

    Args:
        production_line_code (int): Production line code
    """
    progress = create_progress_display()
    
    with progress.temporal_processing_status() as status:
        # Prepare data
        status.update("Loading raw data...")
        data_loader = DataLoader()
        data = data_loader.prepare_data(production_line_code, temporal=False)

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

        # Save processed data
        status.update("Saving processed data...")
        output_path = os.path.join(TEMPORAL_DATA_DIR, f'production_line_{production_line_code}.csv')
        processed_data.to_csv(output_path, index=False)

        # Display results
        status.complete("Temporal features generated successfully")
        
        # Show summary
        summary_data = {
            "Output Path": output_path,
            "Original Shape": str(data.shape),
            "Processed Shape": str(processed_data.shape),
            "Feature Increase": f"{processed_data.shape[1] - data.shape[1]} features"
        }
        progress.display_results_table("Temporal Feature Generation Summary", summary_data)


if __name__ == "__main__":
    import argparse

    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Generate temporal feature dataset')
    parser.add_argument('--production_line', type=int, required=True, help='Production line code')

    # Parse command line arguments
    args = parser.parse_args()

    # Call function with parsed arguments
    generate_and_save_temporal_features(args.production_line)