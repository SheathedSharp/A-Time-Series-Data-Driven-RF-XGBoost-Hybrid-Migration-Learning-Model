from utils.data_loader import DataLoader
from utils.progress_display import create_progress_display

if __name__ == "__main__":
    progress = create_progress_display()
    
    # Load pipeline fault data
    with progress.data_loading_status() as status:
        status.update("Loading production line data...")
        data_loader = DataLoader()
        data = data_loader.prepare_data(1, temporal=False)
        status.update("Loading temporal data...")
        data = data_loader.prepare_data(1, temporal=True)
        status.complete("Data loading completed")
    
    # Display data summary
    progress.display_data_summary(data, "Loaded Data Summary")
