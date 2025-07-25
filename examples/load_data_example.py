from utils.data_loader import DataLoader
from utils.progress_display import create_progress_display

if __name__ == "__main__":
    progress = create_progress_display()
    
    # Load pipeline fault data
    data_loader = DataLoader()
    data = data_loader.prepare_data(1, temporal=False)
    # data = data_loader.prepare_data(1, temporal=True)
    
    # Display data summary
    progress.display_data_summary(data, "Loaded Data Summary")
