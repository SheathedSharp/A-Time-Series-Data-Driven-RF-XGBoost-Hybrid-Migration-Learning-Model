import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils.progress_display import create_progress_display


def set_global_random_state(random_state=42):
    """
    Set global random seed to ensure experiment reproducibility
    
    Args:
        random_state (int): Random seed, default is 42
    """
    random.seed(random_state)
    np.random.seed(random_state)
    
    # Set pandas random seed if available
    try:
        pd.set_option('mode.chained_assignment', None)
    except:
        pass
    
    progress = create_progress_display()
    progress.display_success(f"Global random seed set to: {random_state}")


def get_feature_columns(feature_columns):
    """Get the list of feature columns."""
    return [col for col in feature_columns if col not in ['Date', 'Time']]


def print_data_summary(data, title="Data Summary"):
    """Print a summary of the dataset using progress display."""
    progress = create_progress_display()
    progress.display_data_summary(data, title)


def validate_data_integrity(data, required_columns=None):
    """
    Validate data integrity
    
    Args:
        data (pd.DataFrame): Data to validate
        required_columns (list): List of required column names
        
    Returns:
        bool: Whether data is valid
    """
    progress = create_progress_display()
    
    if data is None or data.empty:
        progress.display_error("Data is empty")
        return False
    
    if required_columns:
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            progress.display_error(f"Missing required columns: {missing_cols}")
            return False
    
    # Check for columns with all NaN values
    nan_columns = [col for col in data.columns if data[col].isna().all()]
    if nan_columns:
        progress.display_warning(f"Found columns with all NaN values: {nan_columns}")
    
    return True

