import os
import subprocess
import pandas as pd
from config import get_feature_path
from utils.progress_display import create_progress_display


def get_selected_features(production_line_code, fault_code, rf_threshold, rf_balance, random_state=42):
    """
    Get or generate selected features for the specified fault code.
    
    Args:
        production_line_code (int): Production line code
        fault_code (int): Fault code
        rf_threshold (float): Threshold for feature selection
        rf_balance (bool): Whether to balance dataset
        random_state (int): Random state for reproducibility
        
    Returns:
        list: List of selected feature names
        
    Raises:
        RuntimeError: If feature selection fails
    """
    features_path = get_feature_path(fault_code)
    progress = create_progress_display()
    
    with progress.feature_selection_status() as status:
        if os.path.exists(features_path):
            status.update("Loading existing feature selection...")
            selected_features = pd.read_csv(features_path)['feature_name'].tolist()
            status.complete(f"Loaded {len(selected_features)} pre-selected features")
        else:
            status.update("No existing feature selection found. Running feature selection...")

            try:
                cmd = [
                    'python', 'scripts/select_features.py',
                    '--production_line', str(production_line_code),
                    '--fault_code', str(fault_code),
                    '--threshold', str(rf_threshold),
                    '--random-state', str(random_state)  # Pass random seed
                ]

                if not rf_balance:  # Only add --no-balance parameter if balance is not needed
                    cmd.append('--no-balance')

                status.update("Running Random Forest feature selection script...")
                
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                # Check if file was generated
                if not os.path.exists(features_path):
                    raise FileNotFoundError("Feature selection file was not generated successfully")
                    
                selected_features = pd.read_csv(features_path)['feature_name'].tolist()
                status.complete(f"Feature selection completed. Selected {len(selected_features)} features")
                
            except subprocess.CalledProcessError as e:
                progress.display_error("Failed to perform feature selection", 
                                    f"Exit code: {e.returncode}\nError output: {e.stderr}")
                raise RuntimeError("Failed to perform feature selection")
                
            except Exception as e:
                progress.display_error("Unexpected error during feature selection", str(e))
                raise
            
    return selected_features


def apply_feature_selection(x_train, x_test, production_line_code, fault_code, 
                          rf_threshold=0.9, rf_balance=True, random_state=42):
    """
    Apply feature selection to training and test datasets.
    
    Args:
        x_train (pd.DataFrame): Training features
        x_test (pd.DataFrame): Test features
        production_line_code (int): Production line code
        fault_code (int): Fault code
        rf_threshold (float): Threshold for feature selection
        rf_balance (bool): Whether to balance dataset
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (x_train_selected, x_test_selected) with selected features
    """
    selected_features = get_selected_features(
        production_line_code, 
        fault_code, 
        rf_threshold, 
        rf_balance,
        random_state
    )
    
    return x_train[selected_features], x_test[selected_features] 