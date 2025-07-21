import os
import subprocess
import pandas as pd
from config import RAW_DATA_DIR, get_production_line_data_path
from utils.progress_display import create_progress_display

class DataLoader:
    """Data loader class to read and prepare data for training and testing."""

    def __init__(self, data_dir=RAW_DATA_DIR):
        self.data_dir = data_dir
        self.progress = create_progress_display()

    @staticmethod
    def read_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        data = pd.read_csv(file_path)
        data.fillna(0, inplace=True)
        return data

    def prepare_data(self, production_line_code: int, temporal: bool = False):
        """Prepare data for training and testing.

        Args:
            production_line_code (int): Production line code (1-4)
            temporal (bool): Whether to use temporal features

        Returns:
            data: Prepared dataset

        Raises:
            ValueError: If production_line_code is invalid
        """
        if not 1 <= production_line_code <= 4:
            raise ValueError("Pipeline code must be between 1 and 4")

        file_path = get_production_line_data_path(production_line_code, temporal=temporal)

        # If temporal data is requested and not found, generate it
        if temporal and not os.path.exists(file_path):
            with self.progress.temporal_processing_status() as status:
                status.update(f"Temporal data not found for production line {production_line_code}")
                status.update("Generating temporal features...")
                
                try:
                    result = subprocess.run(
                        ['python', 'scripts/temporalize_dataset.py', 
                         '--production_line', str(production_line_code)],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    status.complete("Temporal features generated successfully!")

                    if not os.path.exists(file_path):
                        raise FileNotFoundError(
                            "Temporal data file was not generated successfully"
                        )
                        
                except subprocess.CalledProcessError as e:
                    self.progress.display_error("Failed to generate temporal features", 
                                             f"Exit code: {e.returncode}\nError output: {e.stderr}")
                    raise RuntimeError("Failed to generate temporal features")
                    
                except Exception as e:
                    self.progress.display_error("Unexpected error while generating temporal features", 
                                             str(e))
                    raise

        data = self.read_data(file_path)
        return data