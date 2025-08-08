import time
import random
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from config import FEATURE_DIR
from utils.model_evaluation import evaluate_model
from utils.progress_display import create_progress_display


class RFSelector:
    """Random Forest-based feature selector with optimized parameter tuning."""
    
    def __init__(self, random_state=42):
        """Initialize RF selector with fixed random state for reproducibility."""
        self.random_state = random_state
        self.best_model = None
        self.progress = create_progress_display()
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        random.seed(random_state)
        

    def select_features(self, x_train, x_test, y_train, y_test, fault_code, threshold):
        """
        Select important features using Random Forest with optimized parameters.
        """
        # Reset random seeds before each selection to ensure reproducibility
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        with self.progress.feature_selection_status() as status:
            status.update("Training optimized Random Forest model...")
            # Train RF model with optimized parameters
            self._train_optimized_model(x_train, y_train, x_test, y_test)

            status.update("Selecting important features...")
            # Select important features
            selected_features = self._select_important_features(
                x_train, self.best_model.feature_importances_, threshold
            )

            status.update("Saving selected features...")
            # Save selected features
            self._save_important_features(selected_features, fault_code)

            status.complete(f"Feature selection completed. Selected {len(selected_features)} features")

        # Return selected feature datasets
        return (x_train[selected_features],
                x_test[selected_features])

    def _train_optimized_model(self, x_train, y_train, x_test, y_test):
        """Train RF model with reasonable parameters for feature selection."""
        start_time = time.time()
        
        params = {
            'n_estimators': 100,      # Good balance of accuracy and speed
            'max_depth': 10,          # Prevent overfitting while capturing patterns
            'min_samples_split': 5,   # Conservative splitting
            'min_samples_leaf': 2,    # Allow detailed splits
            'max_features': 'sqrt',   # Standard recommendation for classification
            'random_state': self.random_state,
            'n_jobs': -1              # Use all available cores
        }
        
        # Train model with fixed parameters
        self.best_model = RandomForestClassifier(**params)
        self.best_model.fit(x_train, y_train)
        
        training_time = time.time() - start_time
        self._print_training_summary(training_time, params)
        evaluate_model(self.best_model, x_test, y_test)

    def _print_training_summary(self, training_time, best_params):
        """Print training summary in tabular format."""
        # Prepare best parameters table data
        param_data = {param: value for param, value in best_params.items()}
        param_data["Training Time"] = f"{training_time:.2f} seconds"
        
        self.progress.display_results_table("Random Forest Training Summary", param_data)

    def _select_important_features(self, x_train, feature_importances, threshold=0.90):
        """Select features using greedy approach."""
        total_importance = sum(feature_importances)
        sorted_indices = np.argsort(feature_importances)[::-1]

        accumulated_importance = 0
        selected_indices = []
        importances = []
        
        for idx in sorted_indices:
            current_importance = feature_importances[idx]
            accumulated_importance += current_importance
            selected_indices.append(idx)
            importances.append(current_importance)
            if accumulated_importance >= total_importance * threshold:
                break
        
        selected_features = x_train.columns[selected_indices]

        # Display feature selection summary
        feature_summary = {}
        for i, (feature, importance) in enumerate(zip(selected_features, importances)):
            feature_summary[f"Feature {i+1}"] = f"{feature} ({importance/total_importance:.2%})"

        self.progress.display_results_table(f"Feature Selection Summary (Threshold: {threshold:.2%})", feature_summary)

        # Display selection summary
        summary_data = {
            "Selected Features Count": len(selected_features),
            "Total Features Count": len(x_train.columns),
            "Selection Threshold": f"{threshold:.2%}",
            "Achieved Importance": f"{(accumulated_importance/total_importance):.2%}"
        }
        
        self.progress.display_results_table("Selection Summary", summary_data)
        
        return selected_features

    @staticmethod
    def _save_important_features(selected_features, fault_code):
        """Save selected features to CSV file."""
        selected_features_df = pd.DataFrame(selected_features, columns=['feature_name'])
        feature_df_path = f'{FEATURE_DIR}/{fault_code}_selected_features.csv'
        selected_features_df.to_csv(feature_df_path, index=False)

    @staticmethod
    def load_important_features(fault_code):
        """Load pre-selected features from CSV file."""
        selected_features = pd.read_csv(f'./data/selected_features/{fault_code}_selected_features.csv')
        return selected_features['feature_name'].tolist()