import time
import random
import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
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
        
        self.initial_param_space = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }

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
        """Train RF model with parameter optimization."""
        start_time = time.time()
        
        param_space = self.initial_param_space.copy()
        best_precision = 0
        precision_threshold = 0.1

        for iteration in range(10):
            # Use fixed random state for each iteration based on iteration number
            iteration_random_state = self.random_state + iteration
            
            random_search = RandomizedSearchCV(
                estimator=RandomForestClassifier(random_state=self.random_state),
                param_distributions=param_space,
                n_iter=1,
                scoring={'precision': 'precision', 'auc': 'roc_auc'},
                cv=2,
                verbose=0,
                n_jobs=-1,
                refit='precision',
                random_state=iteration_random_state  # Fixed random state for reproducible parameter search
            )

            random_search.fit(x_train, y_train)
            y_pred = random_search.predict(x_test)
            precision = precision_score(y_test, y_pred, zero_division=1)

            if precision > best_precision:
                best_precision = precision
                self.best_model = random_search.best_estimator_

            if precision >= precision_threshold:
                break

            param_space = self._update_param_space(param_space, random_search.best_params_, iteration)

        training_time = time.time() - start_time
        self._print_training_summary(training_time, random_search.best_params_)
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
        selected_features = pd.read_csv(f'./feature/{fault_code}_selected_features.csv')
        return selected_features['feature_name'].tolist()

    def _update_param_space(self, param_space, best_params, iteration):
        """Update parameter space based on best parameters with fixed randomness."""
        # Use iteration-based seed for reproducible parameter space updates
        update_random_state = np.random.RandomState(self.random_state + iteration + 100)
        
        new_param_space = {}
        for param, values in param_space.items():
            if param in best_params:
                best_value = best_params[param]
                if isinstance(values, list) and best_value in values:
                    index = values.index(best_value)
                    new_values = self._get_new_values(values, index)
                else:
                    new_values = values

                if isinstance(values[0], int):
                    new_values = [int(v) for v in new_values if isinstance(v, (int, float))]

                # Use fixed probability with reproducible random state
                if update_random_state.random() < 0.2:
                    new_values = self._add_random_value(values, new_values, update_random_state)
            else:
                new_values = values

            new_param_space[param] = new_values

        return new_param_space

    @staticmethod
    def _get_new_values(values, index):
        """Get new parameter values based on current best value."""
        if index == 0:
            return [values[0], values[1]]
        elif index == len(values) - 1:
            return [values[-2], values[-1]]
        else:
            return [values[index - 1], values[index], values[index + 1]]

    @staticmethod
    def _add_random_value(values, new_values, random_state):
        """Add random value to parameter space with fixed random state."""
        if isinstance(values[0], int):
            new_values.append(random_state.randint(min(values), max(values) + 1))
        elif isinstance(values[0], float):
            new_values.append(random_state.uniform(min(values), max(values)))
        return new_values