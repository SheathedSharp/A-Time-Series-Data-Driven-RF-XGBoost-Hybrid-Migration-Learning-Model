from random import random
import os

import numpy as np
import joblib
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from .parameter_optimizer import ParameterOptimizer
from utils.progress_display import create_progress_display
from utils.model_evaluation import evaluate_model

class XGBoostPredictor:
    """XGBoost based predictor with hyperparameter optimization and transfer learning capabilities."""

    def __init__(self, random_state=42, show_progress=True):
        """Initialize XGBoost predictor with default parameters and fixed random state."""
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.show_progress = show_progress
        self.progress = create_progress_display() if show_progress else None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
        # Safe parameter space with explicit bounds to prevent out-of-range errors
        self.initial_param_space = {
            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.7, 0.8, 0.9],  # Removed 1.0 to prevent expansion errors
            'colsample_bytree': [0.5, 0.7, 0.8, 0.9],  # Removed 1.0 to prevent expansion errors
            'max_depth': [3, 5, 7, 10, 15],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],
            'reg_lambda': [0.1, 1.0, 10.0],
            'random_state': [random_state]  # Ensure XGBoost uses fixed random state
        }
        self.parameter_optimizer = ParameterOptimizer(
            self.initial_param_space, 
            XGBClassifier, 
            random_state=random_state
        )

    def train(self, x_train, x_test, y_train, y_test, parameter_optimization=False):
        """Train the XGBoost model with integrated progress display."""
        if self.show_progress:
            with self.progress.model_training_status() as status:
                return self._train_with_progress(x_train, x_test, y_train, y_test, 
                                                parameter_optimization, status)
        else:
            return self._train_without_progress(x_train, x_test, y_train, y_test, 
                                              parameter_optimization)
    
    def _train_with_progress(self, x_train, x_test, y_train, y_test, parameter_optimization, status):
        """Internal training method with progress updates."""
        status.update("Initializing XGBoost training...")
        
        # Reset random seed before training
        np.random.seed(self.random_state)
        
        status.update("Scaling features...")
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        if parameter_optimization:
            status.update("Optimizing hyperparameters...")
            best_params = self.parameter_optimizer.optimize(
                x_train_scaled, y_train, x_test_scaled, y_test)
        else:
            status.update("Performing parameter search...")
            best_params = self._simple_parameter_search(x_train_scaled, y_train)

        status.update("Configuring final model...")
        # Ensure random_state is always set in final model
        if 'random_state' not in best_params:
            best_params['random_state'] = self.random_state
        
        # Calculate scale_pos_weight to further handle class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        best_params['scale_pos_weight'] = scale_pos_weight
        
        status.update("Training XGBoost model...")    
        self.model = XGBClassifier(**best_params)
        self.model.fit(x_train_scaled, y_train)
        
        status.complete("XGBoost model training completed successfully")
        return self.model
    
    def _train_without_progress(self, x_train, x_test, y_train, y_test, parameter_optimization):
        """Internal training method without progress updates."""
        # Reset random seed before training
        np.random.seed(self.random_state)
        
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        if parameter_optimization:
            best_params = self.parameter_optimizer.optimize(
                x_train_scaled, y_train, x_test_scaled, y_test)
        else:
            best_params = self._simple_parameter_search(x_train_scaled, y_train)

        # Ensure random_state is always set in final model
        if 'random_state' not in best_params:
            best_params['random_state'] = self.random_state
        
        # Calculate scale_pos_weight to further handle class imbalance
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1
        best_params['scale_pos_weight'] = scale_pos_weight
            
        self.model = XGBClassifier(**best_params)
        self.model.fit(x_train_scaled, y_train)
        
        return self.model

    def _simple_parameter_search(self, x_train, y_train):
        """Perform simple random parameter search."""
        random_search = self._perform_random_search(x_train, y_train, self.initial_param_space)
        return random_search.best_params_

    def _perform_random_search(self, x_train, y_train, param_space, n_iter=20, cv=3):
        """Perform randomized search for hyperparameter optimization."""
        base_model = XGBClassifier(random_state=self.random_state)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='f1', 
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(x_train, y_train)
        return random_search

    def transfer_learning(self, target_x_train, target_y_train):
        """Apply transfer learning to adapt model to new data."""
        if self.model is None:
            raise ValueError("No source model available for transfer learning")

        target_x_scaled = self.scaler.transform(target_x_train)
        self.model.fit(target_x_scaled, target_y_train, xgb_model=self.model)
        return self.model

    def predict(self, x):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        x_scaled = self.scaler.transform(x)
        return self.model.predict(x_scaled), self.model.predict_proba(x_scaled)[:, 1]
    
    def evaluate_and_save(self, x_test, y_test, model_path):
        """Evaluate model performance and save the trained model."""
        if self.model is None:
            raise ValueError("No model to evaluate")
            
        if self.show_progress and self.progress:
            with self.progress.model_training_status() as status:
                status.update("Evaluating model performance...")
                x_test_scaled = self.scaler.transform(x_test)
                evaluate_model(self.model, x_test_scaled, y_test)
                
                status.update("Saving trained model...")
                joblib.dump(self.model, model_path)
                status.complete("Model evaluation and saving completed successfully")
        else:
            x_test_scaled = self.scaler.transform(x_test)
            evaluate_model(self.model, x_test_scaled, y_test)
            joblib.dump(self.model, model_path)

    def save_model(self, model_path, show_progress=None):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Use instance show_progress if not specified
        if show_progress is None:
            show_progress = self.show_progress
            
        if show_progress and self.progress:
            with self.progress.model_training_status() as status:
                status.update("Saving trained model...")
                joblib.dump(self.model, model_path)
                status.complete("Model saved successfully")
        else:
            joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        """Load a pre-trained model from disk."""
        self.model = joblib.load(model_path)
        return self.model