from random import random

import numpy as np
import pandas as pd
import joblib
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score
from .parameter_optimizer import ParameterOptimizer

class LightGBMPredictor:
    """LightGBM based predictor with hyperparameter optimization capabilities."""

    def __init__(self, random_state=42):
        """Initialize LightGBM predictor with default parameters and fixed random state."""
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        
        self.initial_param_space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.7, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
            'max_depth': [3, 5, 7, 10], 
            'num_leaves': [15, 31, 50, 100],  
            'min_child_samples': [10, 20, 50], 
            'min_child_weight': [1, 5, 10],
            'min_split_gain': [0.0, 0.1, 0.2], 
            'reg_alpha': [0.0, 0.1, 0.5], 
            'reg_lambda': [0.0, 0.1, 0.5],  
            'random_state': [random_state]  # Ensure LightGBM uses fixed random state
        }
        self.parameter_optimizer = ParameterOptimizer(
            self.initial_param_space, 
            LGBMClassifier, 
            random_state=random_state
        )

    def train(self, x_train, x_test, y_train, y_test, parameter_optimization=False):
        """Train the LightGBM model."""
        # Reset random seed before training
        np.random.seed(self.random_state)
        
        # Preserve feature names during scaling
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)
        
        # Convert back to DataFrame with feature names
        if hasattr(x_train, 'columns'):
            x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns, index=x_train.index)
            x_test_scaled = pd.DataFrame(x_test_scaled, columns=x_test.columns, index=x_test.index)

        if parameter_optimization:
            best_params = self.parameter_optimizer.optimize(
                x_train_scaled, y_train, x_test_scaled, y_test)
        else:
            best_params = self._simple_parameter_search(x_train_scaled, y_train)

        # Ensure random_state is always set in final model
        if 'random_state' not in best_params:
            best_params['random_state'] = self.random_state
            
        self.model = LGBMClassifier(**best_params, verbose=-1)
        self.model.fit(x_train_scaled, y_train)
        
        return self.model

    def _simple_parameter_search(self, x_train, y_train):
        """Perform simple random parameter search."""
        random_search = self._perform_random_search(x_train, y_train, self.initial_param_space)
        return random_search.best_params_

    def _perform_random_search(self, x_train, y_train, param_space, n_iter=20, cv=3):
        """Perform randomized search for hyperparameter optimization."""
        base_model = LGBMClassifier(random_state=self.random_state, verbose=-1)
        
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_space,
            n_iter=n_iter,
            cv=cv,
            scoring='precision',
            random_state=self.random_state,  # Fixed random state for reproducible search
            n_jobs=-1,
            verbose=0
        )
        
        random_search.fit(x_train, y_train)
        return random_search

    def predict(self, x):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")

        x_scaled = self.scaler.transform(x)
        # Preserve feature names to avoid warnings
        if hasattr(x, 'columns'):
            x_scaled = pd.DataFrame(x_scaled, columns=x.columns, index=x.index)
        
        return self.model.predict(x_scaled), self.model.predict_proba(x_scaled)[:, 1]

    def save_model(self, model_path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        """Load a pre-trained model from disk."""
        self.model = joblib.load(model_path)
        return self.model 