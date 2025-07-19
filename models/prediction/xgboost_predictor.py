from random import random

import numpy as np
import joblib
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import precision_score
from .parameter_optimizer import ParameterOptimizer

class XGBoostPredictor:
    """XGBoost based predictor with hyperparameter optimization and transfer learning capabilities."""

    def __init__(self, random_state=42):
        """Initialize XGBoost predictor with default parameters and fixed random state."""
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
            'max_depth': [5, 7, 10, 15],
            'random_state': [random_state]  # Ensure XGBoost uses fixed random state
        }
        self.parameter_optimizer = ParameterOptimizer(
            self.initial_param_space, 
            XGBClassifier, 
            random_state=random_state
        )

    def train(self, x_train, x_test, y_train, y_test, parameter_optimization=False):
        """Train the XGBoost model."""
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
            scoring='precision',
            random_state=self.random_state,  # Fixed random state for reproducible search
            n_jobs=-1,
            verbose=1
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

    def save_model(self, model_path):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        joblib.dump(self.model, model_path)

    def load_model(self, model_path):
        """Load a pre-trained model from disk."""
        self.model = joblib.load(model_path)
        return self.model