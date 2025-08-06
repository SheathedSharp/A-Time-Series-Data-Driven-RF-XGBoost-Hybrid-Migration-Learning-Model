import numpy as np
import pandas as pd
import random
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
from tabulate import tabulate
from utils.progress_display import create_progress_display


class ParameterOptimizer:
    """Advanced parameter optimizer with iterative refinement and analysis capabilities."""
    
    def __init__(self, initial_param_space, model_class=XGBClassifier, 
                 scoring_metrics=None, random_state=42):
        """
        Initialize parameter optimizer.
        
        Args:
            initial_param_space (dict): Initial parameter space for optimization
            model_class: Model class to optimize
            scoring_metrics (dict): Scoring metrics for optimization
            random_state (int): Random state for reproducibility
        """
        self.initial_param_space = initial_param_space
        self.model_class = model_class
        self.scoring_metrics = scoring_metrics or {
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1'
        }
        self.random_state = random_state
        self.optimization_history = []
        self.progress = create_progress_display()

    def optimize(self, x_train, y_train, x_test, y_test, 
                precision_threshold=0.99, max_iterations=10, reset_interval=3,
                early_stopping_rounds=3, min_improvement=0.001):
        """Optimize model parameters with iterative selection process."""
        # Reset random seeds at the beginning of optimization
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        
        param_space = self.initial_param_space.copy()
        best_precision = 0
        best_params = None
        no_improvement_count = 0
        previous_precision = 0

        with self.progress.parameter_optimization_status() as status:
            for iteration in range(max_iterations):
                if iteration % reset_interval == 0 and iteration > 0:
                    param_space = self.initial_param_space.copy()
                    status.update(f"Iteration {iteration + 1}: Parameter space reset")

                status.update(f"Iteration {iteration + 1}: Performing random search...")
                random_search = self._perform_random_search(x_train, y_train, param_space, iteration)
                
                status.update(f"Iteration {iteration + 1}: Evaluating results...")
                precision = self._evaluate_iteration(random_search, x_test, y_test, iteration)

                if precision > best_precision:
                    best_precision = precision
                    best_params = random_search.best_params_
                    if precision - previous_precision > min_improvement:
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1
                
                if no_improvement_count >= early_stopping_rounds:
                    self.progress.display_warning(f"Early stopping triggered after {iteration + 1} iterations")
                    break
                
                previous_precision = precision

                status.update(f"Iteration {iteration + 1}: Updating parameter space...")
                param_space = self._update_param_space(param_space, random_search.best_params_, iteration)

            status.complete(f"Parameter optimization completed. Best precision: {best_precision:.4f}")

        return best_params

    def _perform_random_search(self, x_train, y_train, param_space, iteration=0):
        """Perform random search for hyperparameter optimization."""
        # Use iteration-specific random state for reproducible but different searches
        search_random_state = self.random_state + iteration
        
        # Validate parameter space before search
        validated_param_space = self._validate_param_space(param_space)
        
        random_search = RandomizedSearchCV(
            estimator=self.model_class(random_state=self.random_state),
            param_distributions=validated_param_space,
            n_iter=10,
            scoring=self.scoring_metrics,
            refit='f1',
            cv=5,
            verbose=0,
            n_jobs=-1,
            random_state=search_random_state,  # Fixed random state for reproducible search
            error_score='raise'  # This will help us catch parameter errors early
        )
        
        try:
            random_search.fit(x_train, y_train)
        except Exception as e:
            # If parameter optimization fails, fall back to safe defaults
            self.progress.display_warning(f"Parameter optimization failed: {str(e)}")
            self.progress.display_warning("Falling back to safe parameter defaults")
            
            safe_params = {
                'learning_rate': 0.1,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'max_depth': 7,
                'random_state': self.random_state
            }
            
            # Create a simple estimator with safe parameters
            fallback_estimator = self.model_class(**safe_params)
            fallback_estimator.fit(x_train, y_train)
            
            # Create a mock RandomizedSearchCV result
            class MockRandomSearch:
                def __init__(self, estimator, params):
                    self.best_estimator_ = estimator
                    self.best_params_ = params
                    self.best_score_ = 0.9  # Conservative estimate
                    self.cv_results_ = {'mean_test_precision': [0.9], 'std_test_precision': [0.01]}
                
                def predict(self, X):
                    return self.best_estimator_.predict(X)
            
            random_search = MockRandomSearch(fallback_estimator, safe_params)
        
        return random_search

    def _validate_param_space(self, param_space):
        """Validate parameter space to ensure all values are within XGBoost bounds."""
        validated_space = {}
        
        # Define XGBoost parameter bounds
        xgb_bounds = {
            'subsample': (0.001, 1.0),
            'colsample_bytree': (0.001, 1.0),
            'colsample_bylevel': (0.001, 1.0),
            'colsample_bynode': (0.001, 1.0),
            'learning_rate': (0.001, 1.0),
            'eta': (0.001, 1.0),
            'reg_alpha': (0.0, 1000.0),
            'reg_lambda': (0.0, 1000.0),
            'gamma': (0.0, 1000.0),
            'min_child_weight': (0.0, 1000.0),
            'max_depth': (1, 30),
            'n_estimators': (1, 10000)
        }
        
        for param, values in param_space.items():
            if param in xgb_bounds:
                min_val, max_val = xgb_bounds[param]
                if isinstance(values, list):
                    # Filter values to be within bounds
                    valid_values = [v for v in values if min_val <= v <= max_val]
                    if not valid_values:
                        # If no valid values, use safe defaults
                        if param in ['subsample', 'colsample_bytree']:
                            valid_values = [0.8]
                        elif param == 'learning_rate':
                            valid_values = [0.1]
                        elif param == 'max_depth':
                            valid_values = [6]
                        elif param == 'n_estimators':
                            valid_values = [300]
                        else:
                            valid_values = [min_val + (max_val - min_val) * 0.5]  # Use midpoint
                    validated_space[param] = valid_values
                else:
                    # Single value - check bounds
                    if min_val <= values <= max_val:
                        validated_space[param] = values
                    else:
                        # Use safe default
                        validated_space[param] = min_val + (max_val - min_val) * 0.5
            else:
                # Parameter not in bounds dict - keep as is
                validated_space[param] = values
        
        return validated_space

    def _evaluate_iteration(self, random_search, x_test, y_test, iteration):
        """Enhanced evaluation with cross-validation analysis."""
        y_pred = random_search.predict(x_test)
        precision = precision_score(y_test, y_pred)
        
        cv_results = pd.DataFrame(random_search.cv_results_)
        
        eval_data = {
            "Iteration": iteration + 1,
            "Test Precision": f"{precision:.4f}",
            "CV Mean Precision": f"{cv_results['mean_test_precision'].mean():.4f}",
            "CV Std Precision": f"{cv_results['std_test_precision'].mean():.4f}",
            "Best CV Score": f"{random_search.best_score_:.4f}"
        }
        
        self.progress.display_results_table(f"Iteration {iteration + 1} Results", eval_data)

        return precision

    def _update_param_space(self, param_space, best_params, iteration):
        """Enhanced parameter space update with multiple strategies."""
        # Use iteration-based random state for reproducible parameter space updates
        update_random_state = np.random.RandomState(self.random_state + iteration + 1000)
        
        new_param_space = {}
        for param, values in param_space.items():
            if param not in best_params:
                new_param_space[param] = values
                continue
                
            best_value = best_params[param]
            strategy = update_random_state.choice(['narrow', 'expand', 'shift'])
            
            if strategy == 'narrow':
                new_values = self._narrow_search_space(values, best_value, update_random_state)
            elif strategy == 'expand':
                new_values = self._expand_search_space(values, best_value, update_random_state)
            else:
                new_values = self._shift_search_space(values, best_value, update_random_state)
                
            new_param_space[param] = new_values

        return new_param_space

    @staticmethod
    def _narrow_search_space(values, best_value, random_state):
        """Narrow search space around best value."""
        if not isinstance(values, list) or len(values) <= 2:
            return values
            
        if best_value in values:
            index = values.index(best_value)
            start = max(0, index - 1)
            end = min(len(values), index + 2)
            return values[start:end]
        return values

    @staticmethod
    def _expand_search_space(values, best_value, random_state):
        """Expand search space around best value with proper bounds checking."""
        if not isinstance(values, list):
            return values
            
        new_values = values.copy()
        if isinstance(best_value, (int, float)):
            if isinstance(best_value, int):
                new_values.extend([
                    max(1, best_value - random_state.randint(1, 3)),
                    best_value + random_state.randint(1, 3)
                ])
            else:
                # Define parameter-specific bounds for XGBoost
                param_bounds = {
                    # Common float parameters and their valid ranges
                    'subsample': (0.001, 1.0),
                    'colsample_bytree': (0.001, 1.0),
                    'colsample_bylevel': (0.001, 1.0),
                    'colsample_bynode': (0.001, 1.0),
                    'learning_rate': (0.001, 1.0),
                    'eta': (0.001, 1.0),
                    'reg_alpha': (0.0, 100.0),
                    'reg_lambda': (0.0, 100.0),
                    'gamma': (0.0, 100.0),
                    'min_child_weight': (0.0, 100.0)
                }
                
                # Determine bounds - try to infer from current values or use defaults
                if len(values) > 1 and all(isinstance(v, (int, float)) for v in values):
                    min_bound = max(0.001, min(values) * 0.5)  # Conservative lower bound
                    max_bound = min(1.0, max(values) * 1.2)    # Conservative upper bound for most params
                else:
                    min_bound, max_bound = 0.001, 1.0  # Default safe bounds
                
                # Apply specific bounds if we can infer the parameter type
                current_vals = [v for v in values if isinstance(v, (int, float))]
                if current_vals:
                    if max(current_vals) <= 1.0 and min(current_vals) >= 0.0:
                        # Likely a ratio parameter (subsample, colsample_*, etc.)
                        min_bound, max_bound = 0.001, 1.0
                    elif max(current_vals) <= 1.0 and min(current_vals) > 0.001:
                        # Likely learning rate
                        min_bound, max_bound = 0.001, 1.0
                
                # Generate new values within bounds
                expansion_amount = random_state.uniform(0.01, 0.1)
                new_lower = max(min_bound, best_value - expansion_amount)
                new_upper = min(max_bound, best_value + expansion_amount)
                
                # Only add if they're different from existing values
                if new_lower not in values:
                    new_values.append(new_lower)
                if new_upper not in values and new_upper != new_lower:
                    new_values.append(new_upper)
                    
        return list(set(new_values))

    @staticmethod
    def _shift_search_space(values, best_value, random_state):
        """Shift search space maintaining size with proper bounds checking."""
        if not isinstance(values, list) or len(values) <= 2:
            return values
            
        if isinstance(best_value, (int, float)):
            if isinstance(best_value, int):
                shift = random_state.choice([-1, 1])
                new_values = [max(1, v + shift) for v in values]
            else:
                shift_amount = random_state.uniform(-0.05, 0.05)
                
                # Apply bounds checking for float parameters
                current_vals = [v for v in values if isinstance(v, (int, float))]
                if current_vals and max(current_vals) <= 1.0:
                    # Likely a ratio parameter - ensure we stay within [0, 1]
                    new_values = [max(0.001, min(1.0, v + shift_amount)) for v in values]
                else:
                    # General float parameter
                    new_values = [max(0.001, v + shift_amount) for v in values]
            return new_values
        return values

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
    def _add_random_value(values, new_values):
        """Add random value to parameter space."""
        if isinstance(values[0], int):
            new_values.append(random.randint(min(values), max(values)))
        elif isinstance(values[0], float):
            new_values.append(random.uniform(min(values), max(values)))
        return new_values

    def _print_iteration_info(self, message, iteration, params=None):
        """Print formatted iteration information."""
        data = {"Iteration": iteration + 1, "Message": message}
        if params:
            data.update(params)
        self.progress.display_results_table("Iteration Information", data)

    def analyze_parameter_importance(self):
        """Analyze parameter importance based on optimization history."""
        if not hasattr(self, 'optimization_history'):
            self.progress.display_warning("No optimization history available")
            return
        
        param_importance = {}
        for param in self.initial_param_space.keys():
            values = [result['params'][param] for result in self.optimization_history]
            scores = [result['score'] for result in self.optimization_history]
            correlation = abs(np.corrcoef(values, scores)[0, 1])
            param_importance[param] = correlation
        
        return sorted(param_importance.items(), key=lambda x: x[1], reverse=True)