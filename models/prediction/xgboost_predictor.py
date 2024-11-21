from random import random

import joblib
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import precision_score


class XGBoostPredictor:
    """XGBoost based predictor with hyperparameter optimization and transfer learning capabilities."""

    def __init__(self):
        """Initialize XGBoost predictor with default parameters."""
        self.model = None
        self.scaler = StandardScaler()
        self.initial_param_space = {
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200, 300, 400, 500],
            'subsample': [0.5, 0.7, 0.8, 1.0],
            'colsample_bytree': [0.5, 0.7, 0.8, 1.0],
            'max_depth': [5, 7, 10, 15],
        }

    def train(self, x_train, x_test, y_train, y_test, need_select=False):
        """Train the XGBoost model."""
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        if need_select:
            best_params = self._optimize_parameters_with_selection(x_train_scaled, y_train, x_test_scaled, y_test)
        else:
            best_params = self._simple_parameter_search(x_train_scaled, y_train)


        self.model = XGBClassifier(**best_params)

        self.model.fit(x_train_scaled, y_train)

        return self.model

    def _optimize_parameters_with_selection(self, x_train, y_train, x_test, y_test,
                                            precision_threshold=0.7, max_iterations=10, reset_interval=3):
        """Optimize model parameters with iterative selection process."""
        param_space = self.initial_param_space.copy()
        best_precision = 0
        best_params = None

        for iteration in range(max_iterations):
            if iteration % reset_interval == 0 and iteration > 0:
                param_space = self.initial_param_space.copy()
                self._print_iteration_info("Parameter space reset", iteration)

            random_search = self._perform_random_search(x_train, y_train, param_space)
            precision = self._evaluate_iteration(random_search, x_test, y_test, iteration)

            if precision > best_precision:
                best_precision = precision
                best_params = random_search.best_params_

            if precision >= precision_threshold:
                break

            param_space = self._update_param_space(param_space, random_search.best_params_)

        return best_params

    def _simple_parameter_search(self, x_train, y_train):
        """Perform simple random parameter search."""
        random_search = self._perform_random_search(x_train, y_train, self.initial_param_space)
        return random_search.best_params_

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

    @staticmethod
    def _perform_random_search(x_train, y_train, param_space):
        """Perform random search for hyperparameter optimization."""
        random_search = RandomizedSearchCV(
            estimator=XGBClassifier(),
            param_distributions=param_space,
            n_iter=10,
            scoring={'precision': 'precision', 'auc': 'roc_auc'},
            refit='precision',
            cv=5,
            verbose=0,
            n_jobs=-1,
        )

        random_search.fit(x_train, y_train)
        
        return random_search

    def _evaluate_iteration(self, random_search, x_test, y_test, iteration):
        """Evaluate model performance for current iteration."""
        y_pred = random_search.predict(x_test)
        precision = precision_score(y_test, y_pred)
        
        # 打印评估结果
        eval_data = [
            ["Iteration", iteration + 1],
            ["Precision", f"{precision:.4f}"],
            ["Best Score", f"{random_search.best_score_:.4f}"]
        ]
        print("\nIteration Results:")
        print(tabulate(eval_data, headers=["Metric", "Value"], tablefmt="grid"))
        
        return precision

    def _update_param_space(self, param_space, best_params):
        """Update parameter space based on best parameters."""
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
                
                if random.random() < 0.2:  # 20%的概率添加随机值
                    new_values = self._add_random_value(values, new_values)
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
    def _add_random_value(values, new_values):
        """Add random value to parameter space."""
        if isinstance(values[0], int):
            new_values.append(random.randint(min(values), max(values)))
        elif isinstance(values[0], float):
            new_values.append(random.uniform(min(values), max(values)))
        return new_values

    @staticmethod
    def _print_iteration_info(message, iteration, params=None):
        """Print formatted iteration information."""
        headers = ["Metric", "Value"]
        data = [["Iteration", iteration + 1], ["Message", message]]
        if params:
            for param, value in params.items():
                data.append([param, value])
        print(tabulate(data, headers=headers, tablefmt="grid"))