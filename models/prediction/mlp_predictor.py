import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight
from utils.progress_display import create_progress_display
from utils.model_evaluation import evaluate_model

class MLPPredictor:
    """MLP BASELINE predictor for fair comparison with hybrid RF-XGBoost method.
    
    This predictor uses raw data without any preprocessing (no temporal features,
    no CBSS sampling, no RF feature selection) to serve as a fair baseline.
    """

    def __init__(self, random_state=42, show_progress=True):
        """Initialize MLP baseline predictor with default parameters and fixed random state."""
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.show_progress = show_progress
        self.progress = create_progress_display() if show_progress else None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)

    def train(self, x_train, x_test, y_train, y_test):
        """Train the MLP BASELINE model with integrated progress display.
        
        This baseline model uses raw data without any preprocessing for fair comparison.
        """
        if self.show_progress:
            with self.progress.model_training_status() as status:
                return self._train_with_progress(x_train, x_test, y_train, y_test, status)
        else:
            return self._train_without_progress(x_train, x_test, y_train, y_test)
    
    def _train_with_progress(self, x_train, x_test, y_train, y_test, status):
        """Internal baseline training method with progress updates."""
        status.update("Initializing MLP BASELINE training...")
        
        # Reset random seed before training
        np.random.seed(self.random_state)
        
        status.update("Scaling raw features...")
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        status.update("Configuring baseline model...")
        # Calculate class weights to handle class imbalance in baseline
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
        
        status.update("Training MLP baseline model...")    
        # Use moderate complexity MLP for baseline comparison
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=self.random_state
        )
        
        # Handle class imbalance with sample weights
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        self.model.fit(x_train_scaled, y_train, sample_weight=sample_weights)
        
        status.complete("MLP BASELINE model training completed successfully")
        return self.model
    
    def _train_without_progress(self, x_train, x_test, y_train, y_test):
        """Internal training method without progress updates."""
        # Reset random seed before training
        np.random.seed(self.random_state)
        
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_test_scaled = self.scaler.transform(x_test)

        # Calculate class weights to handle class imbalance
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
            
        # Use moderate complexity MLP for baseline comparison
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50),  # Two hidden layers
            activation='relu',
            solver='adam',
            alpha=0.0001,  # L2 regularization
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=self.random_state
        )
        
        # Handle class imbalance with sample weights
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        self.model.fit(x_train_scaled, y_train, sample_weight=sample_weights)
        
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