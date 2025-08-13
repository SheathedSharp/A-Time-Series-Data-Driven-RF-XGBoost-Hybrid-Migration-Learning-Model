import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from utils.progress_display import create_progress_display
from utils.model_evaluation import evaluate_model

class SVMPredictor:
    """SVM BASELINE predictor for fair comparison with hybrid RF-XGBoost method.
    
    This predictor uses raw data without any preprocessing (no temporal features,
    no CBSS sampling, no RF feature selection) to serve as a fair baseline.
    """

    def __init__(self, random_state=42, show_progress=True):
        """Initialize SVM baseline predictor with default parameters and fixed random state."""
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.show_progress = show_progress
        self.progress = create_progress_display() if show_progress else None
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)

    def train(self, x_train, x_test, y_train, y_test):
        """Train the SVM BASELINE model with integrated progress display.
        
        This baseline model uses raw data without any preprocessing for fair comparison.
        """
        if self.show_progress:
            with self.progress.model_training_status() as status:
                return self._train_with_progress(x_train, x_test, y_train, y_test, status)
        else:
            return self._train_without_progress(x_train, x_test, y_train, y_test)
    
    def _train_with_progress(self, x_train, x_test, y_train, y_test, status):
        """Internal baseline training method with progress updates."""
        status.update("Initializing SVM BASELINE training...")
        
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
        
        status.update("Training SVM baseline model...")    
        # Use linear kernel for faster training on large datasets
        self.model = SVC(
            kernel='linear',
            C=1.0,
            class_weight=class_weight_dict,
            random_state=self.random_state,
            probability=True,
            max_iter=10000  # Limit iterations for faster convergence
        )
        self.model.fit(x_train_scaled, y_train)
        
        status.complete("SVM BASELINE model training completed successfully")
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
            
        # Use linear kernel for faster training on large datasets
        self.model = SVC(
            kernel='linear',
            C=1.0,
            class_weight=class_weight_dict,
            random_state=self.random_state,
            probability=True,
            max_iter=10000  # Limit iterations for faster convergence
        )
        self.model.fit(x_train_scaled, y_train)
        
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