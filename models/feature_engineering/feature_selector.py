from models.prediction.rf_selector import RFSelector


class FeatureSelector:
    """Feature selection interface using Random Forest algorithm."""

    def __init__(self, random_state=42):
        """Initialize feature selector with fixed random state for reproducibility."""
        self.random_state = random_state
        self.rf_selector = RFSelector(random_state=random_state)

    def select_important_features(self, train_data, test_data, fault_code, threshold=0.9, model_exist=False):
        """
        Select important features using Random Forest.

        Args:
            train_data: Training dataset
            test_data: Testing dataset
            fault_code: Fault code to predict
            threshold: Threshold for feature selection
            model_exist: Whether pre-trained model exists

        Returns:
            X_train_selected: Selected features for training
            X_test_selected: Selected features for testing
            y_train_original: Original training labels
            y_test_original: Original testing labels
        """

        y_train = train_data['label']
        x_train = train_data.drop('label', axis=1)

        y_test = test_data['label']
        x_test = test_data.drop('label', axis=1)


        # Load pre-selected features if model exists
        if model_exist:
            selected_features = self.rf_selector.load_important_features(fault_code)
            x_train_selected = x_train[selected_features]
            x_test_selected = x_test[selected_features]
        else:
            x_train_selected, x_test_selected = self.rf_selector.select_features(
                x_train, x_test,
                y_train, y_test,
                fault_code, threshold
            )

        return x_train_selected, x_test_selected