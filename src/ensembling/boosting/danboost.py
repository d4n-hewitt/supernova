import numpy as np


class DanBoost:
    def __init__(self, base_estimator, n_estimators):
        """
        Initialise the Boosting Classifier

        Args:
            base_estimator: The base estimator for boosting
            n_estimators: The number of base estimators in the ensemble
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []

    def fit_estimator(self, X, y):
        """
        Fit a single base estimator

        Args:
            X: Feature data
            y: Target labels
        """
        for _ in range(self.n_estimators):
            model = self.base_estimator(input_shape=X.shape[1], output_shape=1)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict_latest_estimator(self, X):
        """
        Predict using the most recently trained model

        Args:
            X: Feature data
        """
        model = self.models[-1]
        predictions = model.predict(X)
        return predictions

    def calc_error(self, X, y):
        """
        Calculate the absolute difference between predictions and labels

        Args:
            X: Feature data
            y: Target labels
        Returns:
            errors: Absolute differences between predictions and labels
        """
        predictions = self.predict_latest_estimator(self, X)
        errors = np.abs(predictions - y)
        return errors
