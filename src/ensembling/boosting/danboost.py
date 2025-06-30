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

    def calc_error(self, X, y, error_type="absolute"):
        """
        Calculate the absolute difference between predictions and labels

        Args:
            X: Feature data
            y: Target labels
            error_type: Type of error to calculate (default is 'absolute')
        Returns:
            errors: Computed errors based on the specified error type
        Raises:
            ValueError: If an unsupported error type is provided
        """
        error_type_dict = {
            "absolute": lambda x: np.abs(x),
            "squared": lambda x: np.square(x),
        }
        predictions = self.predict_latest_estimator(X)
        differences = predictions - y

        if error_type not in error_type_dict:
            raise ValueError(
                f"""Unsupported error type: {error_type}.
                Supported types are: {list(error_type_dict.keys())}"""
            )

        return error_type_dict[error_type](differences)
