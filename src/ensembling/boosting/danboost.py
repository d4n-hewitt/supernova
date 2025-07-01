import numpy as np


class DanBoost:
    def __init__(self, base_estimator, n_estimators, initial_weights=None):
        """
        Initialise the Boosting Classifier

        Args:
            base_estimator: The base estimator for boosting
            n_estimators: The number of base estimators in the ensemble
            initial_weights: Initial sample weights for the first iteration (default is None)
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.models = []
        self.sample_weights = None
        self.initial_weights = initial_weights

    def fit_ensemble(
        self, X, y, error_type="absolute", sample_scheme="linear", **fit_kwargs
    ):
        """
        Fit an ensemble of models using the base estimator, iteratively adjusting sample weights

        Args:
            X: Feature data
            y: Target labels
            error_type: Type of error to calculate (default is 'absolute')
            sample_scheme: Scheme for computing sample weights (default is 'linear')
        Raises:
            NotImplementedError: If the base estimator does not support sample weights
            ValueError: If an unsupported error type or sample scheme is provided
        """
        self.models = []
        self.sample_weights = (
            self.initial_weights or np.ones(X.shape[0]) / X.shape[0]
        )
        for iteration in range(self.n_estimators):
            model = self.base_estimator(input_shape=X.shape[1], output_shape=1)
            try:
                model.fit(
                    X, y, sample_weight=self.sample_weights, **fit_kwargs
                )
            except TypeError:
                raise NotImplementedError(
                    "The base estimator must support sample weights."
                )
            self.models.append(model)

            if iteration < self.n_estimators - 1:
                errors = self.calc_error(X, y)
                self.sample_weights = self.compute_sample_weights(errors)

    def predict_ensemble(self, X):
        """
        Predict using the ensemble of models

        Args:
            X: Feature data
        """
        for model in self.models:
            if not hasattr(model, "predict"):
                raise NotImplementedError(
                    "The base estimator must implement a predict method."
                )
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)

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
        predictions = self.models[-1].predict(X)
        differences = predictions - y

        if error_type not in error_type_dict:
            raise ValueError(
                f"Unsupported error type: {error_type}. "
                f"Supported types are: {list(error_type_dict.keys())}"
            )

        return error_type_dict[error_type](differences)

    def compute_sample_weights(self, errors, sample_scheme="linear"):
        """
        Compute sample weights based on errors

        Args:
            errors: Computed errors from the previous iteration
            sample_scheme: Scheme for computing sample weights (default is 'linear')
        Returns:
            sample_weights: Computed sample weights based on the specified scheme
        Raises:
            ValueError: If an unsupported sample scheme is provided
        """
        sample_scheme_dict = {
            "linear": lambda x: np.abs(x),
            "exponential": lambda x: np.exp(x),
        }

        if sample_scheme not in sample_scheme_dict:
            raise ValueError(
                f"Unsupported sample scheme: {sample_scheme}. "
                f"Supported schemes are: {list(sample_scheme_dict.keys())}"
            )

        weights = (
            sample_scheme_dict[sample_scheme](errors) + 1e-10
        )  # Adding a small constant to avoid zero weights
        normalised_weights = weights / np.sum(weights)
        return normalised_weights
