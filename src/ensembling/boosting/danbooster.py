import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from src.decorators import require_fitted
from src.ensembling.boosting.error_calculation import DBErrorCalculator


class DanBooster:
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
        self.combination_method = "mean"

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

        error_calculator = DBErrorCalculator(
            error_type=error_type,
            sample_scheme=sample_scheme,
            error_combination="linear_decay",
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
                predictions = model.predict(X).reshape(-1)
                error_calculator.calc_error(predictions, y)
                self.sample_weights = error_calculator.compute_sample_weights()

    @require_fitted
    def predict_ensemble(self, X, combination_method="mean"):
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

        if combination_method is None:
            combination_method = self.combination_method

        aggregation_methods = {
            "mean": lambda preds: np.mean(preds, axis=0).reshape(-1),
            "median": lambda preds: np.median(preds, axis=0).reshape(-1),
            "min": lambda preds: np.min(preds, axis=0).reshape(-1),
            "max": lambda preds: np.max(preds, axis=0).reshape(-1),
        }

        if combination_method in aggregation_methods:
            aggregated_result = aggregation_methods[combination_method](
                predictions
            )

        return aggregated_result

    @require_fitted
    def evaluate(self, X, y):
        """
        Evaluate the ensemble on the given data

        Args:
            X: Feature data
            y: Target labels
        Returns:
            tuple: Loss and accuracy
        """
        predictions = self.predict_ensemble(X, "median")
        loss = log_loss(y, predictions)
        predictions = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(y, predictions)
        return (loss, accuracy)
