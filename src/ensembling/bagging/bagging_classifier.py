import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from src.data_utils.sampling import Sampler
from src.decorators import require_fitted


class BaggingClassifier:

    def __init__(
        self,
        base_estimator,
        n_estimators=1,
        sample_fraction=0.8,
        combination_method="weighted_mean",
    ):
        """
        Initialize the BaggingClassifier.

        Args:
            base_estimator: The base estimator to use for bagging.
            n_estimators (int): The number of base estimators in the ensemble.
            sample_fraction (float): The fraction of samples to use
                for each base estimator.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.models = []
        self.combination_method = combination_method

    def fit(self, X, y):
        """
        Fit the BaggingClassifier.

        Args:
            X: Feature data.
            y: Target labels.
        """

        self.model_losses = []
        self.model_weights = []

        sampler = Sampler(X, y)
        sample_size = sampler.calculate_sample_size(self.sample_fraction)

        for _ in range(self.n_estimators):
            X_sample, y_sample = sampler.sample_with_replacement(sample_size)
            model = self.base_estimator(
                input_shape=X_sample.shape[1], output_shape=1
            )
            model.fit(X_sample, y_sample)
            self.models.append(model)

            # Calculate loss for each model
            predictions = model.predict(X_sample).flatten()
            loss = log_loss(y_sample, predictions)
            self.model_losses.append(loss)

        # Calculate model weights based on inverse losses
        inverse_losses = [1 / np.array(self.model_losses) + 1e-8]
        self.model_weights = inverse_losses / np.sum(inverse_losses)
        return self

    @require_fitted
    def predict(self, X, combination_method=None):
        """
        Make mean predictions using the BaggingClassifier.

        Args:
            X: Feature data.
            combination_method (str): Method to combine predictions.

        Returns:
            list: Predicted labels.
        """

        if combination_method is None:
            combination_method = self.combination_method

        aggregation_methods = {
            "mean": lambda preds: np.mean(preds, axis=0),
            "median": lambda preds: np.median(preds, axis=0),
            "min": lambda preds: np.min(preds, axis=0),
            "max": lambda preds: np.max(preds, axis=0),
            "weighted_mean": lambda preds: np.average(
                preds, axis=0, weights=self.model_weights
            ),
        }

        predictions = [model.predict(X) for model in self.models]
        predictions = np.array(predictions)

        if combination_method in aggregation_methods:
            result = aggregation_methods[combination_method](predictions)
        elif combination_method == "weighted_mean" and not hasattr(
            self, "model_weights"
        ):
            raise ValueError(
                "Model weights not found. "
                "You must call `fit()` before using weighted_mean."
            )
        else:
            raise ValueError(
                f"Invalid combination method: {combination_method}. "
                f"Choose from {list(aggregation_methods.keys())}."
            )

        return result

    @require_fitted
    def evaluate(self, X, y, combination_method=None):
        """
        Evaluate the BaggingClassifier.

        Args:
            X: Feature data.
            y: Target labels.
            combination_method (str): Method to combine predictions.

        Returns:
            tuple: Loss and accuracy.
        """
        predictions = self.predict(X, combination_method)
        loss = log_loss(y, predictions)
        predictions = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(y, predictions)
        return loss, accuracy
