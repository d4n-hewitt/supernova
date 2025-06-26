import numpy as np
from sklearn.metrics import accuracy_score, log_loss

from src.data_utils.sampling import Sampler


class BaggingClassifier:

    def __init__(
        self,
        base_estimator,
        n_estimators=1,
        sample_fraction=0.8,
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
        self.combination_method = "mean"

    def fit(self, X, y):
        """
        Fit the BaggingClassifier.

        Args:
            X: Feature data.
            y: Target labels.
        """

        sampler = Sampler(X, y)
        sample_size = sampler.calculate_sample_size(self.sample_fraction)
        for _ in range(self.n_estimators):
            X_sample, y_sample = sampler.sample_with_replacement(sample_size)
            model = self.base_estimator(
                input_shape=X_sample.shape[1], output_shape=1
            )
            model.fit(X_sample, y_sample)
            self.models.append(model)
        return self

    def predict(self, X, combination_method=None):
        """
        Make mean predictions using the BaggingClassifier.

        Args:
            X: Feature data.

        Returns:
            list: Predicted labels.
        """

        if combination_method is None:
            combination_method = self.combination_method

        aggregation_methods = {
            "mean": np.mean,
            "median": np.median,
            "min": np.min,
            "max": np.max,
        }

        predictions = [model.predict(X) for model in self.models]
        predictions = np.array(predictions)

        if combination_method in aggregation_methods:
            result = aggregation_methods[combination_method](
                predictions, axis=0
            )
        else:
            raise ValueError(
                f"Invalid combination method: {combination_method}. "
                f"Choose from {list(aggregation_methods.keys())}."
            )

        return result

    def evaluate(self, X, y, combination_method=None):
        """
        Evaluate the BaggingClassifier.

        Args:
            X: Feature data.
            y: Target labels.

        Returns:
            tuple: Loss and accuracy.
        """
        predictions = self.predict(X, combination_method)
        loss = log_loss(y, predictions)
        predictions = (predictions > 0.5).astype(int)
        accuracy = accuracy_score(y, predictions)
        return loss, accuracy
