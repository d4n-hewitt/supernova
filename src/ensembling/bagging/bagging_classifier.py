from src.data_utils.sampling import Sampler


class BaggingClassifier:

    def __init__(self, base_estimator, n_estimators=1, sample_fraction=0.8):
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

    def predict(self, X):
        """
        Make mean predictions using the BaggingClassifier.

        Args:
            X: Feature data.

        Returns:
            list: Predicted labels.
        """
        predictions = [model.predict(X) for model in self.models]
        # Average the predictions from all models
        return sum(predictions) / len(predictions)
