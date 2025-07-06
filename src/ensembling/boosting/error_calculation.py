import numpy as np


class DBErrorCalculator:
    """
    Class for calculating errors and sample weights in boosting algorithms.
    This class supports different error types and sample schemes for flexibility in boosting implementations.
    """

    def __init__(self, error_type="absolute", sample_scheme="linear"):
        self.error_type = error_type
        self.sample_scheme = sample_scheme

        self.error_type_dict = {
            "absolute": lambda x: np.abs(x),
            "squared": lambda x: np.square(x),
        }

        self.sample_scheme_dict = {
            "linear": lambda x: np.abs(x),
            "exponential": lambda x: np.exp(x),
        }

    def calc_error(self, predictions, y):
        """
        Calculate the error between predictions and true labels.

        Args:
            predictions (np.ndarray): Predicted values
            y (np.ndarray): Ground truth labels

        Returns:
            np.ndarray: Computed error
        """
        differences = predictions - y

        if self.error_type not in self.error_type_dict:
            raise ValueError(
                f"Unsupported error type: {self.error_type}. "
                f"Supported types are: {list(self.error_type_dict.keys())}"
            )

        return self.error_type_dict[self.error_type](differences)

    def compute_sample_weights(self, errors):
        """
        Compute sample weights from errors.

        Args:
            errors (np.ndarray): Error values

        Returns:
            np.ndarray: Normalised sample weights
        """
        if self.sample_scheme not in self.sample_scheme_dict:
            raise ValueError(
                f"Unsupported sample scheme: {self.sample_scheme}. "
                f"Supported schemes are: {list(self.sample_scheme_dict.keys())}"
            )

        weights = self.sample_scheme_dict[self.sample_scheme](errors) + 1e-10
        return weights / np.sum(weights)
